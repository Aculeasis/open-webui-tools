"""
title: Together.ai image restyle (paid only)
author:
git_url:
description: AI image restyle with Together.ai FLUX API https://docs.together.ai/reference/post_images-generations
requirements: httpx[socks]
version: 1.0
licence: MIT
"""

import base64
import re
from typing import Awaitable, Callable, Literal

import httpx
from fastapi import Request
from open_webui.models.users import Users
from open_webui.routers.images import upload_image
from pydantic import BaseModel, Field

_AVAILABLE_RESOLUTION_STRINGS = (
    "1024x1024",  # square
    "1024x768",   # landscape
    "1440x1024",  # landscape_large
    "768x1024",   # portrait
    "1024x1440",  # portrait_large
    "512x512",    # small - not good
    "256x256",    # tiny - testing only
)
ResolutionStringLiteral = Literal[*_AVAILABLE_RESOLUTION_STRINGS]


class Tools:
    class Valves(BaseModel):
        """Pydantic model for storing settings."""
        API_KEY: str = Field(
            default="", description="Your Together.ai API key"
        )
        FLUX_MODEL: Literal[
            "FLUX.1-depth | $0.025",
            "FLUX.1-canny | $0.025",
        ] = Field(
            default="FLUX.1-depth | $0.025",
            description="Select the FLUX model for image generation"
        )
        SIZE: ResolutionStringLiteral = Field(
            default=_AVAILABLE_RESOLUTION_STRINGS[0], description="Select the image size (WxH) for generation"
        )
        STEPS: int = Field(
            default=28, description="Maximum 28; 0 - provided by LLM"
        )
        VARIANTS: int = Field(
            default=1,
            description="Image variant count; 0 - provided by LLM",
        )
        PROXY: str = Field(
            default="",
            description="Proxy",
        )
        API_URL: str = Field(
            default="https://api.together.xyz/v1/images/generations",
            description="URL for the API",
        )
        SHOW_PROMPT: bool = Field(
            default=False,
            description="Show the prompt in a message",
        )
        DEBUG: bool = Field(
            default=False,
            description="Send detailed information to the chat",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def edit_image_together(
        self,
        prompt: str,
        negative_prompt: str,
        image_variants: int,
        generation_steps: int,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> str:
        """
        Restyles or edits an image via Together.ai using FLUX from a prompt and the structure of a given input image. The image(s) are sent directly to the UI via events. The user provides the input image. This function returns a string confirming the outcome for the LLM.
        Args:
            prompt: An English text prompt for image generation. It must be descriptive (~50-100 words); enrich brief prompts with details (mood, style, color, perspective, media type, etc) using adjectives. Use effective prompting techniques (e.g, for FLUX).
            negative_prompt: A comma-separated list of terms to exclude from the image, English.
            image_variants: How many image variants should be generated. If the user does not mention it, use 1
            generation_steps: More steps mean better quality, but the price and time will be higher. If the user does not mention it, use 28
        Returns:
            A string summarizing the operation's outcome
        """
        async def send_status(msg_: str, done_: bool = False):
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": msg_,
                        "status": "complete" if done_ else "in_progress",
                        "done": done_,
                    },
                }
            )
        async def send_message(msg_: str):
            await __event_emitter__(
                {
                    "type": "message",
                    "data": {"content": msg_},
                }
            )

        width, height = self.valves.SIZE.split('x')  # pylint: disable=no-member
        width, height = int(width), int(height)
        model_name = self.valves.FLUX_MODEL.split(' ', 1)[0]  # pylint: disable=no-member # remove price from model name

        json_ = {
            "model": f"black-forest-labs/{model_name}",
            "prompt": prompt,
            "steps": self.valves.STEPS or generation_steps,
            "width": width,
            "height": height,
            "response_format": "base64",
            "n": self.valves.VARIANTS or image_variants,
        }
        if negative_prompt:
            json_["negative_prompt"] = negative_prompt

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.valves.API_KEY}"
        }
        await send_status("Image extraction started")
        try:
            # extract image from request
            # {'messages': [
            # {'role': 'system', 'content': 'User Context:'},
            # {'role': 'user', 'content': [{'type': 'text', 'text': '...'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,iVBORw0KGg...'}}]}]}
            body = await __request__.json()
            image_url = None
            try:
                for message in body["messages"]:
                    if image_url:
                        break
                    if message["role"] == "user" and isinstance(items := message["content"], list):
                        for item in items:
                            if item["type"] == "image_url":
                                image_url = item["image_url"]["url"]
                                break
            except Exception as e:
                raise ValueError(f"Incorrect request structure: {e}") from e
            if not image_url:
                raise ValueError("User did not provide an image")
            json_["image_url"] = image_url

            await send_status("Image editing started")
            async with httpx.AsyncClient(
                    proxy=self.valves.PROXY or None, timeout=httpx.Timeout(15.0, read=600.0)
            ) as client:
                response = await client.post(self.valves.API_URL, headers=headers, json=json_)
                response.raise_for_status()
                result = response.json()
            if "error" in result:
                raise httpx.HTTPStatusError(str(result["error"]), request=response.request, response=response)

            images_list = []
            try:
                for k in result["data"]:
                    if "b64_json" in k:
                        content_type = f"image/{_get_img_extension(k['b64_json'])}"
                        images_list.append({
                            "content_type": content_type,
                            "image_data": base64.b64decode(k["b64_json"]),
                            "data": {
                                "instances": {"prompt": prompt},
                                "parameters": {
                                    "sampleCount": 1,
                                    "outputOptions": {"mimeType": content_type},
                                }
                            }
                        })
            except Exception as e:
                raise ValueError(f"Incorrect response structure: {e}") from e
            if not images_list:
                raise ValueError("Images not found in response")

            await send_status(f"Saving {len(images_list)} image(s) started")
            url_list = []
            for item in images_list:
                url_list.append(upload_image(
                    __request__,
                    item["data"],
                    item["image_data"],
                    item["content_type"],
                    user=Users.get_user_by_id(__user__["id"]),
                ))

            count_ = f" | {len(url_list)}" if len(url_list) > 1 else ""
            await send_status(f"Image editing finished{count_} | {model_name}", True)

            # This is for LLM, we hide the prompt in an image tag. I wish LLM will understand it :)
            cleaned_prompt = _remove_markdown_special_chars(prompt)
            cleaned_negative_prompt = _remove_markdown_special_chars(negative_prompt)
            msg_for_llm = f"prompt: '{cleaned_prompt}'"
            msg_for_llm += f", negative_prompt: '{cleaned_negative_prompt}'" if cleaned_negative_prompt else ""
            # We don't want to add prompt in multiple images, only in the first.
            result = f"![{msg_for_llm}]({url_list[0]}) "
            if len(url_list) > 1:
                result += " ".join(f"![IMG {i}]({url})" for i, url in enumerate(url_list[1:], start=2)) + " "
            await send_message(result)

            msg = ""
            if self.valves.DEBUG:
                msg = f'json: "{json_}"\n'
            if self.valves.SHOW_PROMPT:
                msg += f'Prompt: "{prompt}"\n'
                msg += f'Negative Prompt: "{negative_prompt}"\n' if negative_prompt else ""
            if msg:
                await send_message(msg)
            return f"{len(url_list)} image(s) generated and displayed above. The user sees them already. Provide follow-up text only. Do not post any image tags or links, they will be added by the system."
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            msg = str(e) or str(type(e))
            await send_status(f"Network error: {msg}", True)
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    e.response.json()
                except Exception:
                    msg = e.response.text[:1000]
                else:
                    msg = f"```json\n{e.response.text}\n```"
                if self.valves.DEBUG:
                    await send_message(f"{msg}\n")
            return f"Tell the user that a network error occurred and the image generation was not successful: {msg}"
        except Exception as e:
            await send_status(f"Error: {str(e)}", True)
            return f"Tell the user that an error occurred and the image generation was not successful: {e}"


def _remove_markdown_special_chars(text: str) -> str:
    """Removes markdown special characters from a string."""
    # Characters to remove: \ ` * _ { } [ ] ( ) # + - . !
    return re.sub(r"[\\`*_{}\[\]()#+-.!]", "", text) if text else text


def _get_img_extension(img_data: str) -> str:
    if img_data.startswith("/9j/"):
        return "jpeg"
    elif img_data.startswith("iVBOR"):
        return "png"
    elif img_data.startswith("R0lG"):
        return "gif"
    elif img_data.startswith("UklGR"):
        return "webp"
    return "jpeg"
