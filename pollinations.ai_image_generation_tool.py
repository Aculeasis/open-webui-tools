"""
title: Pollinations.AI image generation (free)
author: Your Name
git_url: https://github.com/Aculeasis/llm-trash
description: AI image generation with Pollinations.AI API: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#text-to-image-get-%EF%B8%8F
requirements: httpx[socks]
version: 1.0
licence: MIT
"""

import random
import re
from urllib.parse import quote
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

_CACHED_SEEDS = {}


class Tools:
    class Valves(BaseModel):
        """Pydantic model for storing settings."""
        MODEL: Literal[
            "flux",
            "turbo",
        ] = Field(
            default="flux",
            description="Select a model for image generation"
        )
        SIZE: ResolutionStringLiteral = Field(
            default=_AVAILABLE_RESOLUTION_STRINGS[0], description="Select the image size (WxH) for generation"
        )
        ENHANCE: bool = Field(
            default=False,
            description="Enhance the prompt using provider's LLM for more detail.",
        )
        SEED_CACHE: bool = Field(
            default=True,
            description="Allow LLM use previous seed to 'edit' images",
        )
        PROXY: str = Field(
            default="",
            description="Proxy",
        )
        API_URL: str = Field(
            default="https://image.pollinations.ai/prompt",
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

    async def generate_image_pollinations(
        self,
        prompt: str,
        previous_seed: bool,
        __request__: Request,
        __user__: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]],
    ) -> str:
        """
        Generates an image via Pollinations.AI from a prompt. The image is sent directly to the UI via events. This function returns a string confirming the outcome for the LLM.
        Args:
            prompt: An English text prompt for image generation. It must be descriptive (~50-100 words); enrich brief prompts with details (mood, style, color, perspective, media type, etc) using adjectives. Use effective prompting techniques (e.g, for FLUX).
            previous_seed: If True, the seed from the previous generation will be used. This is useful when the user wants to get the same thing but with some modifications. In other cases, or if the user is very dissatisfied with the results, it should not be enabled.
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

        if self.valves.SEED_CACHE:
            seed = _cache(__user__['id'], previous_seed)
        else:
            seed = None
        width, height = self.valves.SIZE.split('x')  # pylint: disable=no-member
        width, height = int(width), int(height)
        url = f"{self.valves.API_URL}/{quote(prompt)}"
        params = {
            "width": width,
            "height": height,
            "model": self.valves.MODEL,
            "enhance": self.valves.ENHANCE,
            "nologo": True,
            "safe": False,
            "private": True,
        }
        if seed is not None:
            params["seed"] = seed

        await send_status("Image generation started")
        try:
            async with httpx.AsyncClient(
                    proxy=self.valves.PROXY or None, timeout=httpx.Timeout(15.0, read=600.0)
            ) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                if not (content_type := response.headers.get("Content-Type", "image/jpeg")).startswith("image/"):
                    content_type = "image/jpeg"
                if not (image_data := response.content):
                    raise ValueError("Images not found in response")

            await send_status("Saving image started")
            data = {
                "instances": {"prompt": prompt},
                "parameters": {
                    "sampleCount": 1,
                    "outputOptions": {"mimeType": content_type},
                },
            }
            url = upload_image(
                __request__,
                data,
                image_data,
                content_type,
                user=Users.get_user_by_id(__user__["id"]),
            )

            emo_seed = "" if seed is None else f" | {_seed_to_emojis(seed)}"
            await send_status(f"Image generation finished | {self.valves.MODEL}{emo_seed}", True)

            # This is for LLM, we hide the prompt in an image tag. I wish LLM will understand it :)
            msg_for_llm = f"prompt: '{_remove_markdown_special_chars(prompt)}'"
            await send_message(f"![{msg_for_llm}]({url}) ")

            msg = ""
            if self.valves.DEBUG:
                msg = f'params: "{params}"\n'
            if self.valves.SHOW_PROMPT:
                msg += f'Prompt: "{prompt}"\n'
            if msg:
                await send_message(msg)
            return "Image generated and displayed above. The user sees it already. Provide follow-up text only. Do not post any image tags or links, they will be added by the system."
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


def _cache(user_id: int, previous_seed: bool) -> int:
    if previous_seed and (old_seed := _CACHED_SEEDS.get(user_id)) is not None:
        return old_seed
    seed = _seed()
    if user_id:
        _CACHED_SEEDS[user_id] = seed
    return seed


def _seed() -> int:
    # https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html
    min_val = -0x8000_0000_0000_0000  # -2^63
    max_val = 0xffff_ffff_ffff_ffff   # 2^64 - 1
    result = random.randint(min_val, max_val)
    return result if result > -1 else max_val + result


def _seed_to_emojis(seed: int, num_emojis: int = 4) -> str:
    """Converts a seed integer into a string of emojis to visually represent it."""
    emoji_palette = [
        "🎲", "✨", "🎨", "🖼️", "💡", "🌟", "🌀", "🍀", "🎉", "🔮",
        "💎", "🧩", "🔑", "🎁", "🎈", "🌈", "🚀", "⏳", "💫", "🌠"
    ]  # Using 20 emojis for more variety

    emojis = []

    # For each emoji, take a different "slice" of the seed number using bitwise operations.
    # This helps ensure that changes in the seed are likely to change the emojis.
    for i in range(num_emojis):
        # Extract an 8-bit chunk (0-255) from the seed.
        # Each iteration uses a different part of the seed by shifting.
        # (i * 8) means we shift by 0, 8, 16, 24 bits for the first 4 emojis.
        chunk = (seed >> (i * 8)) & 0xFF
        index = chunk % len(emoji_palette)
        emojis.append(emoji_palette[index])

    return "".join(emojis)
