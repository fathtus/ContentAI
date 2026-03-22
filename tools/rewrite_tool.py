"""
Rewrite Tool
=============
Uses Mistral-7B-Instruct via the HuggingFace Inference API to rewrite
a raw news summary into a platform-optimized social media post.

Requires HF_API_TOKEN in .env.
"""

import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"

PLATFORM_INSTRUCTIONS = {
    "x":         "Twitter/X post: max 280 characters, punchy hook, 1-2 hashtags, include source URL at the end.",
    "facebook":  "Facebook post: conversational tone, 2-4 sentences, engaging question or call-to-action, include source URL.",
    "instagram": "Instagram caption: visual-first, emojis, 5-10 relevant hashtags, 'Source: [URL]' at the end.",
    "linkedin":  "LinkedIn post: professional thought-leadership tone, insight-driven, 3-5 sentences, include source URL.",
}


class RewritePostInput(BaseModel):
    raw_content: str = Field(description="The raw news article text or summary to rewrite")
    platform: str = Field(description="Target platform: x, facebook, instagram, or linkedin")
    source_url: str = Field(default="", description="Source URL to include in the post")


class RewritePostTool(BaseTool):
    name: str = "Rewrite Post for Platform"
    description: str = (
        "Rewrites a raw news summary into a polished, platform-optimized social media post "
        "using Mistral-7B. Supported platforms: x, facebook, instagram, linkedin."
    )
    args_schema: type[BaseModel] = RewritePostInput

    def _run(self, raw_content: str, platform: str, source_url: str = "") -> str:
        hf_token = os.environ.get("HF_API_TOKEN")
        if not hf_token:
            return f"ERROR: HF_API_TOKEN not set. Cannot rewrite post for {platform}."

        instruction = PLATFORM_INSTRUCTIONS.get(platform.lower())
        if not instruction:
            return f"ERROR: Unknown platform '{platform}'. Use: x, facebook, instagram, linkedin."

        url_hint = f"\nSource URL: {source_url}" if source_url else ""
        messages = [
            {
                "role": "system",
                "content": "You are a social media expert. Write only the final post — no explanation, no preamble.",
            },
            {
                "role": "user",
                "content": (
                    f"Rewrite the following news content as a {platform} post.\n"
                    f"Rules: {instruction}{url_hint}\n\n"
                    f"News content:\n{raw_content}"
                ),
            },
        ]

        headers = {"Authorization": f"Bearer {hf_token}"}
        payload = {
            "model": HF_MODEL,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
        }

        try:
            response = requests.post(HF_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except requests.RequestException as e:
            return f"ERROR rewriting post: {e}"
