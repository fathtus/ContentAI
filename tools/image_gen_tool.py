"""
Image Generation Tool
=====================
Generates a cartoon-style illustration for a news article.

Providers (tried in order):
  1. Hugging Face Inference API — free with HF_API_TOKEN (huggingface.co → Settings → Access Tokens)
     Model: black-forest-labs/FLUX.1-schnell
  2. Pollinations.ai — completely free, no auth needed (fallback when HF not configured)

Saves the image locally and returns the file path.
"""

import os
import time
import requests
from pathlib import Path
from urllib.parse import quote
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

HF_MODEL = "black-forest-labs/FLUX.1-schnell"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"


class ImageGenInput(BaseModel):
    article_title: str = Field(description="The title of the news article")
    article_summary: str = Field(description="A brief summary of the article content (1-3 sentences)")


class GenerateImageTool(BaseTool):
    name: str = "Generate Article Image"
    description: str = (
        "Generates a cartoon-style illustration for a news article. "
        "Uses Hugging Face FLUX.1-schnell (free) or Pollinations.ai as fallback. "
        "Returns the local file path of the saved PNG image."
    )
    args_schema: type[BaseModel] = ImageGenInput

    def _run(self, article_title: str, article_summary: str) -> str:
        prompt = (
            "cartoon flat illustration, bright vivid colors, white background, "
            "cheerful small cartoon character wearing cool round sunglasses "
            "pointing enthusiastically at a large colorful infographic panel, "
            "friendly expressive character at bottom-left corner, "
            "infographic panel with icons and simple visuals about the topic, "
            "clean modern design, absolutely no text no words no letters, "
            f"topic: {article_title[:100]}"
        )

        hf_token = os.environ.get("HF_API_TOKEN")

        if hf_token:
            result = self._generate_huggingface(prompt, article_title, hf_token)
        else:
            result = self._generate_pollinations(prompt, article_title)

        return result

    def _generate_huggingface(self, prompt: str, title: str, token: str) -> str:
        print(f"[ImageGen] Using Hugging Face FLUX.1-schnell for: {title}")
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": 1024, "height": 1024, "num_inference_steps": 4},
        }

        for attempt in range(3):
            try:
                response = requests.post(HF_URL, headers=headers, json=payload, timeout=120)

                # Model loading — wait and retry
                if response.status_code == 503:
                    wait = response.json().get("estimated_time", 20)
                    print(f"[ImageGen] Model loading, waiting {wait:.0f}s...")
                    time.sleep(min(wait, 30))
                    continue

                response.raise_for_status()

                if "image" not in response.headers.get("content-type", ""):
                    return f"ERROR: Unexpected content-type: {response.headers.get('content-type')}"

                return self._save_image(response.content, title)

            except requests.RequestException as e:
                if attempt == 2:
                    return f"ERROR (Hugging Face): {e}"
                time.sleep(5)

        return "ERROR: Hugging Face failed after 3 attempts."

    def _generate_pollinations(self, prompt: str, title: str) -> str:
        print(f"[ImageGen] Using Pollinations.ai for: {title}")
        # Use + instead of %20 for shorter, cleaner URLs
        short_prompt = prompt[:300].replace(",", "").replace("  ", " ")
        encoded = quote(short_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"

        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()

            if "image" not in response.headers.get("content-type", ""):
                return f"ERROR: Pollinations returned non-image. Try setting HF_API_TOKEN instead."

            return self._save_image(response.content, title)

        except Exception as e:
            return (
                f"ERROR (Pollinations): {e}\n"
                "Tip: Set HF_API_TOKEN in .env for a more reliable image provider. "
                "Get a free token at huggingface.co → Settings → Access Tokens."
            )

    def _save_image(self, content: bytes, title: str) -> str:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)

        safe_title = (
            "".join(c if c.isalnum() or c in " -_" else "" for c in title)
            [:50].strip().replace(" ", "_")
        )
        image_path = images_dir / f"{safe_title}.png"
        image_path.write_bytes(content)

        print(f"[ImageGen] Saved: {image_path} ({len(content):,} bytes)")
        return str(image_path)
