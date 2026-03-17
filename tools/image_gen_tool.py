"""
Image Generation Tool
=====================
Generates a cartoon-style illustration for a news article using Microsoft Bing Image Creator.
Requires BING_AUTH_COOKIE in .env (the _U cookie from bing.com after logging in).
Saves the image locally and returns the file path.
"""

import os
import requests
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ImageGenInput(BaseModel):
    article_title: str = Field(description="The title of the news article")
    article_summary: str = Field(description="A brief summary of the article content (1-3 sentences)")


class GenerateImageTool(BaseTool):
    name: str = "Generate Article Image"
    description: str = (
        "Generates a cartoon-style illustration for a news article using Microsoft Bing Image Creator. "
        "The image features a cheerful cartoon character with sunglasses pointing at the topic. "
        "Returns the local file path of the saved PNG image."
    )
    args_schema: type[BaseModel] = ImageGenInput

    def _run(self, article_title: str, article_summary: str) -> str:
        auth_cookie = os.environ.get("BING_AUTH_COOKIE")
        if not auth_cookie:
            return (
                "ERROR: BING_AUTH_COOKIE not set. "
                "Go to bing.com/images/create, log in, open DevTools → Application → Cookies, "
                "copy the '_U' cookie value and add it to .env as BING_AUTH_COOKIE."
            )

        prompt = (
            "Cartoon style, bright vivid colors, flat illustration, white background. "
            "A cheerful small cartoon character wearing cool round sunglasses, "
            "pointing enthusiastically with one hand at a large colorful infographic panel "
            "that visually explains the main topic. "
            "The character is friendly and expressive, standing at the bottom-left corner. "
            "The infographic panel fills most of the image with icons and simple visuals "
            "related to the topic. Clean, modern, engaging design. No text or words in the image. "
            f"Topic: {article_title}. Context: {article_summary[:300]}"
        )

        try:
            from BingImageCreator import ImageGen

            gen = ImageGen(auth_cookie=auth_cookie, quiet=True)
            image_urls = gen.get_images(prompt)

            if not image_urls:
                return "ERROR generating image: Bing returned no images."

            # Download the first image
            response = requests.get(image_urls[0], timeout=30)
            response.raise_for_status()

            # Save image to disk
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)

            safe_title = (
                "".join(c if c.isalnum() or c in " -_" else "" for c in article_title)
                [:50].strip().replace(" ", "_")
            )
            image_path = images_dir / f"{safe_title}.png"
            image_path.write_bytes(response.content)

            print(f"[ImageGen] Saved image: {image_path}")
            return str(image_path)

        except ImportError:
            return "ERROR: BingImageCreator not installed. Run: pip install BingImageCreator"
        except Exception as e:
            return f"ERROR generating image: {e}"
