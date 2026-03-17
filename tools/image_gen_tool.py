"""
Image Generation Tool
=====================
Generates a cartoon-style illustration for a news article using Google Imagen 3.
Saves the image locally and returns the file path.
"""

import os
from pathlib import Path
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class ImageGenInput(BaseModel):
    article_title: str = Field(description="The title of the news article")
    article_summary: str = Field(description="A brief summary of the article content (1-3 sentences)")


class GenerateImageTool(BaseTool):
    name: str = "Generate Article Image"
    description: str = (
        "Generates a cartoon-style illustration for a news article using Google Imagen 3. "
        "The image features a cheerful cartoon character with sunglasses pointing at the topic. "
        "Returns the local file path of the saved PNG image."
    )
    args_schema: type[BaseModel] = ImageGenInput

    def _run(self, article_title: str, article_summary: str) -> str:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "ERROR: GOOGLE_API_KEY not set."

        prompt = (
            "Cartoon style, bright vivid colors, flat illustration, white background. "
            "A cheerful small cartoon character wearing cool round sunglasses, "
            "pointing enthusiastically with one hand at a large colorful infographic panel "
            "that visually explains the main topic. "
            "The character is friendly and expressive, standing at the bottom-left corner. "
            "The infographic panel fills most of the image with icons, simple charts or visuals "
            "related to the topic. Clean, modern, engaging design. Absolutely no text or words "
            "anywhere in the image. "
            f"Topic to illustrate: {article_title}. "
            f"Context: {article_summary[:400]}"
        )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_images(
                model="imagen-3.0-generate-002",
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio="1:1",
                    safety_filter_level="BLOCK_ONLY_HIGH",
                ),
            )

            # Save image to disk
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)

            safe_title = (
                "".join(c if c.isalnum() or c in " -_" else "" for c in article_title)
                [:50].strip().replace(" ", "_")
            )
            image_path = images_dir / f"{safe_title}.png"
            image_bytes = response.generated_images[0].image.image_bytes
            image_path.write_bytes(image_bytes)

            print(f"[ImageGen] Saved image: {image_path}")
            return str(image_path)

        except ImportError:
            return "ERROR: google-genai not installed. Run: pip install google-genai"
        except Exception as e:
            return f"ERROR generating image: {e}"
