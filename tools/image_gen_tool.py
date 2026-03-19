"""
Image Generation Tools
======================
Two tools:
  - GenerateImageTool          : artistic scene, person with laptop (Page 1)
  - GenerateProfessionalImageTool : profession-based person matching the topic (Page 2)

Providers (tried in order):
  1. Hugging Face Inference API — free with HF_API_TOKEN
  2. Pollinations.ai            — no auth fallback
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
        import random
        weather_scenes = [
            "sunny tropical beach with palm trees and golden light",
            "misty rainy forest with colorful umbrella and puddles",
            "snowy mountain peak with cozy scarf and snowflakes",
            "stormy cliffside with dramatic clouds and lightning in the distance",
            "golden autumn countryside with falling leaves and warm fog",
            "desert at sunset with sand dunes and orange sky",
            "magical cherry blossom park in spring breeze",
            "cozy cabin porch during a heavy snowstorm with hot cocoa nearby",
            "lush green Irish countryside under dramatic rainbow after rain",
            "volcanic island at dusk with glowing lava and purple sky",
            "bamboo forest in morning mist with soft golden rays",
            "arctic tundra under shimmering northern lights aurora borealis",
            "mediterranean rooftop terrace overlooking the sea at golden hour",
            "flooded venetian street during a warm summer rain",
            "savanna at sunrise with silhouette of acacia trees and wildlife",
            "himalayan valley blanketed in thick fog and snow",
            "tropical thunderstorm over a rice paddy field at night",
        ]
        scene = random.choice(weather_scenes)
        prompt = (
            f"artistic digital painting, rich vivid colors, cinematic lighting, "
            f"a person sitting outdoors working on a laptop, "
            f"scenic natural setting: {scene}, "
            "floating above the character are glowing semi-transparent chat bubble screens "
            "with blurred chat message lines and interface elements, "
            "holographic UI panels hovering in the air, soft glow effect, futuristic mood, "
            "impressionist brushwork, painterly textures, highly detailed, no readable text"
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

    def _save_image(self, content: bytes, title: str, suffix: str = "") -> str:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)

        safe_title = (
            "".join(c if c.isalnum() or c in " -_" else "" for c in title)
            [:50].strip().replace(" ", "_")
        )
        image_path = images_dir / f"{safe_title}{suffix}.png"
        image_path.write_bytes(content)

        print(f"[ImageGen] Saved: {image_path} ({len(content):,} bytes)")
        return str(image_path)


# ── Topic → Profession mapping ────────────────────────────────────────────────

PROFESSION_MAP = [
    (["health", "medical", "medicine", "hospital", "doctor", "pharma", "biotech"],
     "a doctor in a white coat"),
    (["finance", "bank", "invest", "stock", "crypto", "economy", "market", "trading"],
     "a financial analyst in business attire"),
    (["tech", "ai", "software", "code", "robot", "machine learning", "data", "cyber"],
     "a software engineer"),
    (["law", "legal", "court", "justice", "regulation", "policy"],
     "a lawyer in formal attire"),
    (["education", "school", "university", "student", "teacher", "learning"],
     "a teacher"),
    (["environment", "climate", "energy", "green", "solar", "sustainable"],
     "an environmental scientist outdoors"),
    (["sport", "athlete", "football", "basketball", "tennis", "olympic"],
     "a professional athlete"),
    (["food", "chef", "restaurant", "culinary", "nutrition"],
     "a chef in a kitchen apron"),
    (["art", "music", "film", "culture", "creative", "design"],
     "a creative artist"),
    (["space", "nasa", "astronaut", "rocket", "satellite"],
     "an astronaut in a space suit"),
    (["politic", "government", "election", "president", "senator"],
     "a politician at a podium"),
    (["science", "research", "lab", "experiment", "biology", "chemistry", "physics"],
     "a scientist in a laboratory"),
]

PROFESSIONAL_SCENES = [
    "at a busy city office with floor-to-ceiling windows",
    "in a modern minimalist workspace with natural light",
    "at a rooftop terrace with a city skyline at dusk",
    "in a cozy library surrounded by books",
    "at a seaside cafe with the ocean in the background",
    "in a futuristic glass building lobby",
    "at a conference table with colleagues",
    "in a creative studio with exposed brick walls",
    "outdoors in a vibrant urban plaza",
    "in a high-tech control room with multiple screens",
    "at a standing desk in a loft apartment",
    "in a botanical garden during golden hour",
]


class ProfessionalImageInput(BaseModel):
    article_title: str = Field(description="The title of the news article")
    article_summary: str = Field(description="A brief summary of the article content")
    page_topic: str = Field(description="The overall topic of this Facebook page (e.g. 'healthcare', 'finance')")


class GenerateProfessionalImageTool(BaseTool):
    name: str = "Generate Professional Article Image"
    description: str = (
        "Generates an artistic image of a professional person whose role matches the page topic. "
        "The person is placed in a random professional scene. "
        "Returns the local file path of the saved PNG image."
    )
    args_schema: type[BaseModel] = ProfessionalImageInput

    def _run(self, article_title: str, article_summary: str, page_topic: str) -> str:
        import random

        # Infer profession from topic
        topic_lower = page_topic.lower() + " " + article_title.lower()
        profession = "a professional expert"
        for keywords, prof in PROFESSION_MAP:
            if any(kw in topic_lower for kw in keywords):
                profession = prof
                break

        scene = random.choice(PROFESSIONAL_SCENES)

        prompt = (
            f"artistic digital painting, rich vivid colors, cinematic lighting, "
            f"{profession} working confidently, "
            f"setting: {scene}, "
            "professional atmosphere, sharp focus on the person, "
            "soft bokeh background, highly detailed, photorealistic style, no text"
        )

        hf_token = os.environ.get("HF_API_TOKEN")
        suffix = "_p2"

        if hf_token:
            return self._generate_huggingface(prompt, article_title, hf_token, suffix)
        else:
            return self._generate_pollinations(prompt, article_title, suffix)

    def _generate_huggingface(self, prompt: str, title: str, token: str, suffix: str = "") -> str:
        print(f"[ImageGen-P2] Using Hugging Face FLUX.1-schnell for: {title}")
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "inputs": prompt,
            "parameters": {"width": 1024, "height": 1024, "num_inference_steps": 4},
        }

        for attempt in range(3):
            try:
                response = requests.post(HF_URL, headers=headers, json=payload, timeout=120)
                if response.status_code == 503:
                    wait = response.json().get("estimated_time", 20)
                    print(f"[ImageGen-P2] Model loading, waiting {wait:.0f}s...")
                    time.sleep(min(wait, 30))
                    continue
                response.raise_for_status()
                if "image" not in response.headers.get("content-type", ""):
                    return f"ERROR: Unexpected content-type: {response.headers.get('content-type')}"
                return self._save_image(response.content, title, suffix)
            except requests.RequestException as e:
                if attempt == 2:
                    return f"ERROR (Hugging Face P2): {e}"
                time.sleep(5)
        return "ERROR: Hugging Face P2 failed after 3 attempts."

    def _generate_pollinations(self, prompt: str, title: str, suffix: str = "") -> str:
        short_prompt = prompt[:300].replace(",", "").replace("  ", " ")
        encoded = quote(short_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            if "image" not in response.headers.get("content-type", ""):
                return "ERROR: Pollinations returned non-image."
            return self._save_image(response.content, title, suffix)
        except Exception as e:
            return f"ERROR (Pollinations P2): {e}"

    def _save_image(self, content: bytes, title: str, suffix: str = "") -> str:
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)
        safe_title = (
            "".join(c if c.isalnum() or c in " -_" else "" for c in title)
            [:50].strip().replace(" ", "_")
        )
        image_path = images_dir / f"{safe_title}{suffix}.png"
        image_path.write_bytes(content)
        print(f"[ImageGen-P2] Saved: {image_path} ({len(content):,} bytes)")
        return str(image_path)
