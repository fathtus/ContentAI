"""
Social Media Posting Tools
===========================
Tools for posting content to X (Twitter), Facebook, Instagram, and LinkedIn.
Each tool expects the relevant API credentials to be set in .env.
"""

import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── Input Schemas ─────────────────────────────────────────────────────────────

class PostInput(BaseModel):
    content: str = Field(description="The text content to post")


class FacebookPostInput(BaseModel):
    content: str = Field(description="The text content to post")
    image_path: str = Field(
        default="",
        description="Optional local file path to a PNG/JPG image to attach to the post",
    )


class InstagramPostInput(BaseModel):
    content: str = Field(description="The caption text for the Instagram post")
    image_url: str = Field(
        default="",
        description="Public URL of image to attach (required for Instagram feed posts)"
    )


# ── X (Twitter) Tool ─────────────────────────────────────────────────────────

class PostToXTool(BaseTool):
    name: str = "Post to X (Twitter)"
    description: str = "Posts a tweet/thread to X (Twitter). Content must be ≤280 characters."
    args_schema: type[BaseModel] = PostInput

    def _run(self, content: str) -> str:
        bearer_token       = os.environ.get("X_BEARER_TOKEN")
        api_key            = os.environ.get("X_API_KEY")
        api_secret         = os.environ.get("X_API_SECRET")
        access_token       = os.environ.get("X_ACCESS_TOKEN")
        access_token_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")

        if not all([api_key, api_secret, access_token, access_token_secret]):
            return (
                "SIMULATED POST TO X:\n"
                f"{content}\n\n"
                "[Set X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET in .env to enable real posting]"
            )

        try:
            import tweepy
            client = tweepy.Client(
                bearer_token=bearer_token,
                consumer_key=api_key,
                consumer_secret=api_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
            )
            response = client.create_tweet(text=content[:280])
            tweet_id = response.data["id"]
            return f"Successfully posted to X! Tweet ID: {tweet_id}"
        except ImportError:
            return "ERROR: tweepy not installed. Run: pip install tweepy"
        except Exception as e:
            return f"ERROR posting to X: {e}"


# ── Facebook Tool ─────────────────────────────────────────────────────────────

class PostToFacebookTool(BaseTool):
    name: str = "Post to Facebook"
    description: str = (
        "Posts content to a Facebook Page. "
        "If image_path is provided, the post will include the image as a photo post. "
        "image_path must be a local file path to a PNG or JPG image."
    )
    args_schema: type[BaseModel] = FacebookPostInput

    def _run(self, content: str, image_path: str = "") -> str:
        page_id    = os.environ.get("FACEBOOK_PAGE_ID")
        page_token = os.environ.get("FACEBOOK_PAGE_TOKEN")

        if not all([page_id, page_token]):
            return (
                "SIMULATED POST TO FACEBOOK:\n"
                f"{content}\n"
                f"Image: {image_path or '(no image)'}\n\n"
                "[Set FACEBOOK_PAGE_ID and FACEBOOK_PAGE_TOKEN in .env to enable real posting]"
            )

        try:
            # ── Photo post (with image) ────────────────────────────────────
            if image_path and os.path.isfile(image_path):
                url = f"https://graph.facebook.com/v25.0/{page_id}/photos"
                with open(image_path, "rb") as img_file:
                    files = {
                        "source":       (os.path.basename(image_path), img_file, "image/png"),
                        "message":      (None, content),
                        "access_token": (None, page_token),
                    }
                    response = requests.post(url, files=files, timeout=30)
            else:
                # ── Text-only post ─────────────────────────────────────────
                url = f"https://graph.facebook.com/v25.0/{page_id}/feed"
                files = {
                    "message":      (None, content),
                    "access_token": (None, page_token),
                }
                response = requests.post(url, files=files, timeout=15)

            if not response.ok:
                detail = response.json().get("error", {})
                return (
                    f"ERROR posting to Facebook: {response.status_code} — "
                    f"{detail.get('message', response.text)}"
                )
            post_id = response.json().get("id", response.json().get("post_id", "unknown"))
            return f"Successfully posted to Facebook! Post ID: {post_id}"
        except requests.RequestException as e:
            return f"ERROR posting to Facebook: {e}"


# ── Instagram Tool ────────────────────────────────────────────────────────────

class PostToInstagramTool(BaseTool):
    name: str = "Post to Instagram"
    description: str = (
        "Posts a photo with caption to an Instagram Business account via the Facebook Graph API. "
        "An image_url is required for feed posts."
    )
    args_schema: type[BaseModel] = InstagramPostInput

    def _run(self, content: str, image_url: str = "") -> str:
        ig_user_id  = os.environ.get("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        page_token  = os.environ.get("FACEBOOK_PAGE_TOKEN")

        if not all([ig_user_id, page_token]):
            return (
                "SIMULATED POST TO INSTAGRAM:\n"
                f"Caption: {content}\n"
                f"Image: {image_url or '(no image)'}\n\n"
                "[Set INSTAGRAM_BUSINESS_ACCOUNT_ID and FACEBOOK_PAGE_TOKEN in .env to enable real posting]"
            )

        if not image_url:
            return "ERROR: Instagram feed posts require an image_url."

        try:
            # Step 1: Create media container
            container_url = f"https://graph.facebook.com/v25.0/{ig_user_id}/media"
            container_payload = {
                "image_url": image_url,
                "caption": content,
                "access_token": page_token,
            }
            container_res = requests.post(container_url, data=container_payload, timeout=15)
            container_res.raise_for_status()
            creation_id = container_res.json().get("id")

            # Step 2: Publish the container
            publish_url = f"https://graph.facebook.com/v25.0/{ig_user_id}/media_publish"
            publish_payload = {"creation_id": creation_id, "access_token": page_token}
            publish_res = requests.post(publish_url, data=publish_payload, timeout=15)
            publish_res.raise_for_status()
            media_id = publish_res.json().get("id", "unknown")
            return f"Successfully posted to Instagram! Media ID: {media_id}"
        except requests.RequestException as e:
            return f"ERROR posting to Instagram: {e}"


# ── LinkedIn Tool ─────────────────────────────────────────────────────────────

class PostToLinkedInTool(BaseTool):
    name: str = "Post to LinkedIn"
    description: str = "Posts content to a LinkedIn page or personal profile."
    args_schema: type[BaseModel] = PostInput

    def _run(self, content: str) -> str:
        access_token = os.environ.get("LINKEDIN_ACCESS_TOKEN")
        author_urn   = os.environ.get("LINKEDIN_AUTHOR_URN")  # urn:li:person:xxx or urn:li:organization:xxx

        if not all([access_token, author_urn]):
            return (
                "SIMULATED POST TO LINKEDIN:\n"
                f"{content}\n\n"
                "[Set LINKEDIN_ACCESS_TOKEN and LINKEDIN_AUTHOR_URN in .env to enable real posting]"
            )

        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
        }
        payload = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": content},
                    "shareMediaCategory": "NONE",
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            response.raise_for_status()
            post_id = response.headers.get("x-restli-id", "unknown")
            return f"Successfully posted to LinkedIn! Post ID: {post_id}"
        except requests.RequestException as e:
            return f"ERROR posting to LinkedIn: {e}"
