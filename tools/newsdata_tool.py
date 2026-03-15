"""
NewsData.io Tool
================
Fetches latest news articles from NewsData.io API.
Returns structured list of articles with title, description, and URL.
"""

import os
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional


class NewsDataInput(BaseModel):
    query: str = Field(description="Search query or topic for news")
    language: str = Field(default="en", description="Language code (e.g. 'en')")
    country: Optional[str] = Field(default=None, description="Country code (e.g. 'us')")
    max_results: int = Field(default=5, description="Number of articles to fetch (max 10)")


class NewsDataTool(BaseTool):
    name: str = "NewsData Fetcher"
    description: str = (
        "Fetches the latest news articles from NewsData.io based on a query. "
        "Returns article titles, descriptions, URLs, and publication dates."
    )
    args_schema: type[BaseModel] = NewsDataInput

    def _run(self, query: str, language: str = "en", country: Optional[str] = None, max_results: int = 5) -> str:
        api_key = os.environ.get("NEWSDATA_API_KEY")
        if not api_key:
            return "ERROR: NEWSDATA_API_KEY not set in environment."

        params = {
            "apikey": api_key,
            "q": query,
            "language": language,
            "size": min(max_results, 10),
        }
        if country:
            params["country"] = country

        try:
            response = requests.get("https://newsdata.io/api/1/latest", params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            articles = data.get("results", [])
            if not articles:
                return f"No articles found for query: '{query}'"

            output_lines = [f"Found {len(articles)} articles for '{query}':\n"]
            for i, article in enumerate(articles[:max_results], 1):
                title = article.get("title", "No title")
                description = article.get("description") or article.get("content", "")
                if description and len(description) > 300:
                    description = description[:300] + "..."
                url = article.get("link", "")
                source = article.get("source_id", "Unknown source")
                pub_date = article.get("pubDate", "")

                output_lines.append(
                    f"[{i}] {title}\n"
                    f"    Source: {source} | Published: {pub_date}\n"
                    f"    {description}\n"
                    f"    URL: {url}\n"
                )

            return "\n".join(output_lines)

        except requests.RequestException as e:
            return f"ERROR fetching news: {e}"
