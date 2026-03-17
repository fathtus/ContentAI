"""
ContentAI — News-to-Social Media Pipeline
==========================================
Pipeline:
  1. Fetch latest news from NewsData.io
  2. Content Writer Agent  : Uses Google Gemini to rewrite each article
                             into platform-specific posts (X, Facebook, Instagram, LinkedIn)
  3. Image Creator Agent   : Generates a cartoon illustration per article using Google Imagen 3
  4. Publisher Agent       : Posts content + images to all 4 platforms

Usage:
  python main.py [--topic "AI news"] [--dry-run]
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import (
    NewsDataTool,
    GenerateImageTool,
    PostToXTool,
    PostToFacebookTool,
    PostToInstagramTool,
    PostToLinkedInTool,
)

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ── Validate required keys ────────────────────────────────────────────────────
def check_env():
    missing = []
    required = {
        "GOOGLE_API_KEY": "Google Gemini LLM + Imagen (get at https://aistudio.google.com/)",
        "NEWSDATA_API_KEY": "NewsData.io news feed (get at https://newsdata.io/)",
    }
    for key, desc in required.items():
        if not os.environ.get(key):
            missing.append(f"  {key:30s}  {desc}")
    if missing:
        print("ERROR: Missing required environment variables:\n")
        print("\n".join(missing))
        print("\nCopy .env.example to .env and fill in your keys.")
        sys.exit(1)

check_env()

# ── Parse CLI args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ContentAI News-to-Social pipeline")
parser.add_argument("--topic", default="technology artificial intelligence", help="News search topic")
parser.add_argument("--language", default="en", help="News language code")
parser.add_argument("--articles", type=int, default=3, help="Number of articles to fetch (1-5)")
parser.add_argument("--dry-run", action="store_true", help="Simulate posts without publishing")
args = parser.parse_args()

if args.dry_run:
    print("\n[DRY RUN MODE] Posts will be simulated, not published.\n")

# ── LLM: Google Gemini ────────────────────────────────────────────────────────
llm = LLM(
    model="gemini/gemini-3-flash-preview",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
news_tool       = NewsDataTool()
image_gen_tool  = GenerateImageTool()
post_x_tool     = PostToXTool()
post_fb_tool    = PostToFacebookTool()
post_ig_tool    = PostToInstagramTool()
post_li_tool    = PostToLinkedInTool()

# ════════════════════════════════════════════════════════════════════════════
#  AGENTS
# ════════════════════════════════════════════════════════════════════════════

content_writer_agent = Agent(
    role="Social Media Content Writer",
    goal=(
        "Fetch the latest news on the given topic and rewrite each article "
        "into 4 distinct, platform-optimized posts for X (Twitter), Facebook, "
        "Instagram, and LinkedIn. Each post must feel native to its platform."
    ),
    backstory=(
        "You are a seasoned social media strategist with 10 years of experience "
        "managing brand accounts across all major platforms. You know that X needs "
        "punchy, concise hooks (≤280 chars), Facebook thrives on conversational "
        "storytelling, Instagram needs visual-first captions with emojis and hashtags, "
        "and LinkedIn demands professional insights with thought-leadership tone. "
        "You adapt every piece of content accordingly."
    ),
    llm=llm,
    tools=[news_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=8,
)

image_creator_agent = Agent(
    role="AI Visual Content Creator",
    goal=(
        "For each news article, generate a vivid cartoon-style illustration using "
        "Google Imagen 3. The image must feature a cheerful small character wearing "
        "sunglasses, pointing at the article topic. Return a mapping of article title "
        "to local image file path."
    ),
    backstory=(
        "You are a creative AI illustrator specializing in infographic-style cartoon art. "
        "You distill complex news topics into visually engaging illustrations that stop "
        "the scroll on social media. Your style is always bright, friendly, and on-brand: "
        "a cartoon mascot with sunglasses guiding the viewer through the key information."
    ),
    llm=llm,
    tools=[image_gen_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=10,
)

publisher_agent = Agent(
    role="Social Media Publisher",
    goal=(
        "Take the platform-optimized content and generated images, then publish "
        "each post to its target platform: X, Facebook (with image), Instagram, and LinkedIn."
    ),
    backstory=(
        "You are a meticulous digital publisher responsible for scheduling and "
        "distributing content across social channels. You ensure every post reaches "
        "its intended platform exactly as written, attach images to Facebook posts, "
        "log the outcome of each post, and report a clear summary of results."
    ),
    llm=llm,
    tools=[post_x_tool, post_fb_tool, post_ig_tool, post_li_tool],
    verbose=True,
    allow_delegation=False,
    max_iter=10,
)

# ════════════════════════════════════════════════════════════════════════════
#  TASKS
# ════════════════════════════════════════════════════════════════════════════

task_write_content = Task(
    description=(
        f"Fetch the latest {args.articles} news article(s) on the topic: '{args.topic}' "
        f"(language: {args.language}) using the NewsData Fetcher tool.\n\n"
        "For EACH article, produce exactly 4 platform-specific posts:\n\n"
        "1. X (Twitter) — ≤280 characters total (including the URL). Hook-first. "
        "   Use 1-2 relevant hashtags. End with the article source URL.\n\n"
        "2. Facebook — 2-4 sentences. Conversational tone. Invite engagement with "
        "   a question or CTA. May include 1-2 hashtags. End with: 'Read more: [URL]'.\n\n"
        "3. Instagram — Visual-first caption. Start with a strong opening line. "
        "   Use emojis naturally. 5-10 relevant hashtags at the end. "
        "   Add 'Source: [URL]' before the hashtags.\n\n"
        "4. LinkedIn — Professional tone. 3-5 sentences. Include a key insight or "
        "   business implication. End with a thought-leadership question followed by the URL.\n\n"
        "IMPORTANT: Every post must include the original article URL as a source link.\n\n"
        "Format your output as structured sections clearly labeled per article and per platform. "
        "For each article, include a line: 'Article Title: [exact title]' and "
        "'Article Summary: [2-sentence summary]' at the top of its section."
    ),
    expected_output=(
        "A structured document with clearly labeled sections for each article. "
        "Each section starts with 'Article Title:' and 'Article Summary:', "
        "followed by 4 platform-specific posts (X, Facebook, Instagram, LinkedIn)."
    ),
    agent=content_writer_agent,
    output_file="content_draft.md",
)

task_generate_images = Task(
    description=(
        "Read the content draft produced by the Content Writer. "
        "For EACH article in the draft, extract the 'Article Title' and 'Article Summary', "
        "then call the 'Generate Article Image' tool to create a cartoon illustration.\n\n"
        "The tool will save the image locally and return a file path.\n\n"
        "After generating all images, output a mapping document with:\n"
        "  Article Title: [title]\n"
        "  Image Path: [local file path returned by the tool]\n\n"
        "Repeat for every article. If image generation fails for one article, "
        "log the error and continue with the next."
    ),
    expected_output=(
        "A mapping document listing each article title and its corresponding "
        "generated image file path (e.g., 'images/article_name.png')."
    ),
    agent=image_creator_agent,
    context=[task_write_content],
    output_file="image_mapping.md",
)

task_publish = Task(
    description=(
        "Read the platform-specific content from the Content Writer and the image "
        "mapping from the Image Creator.\n\n"
        "For each article's content, publish to ALL 4 platforms:\n"
        "  - Use 'Post to X (Twitter)' for the X post\n"
        "  - Use 'Post to Facebook' for the Facebook post — ALWAYS include the "
        "    image_path from the image mapping for this article\n"
        "  - Use 'Post to Instagram' for the Instagram post (provide image_url if available)\n"
        "  - Use 'Post to LinkedIn' for the LinkedIn post\n\n"
        "After all posts are attempted, produce a publishing report summarizing:\n"
        "  - Total posts attempted\n"
        "  - Successful posts (with post IDs where available)\n"
        "  - Any failures with error details\n\n"
        "Note: If credentials are missing, the tool will simulate the post and log it."
    ),
    expected_output=(
        "A publishing report listing every post attempted, its platform, "
        "success/failure status, image used (for Facebook), and post IDs."
    ),
    agent=publisher_agent,
    context=[task_write_content, task_generate_images],
    output_file="publishing_report.md",
)

# ════════════════════════════════════════════════════════════════════════════
#  CREW
# ════════════════════════════════════════════════════════════════════════════

crew = Crew(
    agents=[content_writer_agent, image_creator_agent, publisher_agent],
    tasks=[task_write_content, task_generate_images, task_publish],
    process=Process.sequential,
    verbose=True,
    memory=False,
    max_rpm=15,
)

# ════════════════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("   ContentAI — News-to-Social Media Pipeline")
    print("═" * 60)
    print(f"   Topic    : {args.topic}")
    print(f"   Articles : {args.articles}")
    print(f"   Language : {args.language}")
    print(f"   Dry run  : {args.dry_run}")
    print("═" * 60 + "\n")

    result = crew.kickoff()

    print("\n" + "═" * 60)
    print("   Pipeline Complete")
    print("═" * 60)
    print("\nContent draft  → content_draft.md")
    print("Image mapping  → image_mapping.md")
    print("Images         → images/")
    print("Publishing log → publishing_report.md\n")
    print(result)
