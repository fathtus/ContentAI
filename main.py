"""
ContentAI — News-to-Social Media Pipeline
==========================================
Pipeline:
  Page 1:
    1. Content Writer   → fetch news (NewsData) + rewrite per platform
                          Rewriter: Gemini (direct) or Qwen via HuggingFace (RewritePostTool)
                          Platforms: X, Facebook, Instagram, LinkedIn
    2. Image Creator    → artistic scene with person + laptop (HuggingFace FLUX.1-schnell)
    3. Publisher        → post to X, Facebook Page 1 (with image), Instagram (image uploaded
                          to tmpfiles.org for public URL), LinkedIn

  Page 2 (optional — activated when --topic2 is provided):
    4. Content Writer 2 → fetch & rewrite news for Facebook Page 2
    5. Image Creator 2  → professional person matching the page topic (HuggingFace FLUX.1-schnell)
    6. Publisher 2      → post to Facebook Page 2 (with image)

Usage:
  python main.py [--topic "AI news"] [--topic2 "healthcare"] [--dry-run] [--rewriter gemini|qwen]
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import (
    NewsDataTool,
    GenerateImageTool,
    GenerateProfessionalImageTool,
    RewritePostTool,
    PostToXTool,
    PostToFacebookTool,
    PostToFacebookPage2Tool,
    PostToInstagramTool,
    PostToLinkedInTool,
)

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

# ── Validate required keys ────────────────────────────────────────────────────
def check_env():
    missing = []
    required = {
        "GOOGLE_API_KEY":   "Google Gemini LLM (get at https://aistudio.google.com/)",
        "NEWSDATA_API_KEY": "NewsData.io news feed (get at https://newsdata.io/)",
    }
    for key, desc in required.items():
        if not os.environ.get(key):
            missing.append(f"  {key:30s}  {desc}")
    if missing:
        print("ERROR: Missing required environment variables:\n")
        print("\n".join(missing))
        sys.exit(1)

check_env()

# ── Parse CLI args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="ContentAI News-to-Social pipeline")
parser.add_argument("--topic",    default="technology artificial intelligence", help="Page 1 news topic")
parser.add_argument("--language", default="en",  help="Page 1 language code")
parser.add_argument("--articles", type=int, default=3, help="Page 1 articles to fetch (1-5)")
parser.add_argument("--topic2",    default="", help="Page 2 news topic (leave empty to disable)")
parser.add_argument("--language2", default="en", help="Page 2 language code")
parser.add_argument("--articles2", type=int, default=3, help="Page 2 articles to fetch (1-5)")
parser.add_argument("--dry-run", action="store_true", help="Simulate posts without publishing")
parser.add_argument("--skip-page1", action="store_true", help="Skip Page 1 pipeline (run Page 2 only)")
parser.add_argument("--platforms", default="facebook,x,instagram,linkedin", help="Comma-separated platforms for Page 1")
parser.add_argument("--rewriter", default="gemini", choices=["gemini", "qwen"], help="Content rewriter: gemini or qwen")
args = parser.parse_args()

has_page2 = bool(args.topic2.strip())

# ── Page 1 platform selection ──────────────────────────────────────────────────
_ALL_PLATFORMS = ["x", "facebook", "instagram", "linkedin"]
_PLATFORM_LABELS = {"x": "X (Twitter)", "facebook": "Facebook", "instagram": "Instagram", "linkedin": "LinkedIn"}
platforms = set(p.strip() for p in args.platforms.split(",") if p.strip())
platform_list_str = ", ".join(_PLATFORM_LABELS[p] for p in _ALL_PLATFORMS if p in platforms)

if args.dry_run:
    print("\n[DRY RUN MODE] Posts will be simulated, not published.\n")

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7,
)

# ── Tools ─────────────────────────────────────────────────────────────────────
news_tool        = NewsDataTool()
rewrite_tool     = RewritePostTool() if args.rewriter == "qwen" else None
image_gen_tool   = GenerateImageTool()
image_pro_tool   = GenerateProfessionalImageTool()
post_x_tool      = PostToXTool()
post_fb_tool     = PostToFacebookTool()
post_fb2_tool    = PostToFacebookPage2Tool()
post_ig_tool     = PostToInstagramTool()
post_li_tool     = PostToLinkedInTool()

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 AGENTS
# ════════════════════════════════════════════════════════════════════════════

content_writer_agent = Agent(
    role="Social Media Content Writer",
    goal=(
        f"Fetch the latest news on the given topic and rewrite each article "
        f"into platform-optimized posts for: {platform_list_str}."
    ),
    backstory=(
        "You are a seasoned social media strategist with 10 years of experience. "
        "X needs punchy hooks (≤280 chars), Facebook is conversational, "
        "Instagram is visual-first with hashtags, LinkedIn is professional thought-leadership."
    ),
    llm=llm, tools=[t for t in [news_tool, rewrite_tool] if t], verbose=True, allow_delegation=False, max_iter=8,
)

image_creator_agent = Agent(
    role="AI Visual Content Creator",
    goal="Generate an artistic image for each news article using the Generate Article Image tool.",
    backstory=(
        "You generate atmospheric artistic images — a person with a laptop in cinematic nature scenes "
        "with holographic chat UI — to accompany social media posts."
    ),
    llm=llm, tools=[image_gen_tool], verbose=True, allow_delegation=False, max_iter=10,
)

_publisher_tool_map = {"x": post_x_tool, "facebook": post_fb_tool, "instagram": post_ig_tool, "linkedin": post_li_tool}
publisher_tools = [_publisher_tool_map[p] for p in _ALL_PLATFORMS if p in platforms]

publisher_agent = Agent(
    role="Social Media Publisher",
    goal=f"Publish content + images to: {platform_list_str}.",
    backstory=(
        "You distribute content to all platforms, attach AI-generated images to Facebook posts, "
        "and report a clear summary of every outcome."
    ),
    llm=llm, tools=publisher_tools,
    verbose=True, allow_delegation=False, max_iter=10,
)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 1 TASKS
# ════════════════════════════════════════════════════════════════════════════

_write_instructions = []
_num = 1
if "x" in platforms:
    _write_instructions.append(f"{_num}. X (Twitter) — ≤280 chars, hook-first, 1-2 hashtags, source URL."); _num += 1
if "facebook" in platforms:
    _write_instructions.append(f"{_num}. Facebook — 2-4 sentences, conversational, CTA, end with 'Read more: [URL]'."); _num += 1
if "instagram" in platforms:
    _write_instructions.append(f"{_num}. Instagram — visual-first, emojis, 5-10 hashtags, 'Source: [URL]'."); _num += 1
if "linkedin" in platforms:
    _write_instructions.append(f"{_num}. LinkedIn — professional, 3-5 sentences, thought-leadership question + URL."); _num += 1

_rewriter_note = (
    "For each platform post, use the 'Rewrite Post for Platform' tool "
    "(pass the article summary as raw_content, the platform name, and the source URL)."
    if args.rewriter == "qwen" else
    "Write each platform post directly based on the article content."
)

task_write_content = Task(
    description=(
        f"Fetch the latest {args.articles} news article(s) on '{args.topic}' "
        f"(language: {args.language}) using the NewsData Fetcher tool.\n\n"
        f"For EACH article produce posts for: {platform_list_str}:\n"
        + "\n".join(_write_instructions) + "\n\n"
        f"{_rewriter_note}\n\n"
        "Include 'Article Title: [title]' and 'Article Summary: [2 sentences]' at the top of each section."
    ),
    expected_output=f"Structured document with Article Title, Article Summary, and platform posts ({platform_list_str}) per article.",
    agent=content_writer_agent,
    output_file="content_draft.md",
)

task_generate_images = Task(
    description=(
        "Read the content draft. For EACH article, extract 'Article Title' and 'Article Summary', "
        "then call 'Generate Article Image' to create an image. "
        "Output a mapping: 'Article Title: [title]\\nImage Path: [path]' for each article."
    ),
    expected_output="A mapping of article titles to generated image file paths.",
    agent=image_creator_agent,
    context=[task_write_content],
    output_file="image_mapping.md",
)

_publish_steps = []
if "x" in platforms:
    _publish_steps.append("  - 'Post to X (Twitter)' for the X post")
if "facebook" in platforms:
    _publish_steps.append("  - 'Post to Facebook' with the image_path from the mapping")
if "instagram" in platforms:
    _publish_steps.append("  - 'Post to Instagram' with the image_path from the mapping")
if "linkedin" in platforms:
    _publish_steps.append("  - 'Post to LinkedIn' for the LinkedIn post")

task_publish = Task(
    description=(
        f"Read the content draft and image mapping.\n"
        f"For each article publish to: {platform_list_str}:\n"
        + "\n".join(_publish_steps) + "\n\n"
        "Produce a publishing report with total posts, successes (with IDs), and failures."
    ),
    expected_output=f"Publishing report with status and post IDs for: {platform_list_str}.",
    agent=publisher_agent,
    context=[task_write_content, task_generate_images],
    output_file="publishing_report.md",
)

# ════════════════════════════════════════════════════════════════════════════
#  PAGE 2 AGENTS & TASKS (only built when topic2 is provided)
# ════════════════════════════════════════════════════════════════════════════

page2_agents = []
page2_tasks  = []

if has_page2:
    content_writer2_agent = Agent(
        role="Facebook Page 2 Content Writer",
        goal=(
            f"Fetch the latest news on '{args.topic2}' and rewrite each article "
            "into an engaging Facebook post."
        ),
        backstory=(
            "You specialise in creating compelling Facebook content for niche audiences. "
            "Your posts are conversational, informative, and always include the source URL."
        ),
        llm=llm, tools=[news_tool], verbose=True, allow_delegation=False, max_iter=8,
    )

    image_creator2_agent = Agent(
        role="Professional Visual Creator for Page 2",
        goal=(
            f"Generate a professional photo for each article using the "
            f"'Generate Professional Article Image' tool. "
            f"Always pass page_topic='{args.topic2}' so the right profession is used."
        ),
        backstory=(
            f"You create artistic images featuring professionals whose career matches the '{args.topic2}' topic. "
            "Each image places the person in a different professional scene."
        ),
        llm=llm, tools=[image_pro_tool], verbose=True, allow_delegation=False, max_iter=10,
    )

    publisher2_agent = Agent(
        role="Facebook Page 2 Publisher",
        goal="Publish each article's content and professional image to Facebook Page 2.",
        backstory=(
            "You post content exclusively to Facebook Page 2, always attaching the professional image."
        ),
        llm=llm, tools=[post_fb2_tool], verbose=True, allow_delegation=False, max_iter=10,
    )

    task_write2 = Task(
        description=(
            f"Fetch the latest {args.articles2} news article(s) on '{args.topic2}' "
            f"(language: {args.language2}) using the NewsData Fetcher tool.\n\n"
            "For EACH article produce a Facebook post:\n"
            "- 2-4 sentences, engaging and informative tone.\n"
            "- Relevant emoji at the start.\n"
            "- End with 'Read more: [URL]'.\n\n"
            "Include 'Article Title: [title]' and 'Article Summary: [2 sentences]' at the top of each section."
        ),
        expected_output="Structured document with Article Title, Article Summary, and Facebook post per article.",
        agent=content_writer2_agent,
        output_file="content_draft_p2.md",
    )

    task_generate_images2 = Task(
        description=(
            f"Read the Page 2 content draft. For EACH article extract 'Article Title' and 'Article Summary', "
            f"then call 'Generate Professional Article Image' with page_topic='{args.topic2}'. "
            "Output: 'Article Title: [title]\\nImage Path: [path]' for each article."
        ),
        expected_output="Mapping of article titles to professional image file paths.",
        agent=image_creator2_agent,
        context=[task_write2],
        output_file="image_mapping_p2.md",
    )

    task_publish2 = Task(
        description=(
            "Read the Page 2 content draft and image mapping.\n"
            "For each article use 'Post to Facebook Page 2' with the image_path from the mapping.\n"
            "Produce a publishing report with outcomes and post IDs."
        ),
        expected_output="Publishing report for Facebook Page 2.",
        agent=publisher2_agent,
        context=[task_write2, task_generate_images2],
        output_file="publishing_report_p2.md",
    )

    page2_agents = [content_writer2_agent, image_creator2_agent, publisher2_agent]
    page2_tasks  = [task_write2, task_generate_images2, task_publish2]

# ════════════════════════════════════════════════════════════════════════════
#  CREWS  (two independent pipelines — Page 1 and Page 2)
# ════════════════════════════════════════════════════════════════════════════

crew1 = Crew(
    agents=[content_writer_agent, image_creator_agent, publisher_agent],
    tasks=[task_write_content, task_generate_images, task_publish],
    process=Process.sequential,
    verbose=True,
    memory=False,
    max_rpm=15,
)

if has_page2:
    crew2 = Crew(
        agents=[content_writer2_agent, image_creator2_agent, publisher2_agent],
        tasks=[task_write2, task_generate_images2, task_publish2],
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
    print(f"   Page 1 Topic   : {args.topic}")
    print(f"   Page 1 Articles: {args.articles}  |  Language: {args.language}")
    if has_page2:
        print(f"   Page 2 Topic   : {args.topic2}")
        print(f"   Page 2 Articles: {args.articles2}  |  Language: {args.language2}")
    print(f"   Dry run        : {args.dry_run}")
    print("═" * 60 + "\n")

    if not args.skip_page1:
        print("═" * 60)
        print("   Running Page 1 Pipeline")
        print("═" * 60 + "\n")
        result1 = crew1.kickoff()
        print("\n" + "═" * 60)
        print("   Page 1 Pipeline Complete")
        print("═" * 60)
        print("\nPage 1 → content_draft.md | image_mapping.md | publishing_report.md\n")
        print(result1)

    if has_page2:
        print("\n" + "═" * 60)
        print("   Running Page 2 Pipeline")
        print("═" * 60 + "\n")
        result2 = crew2.kickoff()
        print("\n" + "═" * 60)
        print("   Page 2 Pipeline Complete")
        print("═" * 60)
        print("\nPage 2 → content_draft_p2.md | image_mapping_p2.md | publishing_report_p2.md\n")
        print(result2)
