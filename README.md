# ContentAI — News-to-Social Media Pipeline

Fetches news from **NewsData.io**, rewrites it for each platform using **Google Gemini**, and publishes to **X, Facebook, Instagram, and LinkedIn** via CrewAI agents.

## Architecture

```
NewsData.io → Content Writer Agent (Gemini) → Publisher Agent
                    ↓                               ↓
             content_draft.md           publishing_report.md
                                    X | Facebook | Instagram | LinkedIn
```

See [FLOW_DIAGRAM.md](FLOW_DIAGRAM.md) for the full pipeline diagram.

## Project Structure

```
ContentAI/
├── main.py                    # Pipeline entry point (CLI)
├── server.py                  # Flask web interface
├── requirements.txt
├── .env.example               # API keys reference
├── FLOW_DIAGRAM.md            # Visual pipeline diagram
├── tools/
│   ├── __init__.py
│   ├── newsdata_tool.py       # NewsData.io fetcher
│   └── social_media_tools.py # X, Facebook, Instagram, LinkedIn posting
└── templates/
    └── index.html             # Web UI (dark theme, SSE live log)
```

## Agents

| Agent | LLM | Tools |
|---|---|---|
| Content Writer | Gemini 3 Flash Preview | NewsData Fetcher |
| Publisher | Gemini 3 Flash Preview | Post to X, Facebook, Instagram, LinkedIn |

## Setup

```bash
cd ContentAI
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Required Keys

| Key | Where to get it |
|---|---|
| `GOOGLE_API_KEY` | https://aistudio.google.com/app/apikey |
| `NEWSDATA_API_KEY` | https://newsdata.io/ (free tier available) |

## Usage

### Web Interface (recommended)

```bash
python server.py
# Open http://localhost:5050
```

The web UI lets you configure the topic, language, article count, and dry-run mode,
then streams live agent logs and renders the generated content as formatted markdown.

### CLI

```bash
# Default: fetch 3 articles on "technology artificial intelligence"
python main.py

# Custom topic
python main.py --topic "climate change" --articles 5

# Dry run (simulate posts without publishing)
python main.py --topic "business news" --dry-run
```

## Platform Post Styles

| Platform | Format |
|---|---|
| X (Twitter) | ≤280 chars, hook-first, 1–2 hashtags |
| Facebook | Conversational, 2–4 sentences, CTA |
| Instagram | Emoji-rich caption, 5–10 hashtags |
| LinkedIn | Professional insight, thought-leadership tone |

## Social Media Credentials

All social media keys are **optional**. Without them, posts are **simulated** — logged to the console and `publishing_report.md` but not published. See `.env.example` for per-platform setup instructions.

| Platform | Required Keys |
|---|---|
| X (Twitter) | `X_API_KEY`, `X_API_SECRET`, `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET` |
| Facebook | `FACEBOOK_PAGE_ID`, `FACEBOOK_PAGE_TOKEN` |
| Instagram | `INSTAGRAM_BUSINESS_ACCOUNT_ID` + Facebook token above |
| LinkedIn | `LINKEDIN_ACCESS_TOKEN`, `LINKEDIN_AUTHOR_URN` |

## Output Files

| File | Description |
|---|---|
| `content_draft.md` | Platform-specific posts written by Gemini |
| `publishing_report.md` | Post IDs and publish status per platform |
