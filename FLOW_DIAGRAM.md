# ContentAI — Pipeline Flow Diagram

## Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│                                                                     │
│   Web UI (http://localhost:5050)          CLI                       │
│   ┌─────────────────────────┐      python main.py                  │
│   │ Topic / Language /      │        --topic "AI"                  │
│   │ Articles / Dry-run      │        --articles 3                  │
│   │ [Run Pipeline]          │        --dry-run                     │
│   └────────────┬────────────┘                                       │
└────────────────┼────────────────────────────────────────────────────┘
                 │  HTTP POST /run
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                       server.py (Flask)                            │
│                                                                    │
│   Spawns subprocess → main.py                                      │
│   Streams stdout via SSE → browser (Live Log tab)                  │
│   Serves /results → content_draft.md + publishing_report.md       │
└────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────────────────┐
│                    main.py  — CrewAI Pipeline                      │
│                                                                    │
│   LLM: Google Gemini 3 Flash Preview (via LiteLLM)                 │
│   Process: Sequential                                              │
└───────────────────────────────┬────────────────────────────────────┘
                                │
          ┌─────────────────────┴────────────────────┐
          │                                          │
          ▼                                          │
┌─────────────────────────────────────┐              │
│     TASK 1: Write Content           │              │
│                                     │              │
│  Agent: Content Writer              │              │
│                                     │              │
│  1. Calls NewsData Fetcher tool     │              │
│     └─► GET newsdata.io/api/1/latest│              │
│         params: query, language,    │              │
│                 size                │              │
│         returns: N articles         │              │
│                  (title, summary,   │              │
│                   URL, source)      │              │
│                                     │              │
│  2. Gemini rewrites each article    │              │
│     into 4 platform posts:          │              │
│                                     │              │
│     ┌──────────┐  ┌──────────────┐  │              │
│     │    X     │  │   Facebook   │  │              │
│     │ ≤280 ch  │  │ Conversatio- │  │              │
│     │ 1-2 tags │  │ nal + CTA    │  │              │
│     └──────────┘  └──────────────┘  │              │
│     ┌──────────┐  ┌──────────────┐  │              │
│     │Instagram │  │  LinkedIn    │  │              │
│     │ Emojis + │  │ Professional │  │              │
│     │ hashtags │  │ thought lead │  │              │
│     └──────────┘  └──────────────┘  │              │
│                                     │              │
│  Output: content_draft.md           │              │
└─────────────────────────────────────┘              │
          │                                          │
          │  context passed to Task 2               │
          ▼                                          │
┌─────────────────────────────────────┐              │
│     TASK 2: Publish                 │              │
│                                     │              │
│  Agent: Publisher                   │              │
│                                     │              │
│  Reads content from Task 1 and      │              │
│  calls one posting tool per article │              │
│  per platform:                      │              │
│                                     │              │
│  ┌──────────────────────────────┐   │              │
│  │ Post to X (Twitter)          │   │              │
│  │ tweepy → POST /2/tweets      │   │              │
│  │ credentials: X_API_KEY etc.  │   │              │
│  └──────────────────────────────┘   │              │
│  ┌──────────────────────────────┐   │              │
│  │ Post to Facebook             │   │              │
│  │ Graph API v19.0 /page/feed   │   │              │
│  │ credentials: PAGE_TOKEN      │   │              │
│  └──────────────────────────────┘   │              │
│  ┌──────────────────────────────┐   │              │
│  │ Post to Instagram            │   │              │
│  │ Graph API v19.0              │   │              │
│  │ /ig_user/media → /publish    │   │              │
│  │ credentials: IG_ACCOUNT_ID   │   │              │
│  └──────────────────────────────┘   │              │
│  ┌──────────────────────────────┐   │              │
│  │ Post to LinkedIn             │   │              │
│  │ ugcPosts API                 │   │              │
│  │ credentials: ACCESS_TOKEN    │   │              │
│  └──────────────────────────────┘   │              │
│                                     │              │
│  ► If credentials missing:          │              │
│    Tool returns simulated output    │◄─────────────┘
│    (dry-run safe)                   │
│                                     │
│  Output: publishing_report.md       │
└─────────────────────────────────────┘


## Data Flow Summary

  User Input
      │
      ▼
  NewsData.io API  ──►  Raw articles (title, description, URL)
      │
      ▼
  Gemini LLM  ──►  4 × platform-specific posts per article
      │
      ├──►  X (Twitter) ──►  Tweet ID / simulated
      ├──►  Facebook    ──►  Post ID  / simulated
      ├──►  Instagram   ──►  Media ID / simulated
      └──►  LinkedIn    ──►  Post ID  / simulated
      │
      ▼
  content_draft.md  +  publishing_report.md


## Simulation Mode

When social media credentials are not set, every posting tool falls back
to simulation mode automatically — no credentials required to test the
full pipeline end-to-end.

  Credentials set  →  Real API call  →  Post ID returned
  Credentials missing  →  Simulated log  →  No external call made


## File Outputs

  content_draft.md
  ├── Article 1
  │   ├── X post
  │   ├── Facebook post
  │   ├── Instagram post
  │   └── LinkedIn post
  ├── Article 2
  │   └── ...
  └── Article N

  publishing_report.md
  ├── Total attempted
  ├── Successful posts (platform + ID)
  └── Failures (platform + error)
```
