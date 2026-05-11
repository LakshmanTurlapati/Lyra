# SPDX-License-Identifier: MIT
"""Fresh corpora for TC-D2 (web/scraping/HTTP/social) -- avoid overlap with D wave 1."""

DOMAINS = [
    "substack.com", "ghost.io", "dev.to", "hashnode.com", "buttondown.email",
    "patreon.com", "kickstarter.com", "indiegogo.com", "gofundme.com",
    "producthunt.com", "indiehackers.com", "ycombinator.com", "vercel.app",
    "netlify.app", "fly.io", "render.com", "railway.app", "supabase.com",
    "stripe.com", "paypal.com", "shopify.dev", "atlassian.com", "jira.com",
    "notion.so", "asana.com", "monday.com", "trello.com", "airtable.com",
    "miro.com", "mural.co", "figma.com", "framer.com", "webflow.com",
    "spotify.com", "soundcloud.com", "bandcamp.com", "tidal.com", "deezer.com",
    "vimeo.com", "twitch.tv", "kick.com", "rumble.com", "dailymotion.com",
    "pinterest.com", "tumblr.com", "tiktok.com", "snapchat.com", "discord.com",
    "telegram.org", "whatsapp.com", "signal.org", "matrix.org", "slack.com",
    "zoom.us", "webex.com", "teams.microsoft.com", "linkedin.com", "indeed.com",
    "glassdoor.com", "monster.com", "ziprecruiter.com", "weworkremotely.com",
]

PATHS = [
    "/blog/post/2026-04-15", "/category/tutorials", "/u/janedoe/profile",
    "/jobs/12345", "/listings/featured", "/dashboard/projects",
    "/explore/trending", "/release-notes", "/changelog", "/status",
    "/api/v2/data", "/api/v3/users/me", "/oauth/authorize", "/login",
    "/signup", "/forgot-password", "/team/about", "/case-studies/acme",
    "/whitepapers/2026", "/events/upcoming", "/podcast/ep-42",
    "/newsletter/archive", "/showcase", "/integrations", "/marketplace",
]

SEARCH_QUERIES = [
    "best mechanical keyboards under 150",
    "how to deploy nextjs on cloudflare",
    "rust web framework comparison axum actix",
    "GraphQL vs REST API design 2026",
    "Postgres pgvector tutorial",
    "Llama 4 Maverick vs Scout",
    "Apple WWDC 2026 keynote summary",
    "Sora 2 video generation samples",
    "EU Digital Services Act fines latest",
    "ChatGPT enterprise pricing 2026",
    "Mistral Large 3 benchmarks",
    "Tailwind CSS v5 migration guide",
    "Vue 4 composition API examples",
    "Svelte 6 runes deep dive",
    "Bun vs Node runtime benchmarks",
    "Deno KV global database review",
    "WebGPU compute shader tutorial",
    "WASI preview 2 announcement",
    "ARM PC laptops Snapdragon X review",
    "Ryzen 9000 series gaming benchmarks",
    "iPad Pro M5 review battery life",
    "best ergonomic office chairs 2026",
    "indoor air quality monitor recommendations",
    "Tesla Cybertruck production numbers Q1",
    "BYD seal export markets",
    "Northvolt bankruptcy news",
    "TSMC Arizona fab progress",
    "NVIDIA Blackwell shipping delays",
    "Intel 18A node yield rumors",
    "Anthropic raises series E reports",
    "Perplexity AI valuation 2026",
    "xAI Grok 4 performance",
    "Cohere Command R+ release",
    "Stability AI restructuring news",
    "Adobe Firefly 4 features",
    "Sundar Pichai congressional testimony",
    "FTC merger guidelines 2026 update",
    "California AI safety bill SB 53",
    "Texas grid winterization report",
    "California EV mandate 2035 update",
    "EPA methane rule final",
    "ICE breaker fleet expansion plan",
    "Greenland geothermal pilot project",
    "Iceland volcanic eruption Reykjanes",
    "Svalbard seed vault expansion",
]

SUBREDDITS = [
    "selfhosted", "homelab", "DataHoarder", "linuxmemes", "linux4noobs",
    "buildapc", "hardware", "PCMasterRace", "pcgaming", "Steam",
    "EmulationOnAndroid", "androiddev", "iOSProgramming", "swift",
    "kotlin", "java", "csharp", "dotnet", "haskell", "scala",
    "lisp", "clojure", "emacs", "vim", "neovim", "tmux",
    "bashrc", "zsh", "commandline", "linux", "archlinux",
    "debian", "fedora", "Ubuntu", "NixOS", "GUIX",
]

TWITTER_QUERIES = [
    "#OpenSource since:2026-04-01",
    "@karpathy",
    "from:swyx -filter:replies",
    "lang:en \"prompt injection\" min_retweets:50",
    "geocode:40.7128,-74.0060,10km lunch",
    "filter:videos \"WWDC\"",
    "min_faves:5000 #LLM",
    "to:vercel \"deploy failed\"",
    "list:elonmusk/lists/tech-leaders",
    "url:huggingface.co",
]

ARXIV_QUERIES = [
    "state space models mamba",
    "long context retrieval RAG",
    "multimodal embedding contrastive",
    "robotics learning from demonstrations",
    "protein structure folding diffusion",
    "weather forecasting graph neural",
    "LLM agent benchmarks 2026",
    "constitutional AI revisited",
    "reasoning chain of thought distillation",
    "code generation eval HumanEval+",
]

WIKI_TOPICS = [
    "Photoelectric_effect", "Tropical_cyclone", "Renaissance",
    "Industrial_Revolution", "Cold_War", "Big_Bang",
    "General_relativity", "Mitochondrion", "Continental_drift",
    "French_Revolution", "Silk_Road", "Mariana_Trench",
    "Sahara", "Amazon_rainforest", "Antarctica",
    "Ada_Lovelace", "Marie_Curie", "Alan_Turing",
]

GITHUB_REPOS = [
    "ollama/ollama", "vllm-project/vllm", "ggerganov/whisper.cpp",
    "langchain-ai/langchain", "run-llama/llama_index",
    "microsoft/autogen", "stanfordnlp/dspy",
    "BerriAI/litellm", "Mintplex-Labs/anything-llm",
    "open-webui/open-webui", "lobehub/lobe-chat",
    "RVC-Boss/GPT-SoVITS", "comfyanonymous/ComfyUI",
    "Stability-AI/generative-models", "AUTOMATIC1111/stable-diffusion-webui",
    "tinygrad/tinygrad", "karpathy/llm.c",
    "denoland/deno", "oven-sh/bun",
    "withastro/astro", "remix-run/remix",
    "solidjs/solid", "qwikifiers/qwik",
]

YT_SEARCHES = [
    "fastapi production deployment",
    "tailwind v5 new features",
    "ollama local llm setup",
    "stable diffusion comfyui workflow",
    "raspberry pi 5 home server",
    "obsidian plugins workflow",
    "neovim from scratch 2026",
    "docker compose tutorial",
    "kubernetes ingress nginx",
    "terraform aws complete guide",
]

WIKI_QUERIES = [
    "general relativity black holes",
    "Ottoman Empire decline",
    "photovoltaic solar cell efficiency",
    "Hubble Space Telescope discoveries",
    "Pacific Ring of Fire volcanoes",
    "neolithic agricultural revolution",
    "amino acids protein synthesis",
    "tectonic plate boundaries",
]

GH_REPO_QUERIES = [
    "language:python topic:rag stars:>500",
    "language:typescript framework:nextjs",
    "topic:vector-database",
    "topic:agent-framework language:python",
    "language:zig stars:>100",
    "language:elixir phoenix",
    "topic:webassembly language:rust",
    "topic:embedded language:rust",
]

NEWS_QUERIES = [
    "AI regulation Europe enforcement",
    "semiconductor supply chain Taiwan",
    "central bank digital currency pilot",
    "carbon capture funding US",
    "ocean plastic cleanup progress",
    "lab grown meat approval Europe",
    "remote work return to office mandates",
    "creator economy platform fees",
    "open source security CVE 2026",
    "data center water usage drought",
]

IMAGE_QUERIES = [
    "japanese tea garden kyoto",
    "art deco interior",
    "vintage muscle car photo",
    "icelandic glacier aerial",
    "northern lights timelapse still",
    "minimalist desk setup wood",
    "industrial loft kitchen",
    "tropical beach drone view",
    "mountain bike singletrack",
    "vinyl record collection shelf",
]

REDDIT_TEXT_QUERIES = [
    "best self-hosted password manager",
    "how to migrate from VMware to Proxmox",
    "fiber install vs cable comparison",
    "any good open weight TTS model",
    "is Linux desktop ready 2026",
    "what GPU for local LLM 16GB",
    "thinkpad t14 vs framework 13",
    "pop os vs fedora workstation",
]

SHORT_URLS = [
    "https://buff.ly/abcd",
    "https://ift.tt/9zXyW",
    "https://lnkd.in/eVpQrSt",
    "https://goo.gl/maps/QwErTy",
    "https://youtu.be/xKzLm9pQ",
    "https://amzn.to/3jKpL",
    "https://rb.gy/9xYz",
    "https://shorturl.at/aBcDe",
]

LANGS = ["it", "pt", "nl", "sv", "pl", "tr", "ko", "zh", "ar", "hi"]

COUNTRIES = ["CA", "AU", "FR", "JP", "BR", "IN", "MX", "ZA"]

CSS_SELECTORS = [
    "article h3.title",
    "section.featured > div.card",
    "ul.nav li:nth-child(2) a",
    "[data-testid='price']",
    "main .post-body p",
    "footer .social-link",
    "table.results tr td:first-child",
    ".sidebar .widget h4",
]

XPATHS = [
    "//meta[@property='og:image']/@content",
    "//article//time/@datetime",
    "//ul[@class='breadcrumb']//a/text()",
    "//table[@id='results']//tr[position()>1]",
    "//script[@type='application/ld+json']/text()",
    "//div[contains(@class,'price')]//text()",
]
