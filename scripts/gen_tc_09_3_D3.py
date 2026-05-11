#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_D.py -- Generate 500 ShareGPT tool-calling samples for TC-D.

Domain: web search, scraping, HTTP, RSS, social-media APIs, URL utilities.
Seed: 1009307D
Output: datasets/tool-calling/raw-09.3/batch-07-D.jsonl (exactly 500 lines)
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-09-D.jsonl"

SEED = "1009309D"
SYSTEM_PROMPT = "You are a helpful assistant. Prefer calling tools over guessing."

SUFFIX_POOL = [
    "That's all pulled up and ready to go.",
    "Done — the data came back clean, everything lines up with what you originally asked for, and I don't see any mismatches worth flagging in the payload before you move on to the next step.",
    "Pulled the record successfully; the value you needed is included in the response above.",
    "All set on my end.",
    "Got it handled — want me to take this further, or is that enough?",
    "Operation wrapped up without issues, so you should be good to proceed from here.",
    "Finished pulling that together; let me know if you'd like a deeper breakdown of any particular field, or if you'd prefer I reformat the response into something more readable for downstream consumption.",
    "The call routed through cleanly and returned exactly what you were after.",
    "Wrapped up — anything else you'd like me to chase down while we're here?",
    "That request returned successfully, and the full payload sits just above for your review whenever you're ready to look through it at your own pace.",
    "Verified and delivered.",
    "Fetched and parsed — the numbers check out against what the API reported.",
    "Sorted out; the output is self-explanatory but I'm happy to walk through it.",
    "Query executed, results returned, and nothing looked off in the response body.",
    "All yours — holler if you need a follow-up lookup, want to drill into the details, or spot anything in the output that deserves a second pass from my end.",
    "Task closed out. If this raises new questions, just say the word and I'll pivot.",
    "Response is ready above, covering each of the fields you originally requested.",
    "Returned cleanly without errors.",
    "Everything ran end-to-end, the result matches the shape of what you were expecting, and I'd lean toward calling this one finished unless you want me to cross-check anything against another source for sanity.",
    "Just finished the lookup — does this cover what you needed, or should I keep digging?",
    "Output is queued up above; it should give you what you need to move forward.",
    "Sent off, processed, confirmed — the operation completed in a single round-trip.",
    "Here you go.",
    "Call succeeded on the first attempt, no retries or fallback logic had to kick in this time, and the timings look perfectly normal compared to prior runs of the same endpoint.",
    "That's wrapped — feel free to ask if any of the returned fields need clarification.",
    "Information retrieved; I've kept the raw response intact so you can inspect it directly.",
    "Done and dusted.",
    "Happy to refine further if the result isn't quite what you were picturing — otherwise, we're good to call this one shipped and roll on to whatever's next on your list.",
    "Fetched successfully, and the shape of the payload aligns with the documented schema.",
    "That should do it on this one.",
]
assert len(SUFFIX_POOL) == 30


# (tool_name, user_prompt_template, args_builder, result_builder)
# We define many specs; each spec can produce multiple variants by varying parameters.

# Helpers
DOMAINS = [
    "example.com", "news.ycombinator.com", "github.com", "wikipedia.org",
    "reddit.com", "arxiv.org", "stackoverflow.com", "medium.com",
    "techcrunch.com", "bbc.co.uk", "nytimes.com", "bloomberg.com",
    "reuters.com", "theguardian.com", "wired.com", "engadget.com",
    "verge.com", "ars-technica.com", "nature.com", "science.org",
    "youtube.com", "twitter.com", "x.com", "linkedin.com",
    "amazon.com", "ebay.com", "etsy.com", "shopify.com",
    "apple.com", "microsoft.com", "google.com", "meta.com",
    "openai.com", "anthropic.com", "huggingface.co", "kaggle.com",
    "duckduckgo.com", "mozilla.org", "wordpress.com", "blogspot.com",
]

PATHS = [
    "/about", "/blog", "/articles/2026/05", "/news", "/posts/123",
    "/help/faq", "/products", "/contact", "/privacy", "/terms",
    "/docs/api", "/wiki/Main_Page", "/r/python/top", "/users/jdoe",
    "/search?q=ai", "/topic/health", "/category/science", "/2026/04/30/headline",
    "/api/v1/items", "/feed.rss", "/sitemap.xml", "/robots.txt",
]

SEARCH_QUERIES = [
    "SEC ETF approval latest", "best Python web frameworks 2026",
    "climate change Arctic ice 2026", "GPT-5 release rumors",
    "fed interest rate decision May", "World Cup 2026 schedule",
    "rust vs go performance benchmarks", "best noise cancelling headphones 2026",
    "kubernetes 1.32 release notes", "Apple Vision Pro 2 leaks",
    "tesla Q1 earnings", "AI safety research papers 2026",
    "Boston Dynamics atlas humanoid", "OpenAI Sora demo",
    "Bitcoin halving impact", "James Webb telescope new images",
    "SpaceX Starship test flight", "EU AI Act compliance",
    "TypeScript 5.5 features", "PostgreSQL 17 vs MySQL 8",
    "Rust async runtime tokio vs async-std", "Vercel pricing changes",
    "Llama 4 benchmarks", "Anthropic Claude 4.7 review",
    "stock market crash 2026", "Ozempic side effects",
    "Mars Sample Return mission", "quantum computing IBM 2026",
    "Wimbledon 2026 draw", "Olympics 2028 Los Angeles venues",
    "Ford Mustang Mach-E recall", "Boeing 737 MAX update",
    "Chinese economy 2026 outlook", "ECB rate cut June",
    "Nvidia H300 specs", "TSMC 2nm node",
    "Threads vs Bluesky users", "Mastodon adoption 2026",
    "remote work trends 2026", "GDPR fines tracker",
    "Supreme Court ruling ETF", "FDA approval Alzheimer drug",
    "Boeing Starliner astronauts return", "Polaris Dawn spacewalk",
    "WHO pandemic preparedness treaty", "NATO summit 2026 agenda",
]

SUBREDDITS = ["python", "MachineLearning", "rust", "webdev", "programming",
              "technology", "science", "worldnews", "news", "investing",
              "stocks", "personalfinance", "AskScience", "explainlikeimfive",
              "todayilearned", "Futurology", "space", "datascience"]

TWITTER_QUERIES = ["#AI", "@elonmusk", "OpenAI from:sama", "claude opus",
                   "lang:en geocode:37.7749,-122.4194,5km coffee",
                   "filter:images SpaceX", "min_faves:1000 GPT-5",
                   "#WebDev since:2026-05-01"]

ARXIV_QUERIES = ["transformer scaling laws", "diffusion model audio",
                 "RLHF alignment 2026", "mixture of experts routing",
                 "vision language models open", "graph neural network molecular",
                 "differential privacy federated", "speculative decoding inference"]

WIKI_TOPICS = ["Python_(programming_language)", "Quantum_computing",
               "Photosynthesis", "Roman_Empire", "Black_hole",
               "Renewable_energy", "World_War_II", "Artificial_intelligence",
               "DNA", "Plate_tectonics", "Internet", "Mount_Everest"]

GITHUB_REPOS = ["huggingface/transformers", "pytorch/pytorch", "tensorflow/tensorflow",
                "rust-lang/rust", "golang/go", "python/cpython",
                "facebook/react", "vercel/next.js", "vuejs/vue",
                "django/django", "fastapi/fastapi", "rails/rails",
                "ggerganov/llama.cpp", "openai/whisper", "anthropics/anthropic-sdk-python"]

# 45 tool names for variety
TOOLS = [
    "web_search", "fetch_url", "fetch_page_metadata", "extract_text_from_url",
    "extract_links", "extract_images", "screenshot_page", "http_get",
    "http_post", "http_put", "http_delete", "parse_html",
    "xpath_query", "css_select", "download_file", "check_url_status",
    "resolve_redirect", "expand_short_url", "get_robots_txt", "get_sitemap",
    "rss_fetch", "rss_parse", "twitter_search", "twitter_get_tweet",
    "reddit_search", "reddit_get_post", "hn_top_stories", "youtube_search",
    "youtube_video_info", "wikipedia_search", "wikipedia_summary", "arxiv_search",
    "github_search_repos", "github_get_readme", "news_search", "image_search",
    "translate_url_content", "archive_url", "submit_form", "cookie_jar_get",
    "url_encode", "url_decode", "parse_query_string", "build_url",
    "scrape_table", "extract_emails", "extract_phone_numbers", "ping_url",
]

assert len(set(TOOLS)) >= 40


def build_user_and_call(rng, tool):
    """Return (user_text, args_dict, tool_result_str_or_None)."""
    if tool == "web_search":
        q = rng.choice(SEARCH_QUERIES)
        verbs = [f"Search for the latest news on {q}.",
                 f"Run a web search for \"{q}\".",
                 f"Find me information about {q}.",
                 f"What's out there on {q}?",
                 f"Look up {q} on the web."]
        u = rng.choice(verbs)
        args = {"query": q, "limit": rng.choice([5, 10, 15, 20])}
        res = json.dumps({"results": [{"title": q.title(), "url": f"https://{rng.choice(DOMAINS)}/article"}], "count": args["limit"]})
        return u, args, res
    if tool == "fetch_url":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Fetch the contents of {url}.", f"Grab the page at {url}.",
                        f"Pull down the HTML from {url}.", f"Download {url} for me."])
        args = {"url": url}
        res = json.dumps({"status": 200, "bytes": rng.randint(1024, 50000)})
        return u, args, res
    if tool == "fetch_page_metadata":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"What's the title of the page at {url}?",
                        f"Get the metadata for {url}.",
                        f"Tell me the og:title and description for {url}."])
        args = {"url": url}
        res = json.dumps({"title": "About Us", "description": "Learn about our company"})
        return u, args, res
    if tool == "extract_text_from_url":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Strip the article text out of {url}.",
                        f"Give me the readable text from {url}.",
                        f"Extract just the body copy from {url}."])
        args = {"url": url}
        res = json.dumps({"text": "Article begins here...", "word_count": rng.randint(200, 3000)})
        return u, args, res
    if tool == "extract_links":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"List every link on {url}.", f"Pull all hrefs from {url}.",
                        f"What pages does {url} link out to?"])
        args = {"url": url}
        res = json.dumps({"links": [f"https://{rng.choice(DOMAINS)}/x" for _ in range(3)]})
        return u, args, res
    if tool == "extract_images":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Get the image URLs on {url}.",
                        f"List images embedded in {url}."])
        args = {"url": url}
        res = json.dumps({"images": [f"https://{rng.choice(DOMAINS)}/img/{i}.jpg" for i in range(2)]})
        return u, args, res
    if tool == "screenshot_page":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Take a screenshot of {url}.",
                        f"Capture {url} as an image.",
                        f"Render {url} to PNG."])
        args = {"url": url, "width": rng.choice([1024, 1280, 1920]), "full_page": rng.choice([True, False])}
        res = json.dumps({"path": "/tmp/shot.png", "size_kb": rng.randint(50, 800)})
        return u, args, res
    if tool == "http_get":
        url = f"https://api.{rng.choice(DOMAINS)}/v1/items/{rng.randint(100,999)}"
        u = rng.choice([f"Make a GET request to {url}.",
                        f"Hit {url} with a GET.",
                        f"Call the API at {url}."])
        args = {"url": url, "headers": {"Accept": "application/json"}}
        res = json.dumps({"status": 200, "body": {"id": rng.randint(100,999), "ok": True}})
        return u, args, res
    if tool == "http_post":
        url = f"https://api.{rng.choice(DOMAINS)}/v1/submit"
        u = rng.choice([f"POST a payload to {url}.",
                        f"Submit a form via POST to {url}."])
        args = {"url": url, "json": {"name": "test", "value": rng.randint(1,100)}}
        res = json.dumps({"status": 201, "id": rng.randint(1000,9999)})
        return u, args, res
    if tool == "http_put":
        url = f"https://api.{rng.choice(DOMAINS)}/v1/resource/{rng.randint(1,99)}"
        u = rng.choice([f"PUT an update to {url}.", f"Replace the resource at {url}."])
        args = {"url": url, "json": {"status": "active"}}
        res = json.dumps({"status": 200, "updated": True})
        return u, args, res
    if tool == "http_delete":
        url = f"https://api.{rng.choice(DOMAINS)}/v1/resource/{rng.randint(1,99)}"
        u = rng.choice([f"DELETE the resource at {url}.", f"Remove {url} via HTTP DELETE."])
        args = {"url": url}
        res = json.dumps({"status": 204, "deleted": True})
        return u, args, res
    if tool == "parse_html":
        u = rng.choice(["Parse this HTML snippet and tell me the h1.",
                        "Run an HTML parser over the markup I'm about to share."])
        args = {"html": "<html><body><h1>Hello</h1></body></html>"}
        res = json.dumps({"h1": ["Hello"], "links": []})
        return u, args, res
    if tool == "xpath_query":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        xp = rng.choice(["//h1/text()", "//a[@class='product']/@href", "//div[@id='main']//p"])
        u = rng.choice([f"Run XPath {xp} against {url}.",
                        f"Query {url} with XPath: {xp}"])
        args = {"url": url, "xpath": xp}
        res = json.dumps({"matches": ["Item one", "Item two"], "count": 2})
        return u, args, res
    if tool == "css_select":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        sel = rng.choice(["h2.headline", "div.product > a", ".price", "#main p"])
        u = rng.choice([f"Use CSS selector '{sel}' on {url}.",
                        f"Pull elements matching {sel} from {url}."])
        args = {"url": url, "selector": sel}
        res = json.dumps({"matches": 4, "samples": ["Sample text one"]})
        return u, args, res
    if tool == "download_file":
        url = f"https://{rng.choice(DOMAINS)}/files/report.pdf"
        u = rng.choice([f"Download the file at {url}.",
                        f"Save {url} to disk."])
        args = {"url": url, "dest": "/tmp/report.pdf"}
        res = json.dumps({"saved": "/tmp/report.pdf", "bytes": rng.randint(50000, 5000000)})
        return u, args, res
    if tool == "check_url_status":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Is {url} alive?", f"Check the HTTP status of {url}.",
                        f"Ping {url} and tell me the response code."])
        args = {"url": url}
        res = json.dumps({"status": rng.choice([200, 301, 404, 500]), "ok": True})
        return u, args, res
    if tool == "resolve_redirect":
        url = f"https://{rng.choice(DOMAINS)}/r/{rng.randint(1000,9999)}"
        u = rng.choice([f"Where does {url} redirect to?",
                        f"Follow redirects starting from {url}."])
        args = {"url": url}
        res = json.dumps({"final": f"https://{rng.choice(DOMAINS)}/landing", "hops": rng.randint(1,3)})
        return u, args, res
    if tool == "expand_short_url":
        s = rng.choice(["https://bit.ly/3xY9aZ", "https://t.co/abc123", "https://tinyurl.com/xyz"])
        u = rng.choice([f"Expand {s}.", f"Where does {s} actually point?"])
        args = {"url": s}
        res = json.dumps({"expanded": f"https://{rng.choice(DOMAINS)}/article/12345"})
        return u, args, res
    if tool == "get_robots_txt":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Pull the robots.txt for {d}.",
                        f"What does {d} disallow in robots.txt?"])
        args = {"domain": d}
        res = json.dumps({"user_agents": ["*"], "disallow": ["/admin", "/private"]})
        return u, args, res
    if tool == "get_sitemap":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Fetch the sitemap.xml for {d}.",
                        f"Show me {d}'s sitemap."])
        args = {"domain": d}
        res = json.dumps({"urls": rng.randint(50, 5000), "last_mod": "2026-04-30"})
        return u, args, res
    if tool == "rss_fetch":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Fetch the RSS feed from {d}.",
                        f"Grab the latest items from {d}/feed.rss."])
        args = {"url": f"https://{d}/feed.rss"}
        res = json.dumps({"items": rng.randint(10, 50), "title": "Daily News"})
        return u, args, res
    if tool == "rss_parse":
        u = rng.choice(["Parse this RSS XML I just pasted.",
                        "Run the RSS parser on the feed body."])
        args = {"xml": "<rss><channel><title>Feed</title><item><title>Post 1</title></item></channel></rss>"}
        res = json.dumps({"title": "Feed", "items": [{"title": "Post 1"}]})
        return u, args, res
    if tool == "twitter_search":
        q = rng.choice(TWITTER_QUERIES)
        u = rng.choice([f"Search Twitter for {q}.",
                        f"Find recent tweets matching {q}.",
                        f"What's being said on X about {q}?"])
        args = {"query": q, "max_results": rng.choice([10, 25, 50])}
        res = json.dumps({"tweets": rng.randint(0, 50), "newest_id": str(rng.randint(10**18, 10**19))})
        return u, args, res
    if tool == "twitter_get_tweet":
        tid = str(rng.randint(10**18, 10**19))
        u = rng.choice([f"Pull tweet {tid}.", f"Get the content of tweet id {tid}.",
                        f"Look up X post {tid}."])
        args = {"id": tid}
        res = json.dumps({"text": "Just shipped a new release.", "likes": rng.randint(0, 9999)})
        return u, args, res
    if tool == "reddit_search":
        q = rng.choice(["claude opus", "rust async", "fed rate cut", "GPT-5", "kubernetes"])
        sub = rng.choice(SUBREDDITS)
        u = rng.choice([f"Search r/{sub} for posts about {q}.",
                        f"Find Reddit threads on {q} in r/{sub}."])
        args = {"query": q, "subreddit": sub, "limit": rng.choice([10, 25])}
        res = json.dumps({"posts": rng.randint(0, 25), "subreddit": sub})
        return u, args, res
    if tool == "reddit_get_post":
        pid = "".join(rng.choices("abcdefghijklmnop0123456789", k=7))
        u = rng.choice([f"Get the Reddit post with id {pid}.",
                        f"Pull post {pid} including top comments."])
        args = {"id": pid, "include_comments": rng.choice([True, False])}
        res = json.dumps({"title": "TIL Reddit threads can be deep", "score": rng.randint(1, 5000)})
        return u, args, res
    if tool == "hn_top_stories":
        u = rng.choice(["What's on the Hacker News front page?",
                        "Pull the top HN stories right now.",
                        "Show me the current top 10 on Hacker News."])
        args = {"limit": rng.choice([10, 20, 30])}
        res = json.dumps({"stories": [{"id": rng.randint(10**7, 10**8), "title": "Show HN: My new tool"}]})
        return u, args, res
    if tool == "youtube_search":
        q = rng.choice(["rust tutorial", "linux kernel deep dive", "kubernetes intro",
                        "react server components", "transformer architecture"])
        u = rng.choice([f"Search YouTube for '{q}'.",
                        f"Find YouTube videos about {q}."])
        args = {"query": q, "max_results": rng.choice([5, 10, 20])}
        res = json.dumps({"videos": rng.randint(5, 20), "first_id": "dQw4w9WgXcQ"})
        return u, args, res
    if tool == "youtube_video_info":
        vid = "".join(rng.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", k=11))
        u = rng.choice([f"Get video info for YouTube id {vid}.",
                        f"Look up the metadata for YouTube video {vid}."])
        args = {"video_id": vid}
        res = json.dumps({"title": "Sample Video", "views": rng.randint(100, 10**7), "duration_s": rng.randint(60, 3600)})
        return u, args, res
    if tool == "wikipedia_search":
        q = rng.choice(["quantum entanglement", "Roman aqueducts", "great barrier reef",
                        "industrial revolution", "Pythagorean theorem"])
        u = rng.choice([f"Search Wikipedia for {q}.",
                        f"Find Wikipedia articles related to {q}."])
        args = {"query": q, "limit": rng.choice([5, 10])}
        res = json.dumps({"results": rng.randint(3, 10), "top": q.title()})
        return u, args, res
    if tool == "wikipedia_summary":
        t = rng.choice(WIKI_TOPICS)
        u = rng.choice([f"Give me the Wikipedia summary for {t.replace('_',' ')}.",
                        f"Pull the lead paragraph from Wikipedia for {t.replace('_',' ')}."])
        args = {"title": t}
        res = json.dumps({"extract": "Brief summary text...", "pageid": rng.randint(100, 99999)})
        return u, args, res
    if tool == "arxiv_search":
        q = rng.choice(ARXIV_QUERIES)
        u = rng.choice([f"Search arXiv for '{q}'.",
                        f"Find recent arXiv papers on {q}."])
        args = {"query": q, "max_results": rng.choice([5, 10, 25])}
        res = json.dumps({"papers": rng.randint(5, 25), "first_id": f"2604.{rng.randint(10000,99999)}"})
        return u, args, res
    if tool == "github_search_repos":
        q = rng.choice(["stars:>1000 language:rust", "topic:llm", "react hooks",
                        "language:go cli", "transformer pytorch"])
        u = rng.choice([f"Search GitHub repos for: {q}.",
                        f"Find GitHub repositories matching {q}."])
        args = {"query": q, "sort": rng.choice(["stars", "updated", "best-match"])}
        res = json.dumps({"total_count": rng.randint(100, 5000), "first": rng.choice(GITHUB_REPOS)})
        return u, args, res
    if tool == "github_get_readme":
        repo = rng.choice(GITHUB_REPOS)
        u = rng.choice([f"Fetch the README for {repo}.",
                        f"Pull the README.md from {repo}."])
        args = {"repo": repo}
        res = json.dumps({"length": rng.randint(1000, 50000), "encoding": "utf-8"})
        return u, args, res
    if tool == "news_search":
        q = rng.choice(SEARCH_QUERIES)
        u = rng.choice([f"Search news outlets for {q}.",
                        f"Find news articles about {q}."])
        args = {"query": q, "from": "2026-04-01", "limit": rng.choice([10, 25])}
        res = json.dumps({"articles": rng.randint(5, 50), "sources": rng.randint(2, 15)})
        return u, args, res
    if tool == "image_search":
        q = rng.choice(["sunset over ocean", "modern kitchen design", "border collie puppy",
                        "neural network diagram", "space telescope photo"])
        u = rng.choice([f"Find images of {q}.",
                        f"Search for pictures of {q}."])
        args = {"query": q, "size": rng.choice(["small", "medium", "large"])}
        res = json.dumps({"images": rng.randint(20, 200), "first": f"https://{rng.choice(DOMAINS)}/img.jpg"})
        return u, args, res
    if tool == "translate_url_content":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        lang = rng.choice(["en", "es", "fr", "de", "ja"])
        u = rng.choice([f"Translate the content of {url} into {lang}.",
                        f"Fetch {url} and translate to {lang}."])
        args = {"url": url, "target_lang": lang}
        res = json.dumps({"translated_chars": rng.randint(500, 20000), "source_lang": "en"})
        return u, args, res
    if tool == "archive_url":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Archive {url} on the Wayback Machine.",
                        f"Submit {url} to web.archive.org."])
        args = {"url": url}
        res = json.dumps({"archive_url": f"https://web.archive.org/web/2026050812000/{url}"})
        return u, args, res
    if tool == "submit_form":
        url = f"https://{rng.choice(DOMAINS)}/form"
        u = rng.choice([f"Submit a form to {url} with name=Alice.",
                        f"POST a contact-form submission to {url}."])
        args = {"url": url, "fields": {"name": "Alice", "email": "a@example.com"}}
        res = json.dumps({"status": 200, "confirmation": "ok"})
        return u, args, res
    if tool == "cookie_jar_get":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"What cookies are stored for {d}?",
                        f"Read the cookie jar for {d}."])
        args = {"domain": d}
        res = json.dumps({"cookies": rng.randint(0, 10), "session": True})
        return u, args, res
    if tool == "url_encode":
        s = rng.choice(["hello world", "café & restaurant", "a/b?c=d&e=f"])
        u = rng.choice([f"URL-encode the string '{s}'.",
                        f"Percent-encode '{s}' for me."])
        args = {"text": s}
        res = json.dumps({"encoded": "hello%20world"})
        return u, args, res
    if tool == "url_decode":
        s = rng.choice(["hello%20world", "caf%C3%A9", "a%2Fb%3Fc%3Dd"])
        u = rng.choice([f"Decode the URL-encoded string '{s}'.",
                        f"What does '{s}' decode to?"])
        args = {"text": s}
        res = json.dumps({"decoded": "hello world"})
        return u, args, res
    if tool == "parse_query_string":
        url = f"https://{rng.choice(DOMAINS)}/search?q=cats&page=2&sort=date"
        u = rng.choice([f"Parse the query string from {url}.",
                        f"Break out the query params in {url}."])
        args = {"url": url}
        res = json.dumps({"q": "cats", "page": "2", "sort": "date"})
        return u, args, res
    if tool == "build_url":
        u = rng.choice(["Build a URL with base example.com path /search query q=cats page=2.",
                        "Construct a URL from these parts."])
        args = {"base": "https://example.com", "path": "/search", "query": {"q": "cats", "page": 2}}
        res = json.dumps({"url": "https://example.com/search?q=cats&page=2"})
        return u, args, res
    if tool == "scrape_table":
        url = f"https://{rng.choice(DOMAINS)}/stats"
        u = rng.choice([f"Scrape the data table on {url}.",
                        f"Pull the HTML table from {url} into rows."])
        args = {"url": url, "table_index": rng.choice([0, 1, 2])}
        res = json.dumps({"rows": rng.randint(5, 200), "cols": rng.randint(2, 12)})
        return u, args, res
    if tool == "extract_emails":
        url = f"https://{rng.choice(DOMAINS)}/contact"
        u = rng.choice([f"Pull every email address from {url}.",
                        f"Extract emails listed on {url}."])
        args = {"url": url}
        res = json.dumps({"emails": ["info@example.com", "support@example.com"]})
        return u, args, res
    if tool == "extract_phone_numbers":
        url = f"https://{rng.choice(DOMAINS)}/contact"
        u = rng.choice([f"Find all phone numbers on {url}.",
                        f"Extract phone numbers from {url}."])
        args = {"url": url, "country": rng.choice(["US", "UK", "DE"])}
        res = json.dumps({"numbers": ["+1-555-0123", "+1-555-0199"]})
        return u, args, res
    if tool == "ping_url":
        url = f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"
        u = rng.choice([f"Ping {url} and report latency.",
                        f"Time a single GET against {url}."])
        args = {"url": url}
        res = json.dumps({"latency_ms": rng.randint(20, 800), "status": 200})
        return u, args, res
    raise ValueError(f"unknown tool {tool}")


def make_sample(rng, tool, single_turn, suffix_phrase):
    user, args, result = build_user_and_call(rng, tool)
    if single_turn:
        return {"messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
        ]}
    # multi-turn
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
        {"role": "tool", "name": tool, "content": result},
        {"role": "assistant", "content": f"{suffix_phrase}"},
    ]}


def main():
    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 500
    single_turn_count = 75  # ~15%
    multi_turn_count = total - single_turn_count  # 425

    # Per-tool cap: max 5% = 25 samples per tool.
    # 45 tools available; round-robin to balance.
    per_tool_target = total // len(TOOLS)  # 11
    extra = total - per_tool_target * len(TOOLS)  # remainder
    plan_tools = []
    for t in TOOLS:
        plan_tools.extend([t] * per_tool_target)
    # distribute extras
    extras_tools = rng.sample(TOOLS, extra)
    plan_tools.extend(extras_tools)
    assert len(plan_tools) == total
    rng.shuffle(plan_tools)

    # Suffix assignment for multi-turn: 14 uses each * 30 = 420; need 425 -> 5 phrases get +1 (15 uses).
    suffix_assignments = []
    for phrase in SUFFIX_POOL:
        suffix_assignments.extend([phrase] * 14)
    # 420 so far, need 5 more
    extras_suffix = rng.sample(SUFFIX_POOL, 5)
    suffix_assignments.extend(extras_suffix)
    assert len(suffix_assignments) == multi_turn_count
    rng.shuffle(suffix_assignments)

    # single-turn flags
    is_single = [True] * single_turn_count + [False] * multi_turn_count
    rng.shuffle(is_single)

    samples = []
    suffix_iter = iter(suffix_assignments)
    for i, tool in enumerate(plan_tools):
        single = is_single[i]
        suffix = "" if single else next(suffix_iter)
        samples.append(make_sample(rng, tool, single, suffix))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # stats
    tool_counts = {}
    single_actual = 0
    suffix_counts = {p: 0 for p in SUFFIX_POOL}
    for s in samples:
        msgs = s["messages"]
        last = msgs[-1]
        if last["role"] == "assistant" and last.get("tool_calls"):
            single_actual += 1
            tname = last["tool_calls"][0]["function"]["name"]
        else:
            # find tool call in msgs[2]
            tname = msgs[2]["tool_calls"][0]["function"]["name"]
            txt = last["content"]
            for p in SUFFIX_POOL:
                if txt.startswith(p):
                    suffix_counts[p] += 1
                    break
        tool_counts[tname] = tool_counts.get(tname, 0) + 1

    print(f"lines: {len(samples)}")
    print(f"distinct_tools: {len(tool_counts)}")
    print(f"single_turn: {single_actual}")
    print(f"multi_turn: {len(samples) - single_actual}")
    print(f"max_tool_share: {max(tool_counts.values())/len(samples):.4f} ({max(tool_counts, key=tool_counts.get)})")
    print(f"suffix_coverage: {sum(1 for v in suffix_counts.values() if v>0)}/30")
    print(f"suffix_min: {min(suffix_counts.values())}, suffix_max: {max(suffix_counts.values())}")


if __name__ == "__main__":
    main()
