#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_D2.py -- 500 ShareGPT tool-calling samples for TC-D2 (web/scraping/HTTP/social).

Fresh batch (wave 2). Domain corpora live in gen_tc_09_3_D2_data.py.
Seed: 1009308D
Output: datasets/tool-calling/raw-09.3/batch-08-D.jsonl
"""
import json
import random
from pathlib import Path

from gen_tc_09_3_D2_data import (
    DOMAINS, PATHS, SEARCH_QUERIES, SUBREDDITS, TWITTER_QUERIES, ARXIV_QUERIES,
    WIKI_TOPICS, GITHUB_REPOS, YT_SEARCHES, WIKI_QUERIES, GH_REPO_QUERIES,
    NEWS_QUERIES, IMAGE_QUERIES, REDDIT_TEXT_QUERIES, SHORT_URLS, LANGS,
    COUNTRIES, CSS_SELECTORS, XPATHS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-08-D.jsonl"

SEED = "1009308D"
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

BLACKLIST = [
    "I've gathered all the information",
    "I've completed the task",
    "Here's what I found:",
    "Based on the results,",
    "The results show that",
]

# 48 distinct tools (rename / add fresh ones vs wave 1)
TOOLS = [
    "web_search", "deep_web_search", "fetch_url", "fetch_page_metadata",
    "extract_text_from_url", "extract_links", "extract_images", "screenshot_page",
    "http_get", "http_post", "http_put", "http_delete", "http_patch", "http_head",
    "parse_html", "xpath_query", "css_select", "download_file",
    "check_url_status", "resolve_redirect", "expand_short_url",
    "get_robots_txt", "get_sitemap", "rss_fetch", "rss_parse", "atom_feed_fetch",
    "twitter_search", "twitter_get_tweet", "twitter_user_lookup",
    "reddit_search", "reddit_get_post", "reddit_user_about",
    "hn_top_stories", "hn_get_item",
    "youtube_search", "youtube_video_info", "youtube_channel_videos",
    "wikipedia_search", "wikipedia_summary",
    "arxiv_search", "github_search_repos", "github_get_readme", "github_list_issues",
    "news_search", "image_search", "translate_url_content",
    "archive_url", "wayback_lookup", "submit_form",
    "url_encode", "url_decode", "parse_query_string", "build_url",
    "scrape_table", "extract_emails", "extract_phone_numbers", "ping_url",
    "dns_lookup", "whois_lookup", "ssl_cert_info", "ip_geolocate",
    "mastodon_search", "bluesky_search", "linkedin_company_lookup",
    "discord_webhook_post", "slack_post_message", "telegram_send_message",
    "stackoverflow_search", "producthunt_today",
]
assert len(set(TOOLS)) >= 40

# subset for plan to keep cap <=5%; using 50 plan tools each ~10 samples
TOOL_PLAN = TOOLS[:50] if len(TOOLS) >= 50 else TOOLS


def _url(rng):
    return f"https://{rng.choice(DOMAINS)}{rng.choice(PATHS)}"


def _api(rng):
    return f"https://api.{rng.choice(DOMAINS)}/v{rng.choice([1,2,3])}/{rng.choice(['items','records','events','users','assets'])}/{rng.randint(100,9999)}"


def build_user_and_call(rng, tool):
    if tool == "web_search":
        q = rng.choice(SEARCH_QUERIES)
        u = rng.choice([
            f"Run a search across the web for: {q}",
            f"Could you look online for {q}?",
            f"I need search results on '{q}' — top hits please.",
            f"Web search this: {q}",
        ])
        args = {"query": q, "num_results": rng.choice([5, 10, 20, 30])}
        res = json.dumps({"hits": [{"title": q[:40], "url": _url(rng)}], "total": args["num_results"]})
        return u, args, res
    if tool == "deep_web_search":
        q = rng.choice(SEARCH_QUERIES)
        u = rng.choice([
            f"Do a deep search and crawl the top results for: {q}",
            f"Run a research-grade web sweep on {q}.",
        ])
        args = {"query": q, "depth": rng.choice([2, 3]), "max_pages": rng.choice([10, 20])}
        res = json.dumps({"pages_crawled": args["max_pages"], "summary_chars": rng.randint(2000, 9000)})
        return u, args, res
    if tool == "fetch_url":
        url = _url(rng)
        u = rng.choice([f"Could you retrieve {url}?", f"Hit {url} and bring back the body.",
                        f"Fetch the page at {url} for me."])
        args = {"url": url, "timeout_s": rng.choice([5, 10, 30])}
        res = json.dumps({"status": 200, "bytes": rng.randint(2048, 80000), "content_type": "text/html"})
        return u, args, res
    if tool == "fetch_page_metadata":
        url = _url(rng)
        u = rng.choice([f"What does the og:image look like for {url}?",
                        f"Pull the meta tags from {url}.",
                        f"Read out the page metadata at {url}."])
        args = {"url": url}
        res = json.dumps({"og:title": "Article", "og:image": _url(rng) + ".jpg", "twitter:card": "summary_large_image"})
        return u, args, res
    if tool == "extract_text_from_url":
        url = _url(rng)
        u = rng.choice([f"Pull the readable body from {url}.",
                        f"Strip boilerplate and give me the article from {url}."])
        args = {"url": url, "include_links": rng.choice([True, False])}
        res = json.dumps({"text": "Article body...", "word_count": rng.randint(300, 4000)})
        return u, args, res
    if tool == "extract_links":
        url = _url(rng)
        u = rng.choice([f"Enumerate outgoing links on {url}.",
                        f"Pull every anchor href from {url}."])
        args = {"url": url, "external_only": rng.choice([True, False])}
        res = json.dumps({"links": [_url(rng) for _ in range(4)]})
        return u, args, res
    if tool == "extract_images":
        url = _url(rng)
        u = rng.choice([f"Get all <img> sources on {url}.",
                        f"List image URLs embedded in {url}."])
        args = {"url": url, "min_width": rng.choice([0, 200, 600])}
        res = json.dumps({"images": [_url(rng) + f"/i{i}.png" for i in range(3)]})
        return u, args, res
    if tool == "screenshot_page":
        url = _url(rng)
        u = rng.choice([f"Snapshot {url} as PNG.",
                        f"Render {url} into a screenshot please."])
        args = {"url": url, "viewport": rng.choice(["1280x720", "1440x900", "1920x1080"]),
                "full_page": rng.choice([True, False])}
        res = json.dumps({"path": "/tmp/page.png", "kb": rng.randint(60, 1200)})
        return u, args, res
    if tool == "http_get":
        url = _api(rng)
        u = rng.choice([f"GET {url} with auth bearer.",
                        f"Issue a GET to {url}.", f"Call the endpoint {url}."])
        args = {"url": url, "headers": {"Authorization": "Bearer XXX", "Accept": "application/json"}}
        res = json.dumps({"status": 200, "body": {"id": rng.randint(1,9999), "active": True}})
        return u, args, res
    if tool == "http_post":
        url = _api(rng)
        u = rng.choice([f"POST a JSON body to {url}.", f"Submit a payload to {url} via POST."])
        args = {"url": url, "json": {"action": "create", "value": rng.randint(1, 1000)}}
        res = json.dumps({"status": 201, "id": rng.randint(10000, 99999)})
        return u, args, res
    if tool == "http_put":
        url = _api(rng)
        u = rng.choice([f"PUT a replacement record to {url}.", f"Overwrite the resource at {url}."])
        args = {"url": url, "json": {"name": "alpha", "active": False}}
        res = json.dumps({"status": 200, "etag": "W/\"abc123\""})
        return u, args, res
    if tool == "http_delete":
        url = _api(rng)
        u = rng.choice([f"DELETE the record at {url}.", f"Remove {url} via DELETE."])
        args = {"url": url}
        res = json.dumps({"status": 204})
        return u, args, res
    if tool == "http_patch":
        url = _api(rng)
        u = rng.choice([f"PATCH {url} to flip status to inactive.",
                        f"Apply a partial update to {url}."])
        args = {"url": url, "json": {"status": "inactive"}}
        res = json.dumps({"status": 200, "patched": True})
        return u, args, res
    if tool == "http_head":
        url = _url(rng)
        u = rng.choice([f"Send a HEAD request to {url}.",
                        f"Just give me the headers from {url} via HEAD."])
        args = {"url": url}
        res = json.dumps({"status": 200, "content_length": rng.randint(1000, 200000), "last_modified": "2026-04-30"})
        return u, args, res
    if tool == "parse_html":
        u = rng.choice(["Parse this HTML chunk and surface every h2.",
                        "Run a parser over the snippet I'm pasting in."])
        args = {"html": "<html><body><h2>Section A</h2><h2>Section B</h2></body></html>",
                "selector": "h2"}
        res = json.dumps({"matches": ["Section A", "Section B"]})
        return u, args, res
    if tool == "xpath_query":
        url = _url(rng)
        xp = rng.choice(XPATHS)
        u = rng.choice([f"Apply XPath {xp} to {url}.", f"XPath query {xp} against {url}."])
        args = {"url": url, "xpath": xp}
        res = json.dumps({"matches": ["val1", "val2", "val3"], "count": 3})
        return u, args, res
    if tool == "css_select":
        url = _url(rng)
        sel = rng.choice(CSS_SELECTORS)
        u = rng.choice([f"Use CSS selector {sel!r} on {url}.",
                        f"Pull elements matching {sel!r} out of {url}."])
        args = {"url": url, "selector": sel}
        res = json.dumps({"matches": rng.randint(1, 25), "first_text": "Heading sample"})
        return u, args, res
    if tool == "download_file":
        url = f"https://{rng.choice(DOMAINS)}/downloads/asset-{rng.randint(100,999)}.{rng.choice(['zip','pdf','tar.gz','csv'])}"
        u = rng.choice([f"Save {url} locally.", f"Download {url} into /tmp."])
        args = {"url": url, "dest": f"/tmp/asset.{url.rsplit('.',1)[-1]}"}
        res = json.dumps({"saved": args["dest"], "bytes": rng.randint(100000, 10000000)})
        return u, args, res
    if tool == "check_url_status":
        url = _url(rng)
        u = rng.choice([f"Status code on {url}?", f"Is {url} returning 200?",
                        f"Quick health check on {url}."])
        args = {"url": url, "follow_redirects": rng.choice([True, False])}
        res = json.dumps({"status": rng.choice([200, 301, 403, 404, 503]), "elapsed_ms": rng.randint(40, 700)})
        return u, args, res
    if tool == "resolve_redirect":
        url = f"https://{rng.choice(DOMAINS)}/go/{rng.randint(1000,9999)}"
        u = rng.choice([f"Trace redirects from {url}.", f"Where does {url} land?"])
        args = {"url": url, "max_hops": rng.choice([3, 5, 10])}
        res = json.dumps({"final_url": _url(rng), "hops": rng.randint(1, 4)})
        return u, args, res
    if tool == "expand_short_url":
        s = rng.choice(SHORT_URLS)
        u = rng.choice([f"Expand short link {s}.", f"What's behind the shortlink {s}?"])
        args = {"short_url": s}
        res = json.dumps({"expanded": _url(rng)})
        return u, args, res
    if tool == "get_robots_txt":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"What's in robots.txt for {d}?", f"Pull {d}'s crawler rules."])
        args = {"domain": d, "user_agent": rng.choice(["*", "Googlebot", "Bingbot"])}
        res = json.dumps({"allow": ["/"], "disallow": ["/private", "/checkout"], "crawl_delay": rng.choice([0, 1, 5])})
        return u, args, res
    if tool == "get_sitemap":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Grab the sitemap for {d}.", f"Pull every URL in {d}'s sitemap.xml."])
        args = {"domain": d}
        res = json.dumps({"url_count": rng.randint(100, 50000), "last_modified": "2026-05-01"})
        return u, args, res
    if tool == "rss_fetch":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Pull the RSS feed at {d}/feed.", f"Subscribe-fetch {d}'s RSS."])
        args = {"feed_url": f"https://{d}/feed", "limit": rng.choice([10, 25, 50])}
        res = json.dumps({"items_returned": args["limit"], "channel_title": "Site Updates"})
        return u, args, res
    if tool == "rss_parse":
        u = rng.choice(["Parse this RSS XML payload.", "Run the RSS parser on the body I'm pasting."])
        args = {"xml": "<rss><channel><title>Updates</title><item><title>Hello</title></item></channel></rss>"}
        res = json.dumps({"channel": "Updates", "item_count": 1})
        return u, args, res
    if tool == "atom_feed_fetch":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Fetch {d}'s Atom feed.", f"Grab the Atom XML from {d}/atom.xml."])
        args = {"feed_url": f"https://{d}/atom.xml"}
        res = json.dumps({"entries": rng.randint(5, 40), "updated": "2026-05-07T10:00:00Z"})
        return u, args, res
    if tool == "twitter_search":
        q = rng.choice(TWITTER_QUERIES)
        u = rng.choice([f"Search X for {q}.", f"Find recent posts on Twitter matching {q}."])
        args = {"query": q, "max_results": rng.choice([10, 25, 50, 100])}
        res = json.dumps({"count": rng.randint(0, 100), "newest_id": str(rng.randint(10**18, 10**19))})
        return u, args, res
    if tool == "twitter_get_tweet":
        tid = str(rng.randint(10**18, 10**19))
        u = rng.choice([f"Pull X post {tid}.", f"Fetch tweet {tid} with engagement stats."])
        args = {"tweet_id": tid, "expansions": rng.choice(["author_id", "attachments.media_keys"])}
        res = json.dumps({"text": "Sample post body.", "likes": rng.randint(0, 50000), "retweets": rng.randint(0, 5000)})
        return u, args, res
    if tool == "twitter_user_lookup":
        handle = rng.choice(["sama", "karpathy", "swyx", "elonmusk", "patio11", "dhh", "yoheinakajima"])
        u = rng.choice([f"Look up X profile @{handle}.", f"Get bio info for @{handle} on X."])
        args = {"username": handle}
        res = json.dumps({"id": str(rng.randint(10**8, 10**10)), "followers": rng.randint(1000, 10**7)})
        return u, args, res
    if tool == "reddit_search":
        q = rng.choice(REDDIT_TEXT_QUERIES)
        sub = rng.choice(SUBREDDITS)
        u = rng.choice([f"Search r/{sub} for '{q}'.", f"Find threads about {q} on r/{sub}."])
        args = {"query": q, "subreddit": sub, "sort": rng.choice(["relevance", "new", "top"])}
        res = json.dumps({"posts_returned": rng.randint(0, 25), "subreddit": sub})
        return u, args, res
    if tool == "reddit_get_post":
        pid = "".join(rng.choices("abcdefghijklmnop0123456789", k=7))
        u = rng.choice([f"Pull Reddit post {pid}.", f"Fetch r/post/{pid} with comments."])
        args = {"post_id": pid, "comment_depth": rng.choice([1, 3, 5])}
        res = json.dumps({"title": "Discussion thread", "score": rng.randint(0, 30000), "num_comments": rng.randint(0, 2000)})
        return u, args, res
    if tool == "reddit_user_about":
        user = rng.choice(["spez", "kn0thing", "gallowboob", "shitty_watercolour", "Unidan"])
        u = rng.choice([f"Look up Reddit user {user}.", f"Get karma stats for /u/{user}."])
        args = {"username": user}
        res = json.dumps({"link_karma": rng.randint(100, 10**6), "comment_karma": rng.randint(100, 10**6)})
        return u, args, res
    if tool == "hn_top_stories":
        u = rng.choice(["What's hot on Hacker News?", "Pull the HN front page.",
                        "Top stories on HN right now please."])
        args = {"limit": rng.choice([10, 20, 50])}
        res = json.dumps({"ids": [rng.randint(10**7, 10**8) for _ in range(5)]})
        return u, args, res
    if tool == "hn_get_item":
        iid = rng.randint(10**7, 10**8)
        u = rng.choice([f"Fetch HN item {iid}.", f"Pull the HN story with id {iid}."])
        args = {"item_id": iid}
        res = json.dumps({"title": "Show HN: A new tool", "by": "alice", "score": rng.randint(1, 2000)})
        return u, args, res
    if tool == "youtube_search":
        q = rng.choice(YT_SEARCHES)
        u = rng.choice([f"Search YouTube for {q!r}.", f"Find YT videos on {q}."])
        args = {"query": q, "max_results": rng.choice([5, 10, 25, 50])}
        res = json.dumps({"video_count": rng.randint(5, 50), "first_id": "".join(rng.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", k=11))})
        return u, args, res
    if tool == "youtube_video_info":
        vid = "".join(rng.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", k=11))
        u = rng.choice([f"YT metadata for {vid}.", f"Get title/views for YouTube video {vid}."])
        args = {"video_id": vid, "include_captions": rng.choice([True, False])}
        res = json.dumps({"title": "Tutorial: build it yourself", "views": rng.randint(1000, 5*10**7), "length_s": rng.randint(120, 7200)})
        return u, args, res
    if tool == "youtube_channel_videos":
        cid = "UC" + "".join(rng.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-", k=22))
        u = rng.choice([f"List recent uploads from YT channel {cid}.",
                        f"Pull the latest videos posted by channel {cid}."])
        args = {"channel_id": cid, "limit": rng.choice([10, 25, 50])}
        res = json.dumps({"videos_returned": args["limit"], "channel_id": cid})
        return u, args, res
    if tool == "wikipedia_search":
        q = rng.choice(WIKI_QUERIES)
        u = rng.choice([f"Search Wikipedia for: {q}.", f"Find articles on {q}."])
        args = {"query": q, "limit": rng.choice([5, 10, 20])}
        res = json.dumps({"hits": rng.randint(3, 20), "top": q[:30].title()})
        return u, args, res
    if tool == "wikipedia_summary":
        t = rng.choice(WIKI_TOPICS)
        u = rng.choice([f"Wikipedia summary for {t.replace('_',' ')}.",
                        f"Lead paragraph from Wikipedia for '{t.replace('_',' ')}'."])
        args = {"title": t, "sentences": rng.choice([2, 3, 5])}
        res = json.dumps({"extract": "Summary text...", "page_id": rng.randint(1000, 9999999)})
        return u, args, res
    if tool == "arxiv_search":
        q = rng.choice(ARXIV_QUERIES)
        u = rng.choice([f"Search arXiv on {q}.", f"Find papers about {q}."])
        args = {"query": q, "max_results": rng.choice([5, 10, 20]),
                "sort_by": rng.choice(["relevance", "lastUpdatedDate", "submittedDate"])}
        res = json.dumps({"papers": args["max_results"], "first_id": f"2605.{rng.randint(10000,99999)}"})
        return u, args, res
    if tool == "github_search_repos":
        q = rng.choice(GH_REPO_QUERIES)
        u = rng.choice([f"Search GitHub repos for: {q}.", f"Find GH repos matching {q}."])
        args = {"query": q, "sort": rng.choice(["stars", "forks", "updated"]), "per_page": rng.choice([10, 25])}
        res = json.dumps({"total_count": rng.randint(50, 9000), "first_repo": rng.choice(GITHUB_REPOS)})
        return u, args, res
    if tool == "github_get_readme":
        repo = rng.choice(GITHUB_REPOS)
        u = rng.choice([f"Fetch README for {repo}.", f"Pull {repo}'s README.md."])
        args = {"repo": repo, "ref": rng.choice(["main", "master", "develop"])}
        res = json.dumps({"size_bytes": rng.randint(2000, 80000), "encoding": "utf-8"})
        return u, args, res
    if tool == "github_list_issues":
        repo = rng.choice(GITHUB_REPOS)
        u = rng.choice([f"List open issues on {repo}.", f"Pull recent GH issues for {repo}."])
        args = {"repo": repo, "state": rng.choice(["open", "closed", "all"]),
                "labels": rng.choice(["bug", "enhancement", "good first issue"])}
        res = json.dumps({"issues_returned": rng.randint(0, 30), "repo": repo})
        return u, args, res
    if tool == "news_search":
        q = rng.choice(NEWS_QUERIES)
        u = rng.choice([f"Search news for: {q}.", f"Find recent articles on {q}."])
        args = {"query": q, "from_date": "2026-04-01", "language": rng.choice(["en", "es", "fr"])}
        res = json.dumps({"articles": rng.randint(5, 80), "sources": rng.randint(2, 25)})
        return u, args, res
    if tool == "image_search":
        q = rng.choice(IMAGE_QUERIES)
        u = rng.choice([f"Find images of {q}.", f"Search for photos of {q}."])
        args = {"query": q, "size": rng.choice(["any", "medium", "large", "xlarge"]),
                "license": rng.choice(["any", "creative_commons", "public_domain"])}
        res = json.dumps({"image_count": rng.randint(20, 500), "first": _url(rng) + ".jpg"})
        return u, args, res
    if tool == "translate_url_content":
        url = _url(rng)
        lang = rng.choice(LANGS)
        u = rng.choice([f"Translate the page at {url} to {lang}.",
                        f"Render {url} contents in {lang}."])
        args = {"url": url, "target_language": lang}
        res = json.dumps({"chars_translated": rng.randint(800, 25000), "detected_source": "en"})
        return u, args, res
    if tool == "archive_url":
        url = _url(rng)
        u = rng.choice([f"Save {url} on the Wayback Machine.",
                        f"Snapshot {url} via web.archive.org."])
        args = {"url": url}
        res = json.dumps({"snapshot": f"https://web.archive.org/web/2026050812{rng.randint(1000,9999)}/{url}"})
        return u, args, res
    if tool == "wayback_lookup":
        url = _url(rng)
        u = rng.choice([f"What did {url} look like in 2018?",
                        f"Find a Wayback snapshot of {url} from 2020."])
        args = {"url": url, "timestamp": rng.choice(["20180601", "20200115", "20211231"])}
        res = json.dumps({"snapshot_url": f"https://web.archive.org/web/{args['timestamp']}/{url}", "available": True})
        return u, args, res
    if tool == "submit_form":
        url = f"https://{rng.choice(DOMAINS)}/contact"
        u = rng.choice([f"POST a contact form to {url} with my name.",
                        f"Submit form fields to {url}."])
        args = {"url": url, "fields": {"name": "Sam", "email": "sam@example.com", "message": "Hi"}}
        res = json.dumps({"status": 200, "ticket_id": f"T-{rng.randint(1000,9999)}"})
        return u, args, res
    if tool == "url_encode":
        s = rng.choice(["coffee & tea", "naïve résumé", "100% wool", "a=b/c d", "русский текст"])
        u = rng.choice([f"URL-encode: {s!r}.", f"Percent-encode this string: {s!r}."])
        args = {"text": s}
        res = json.dumps({"encoded": "coffee%20%26%20tea"})
        return u, args, res
    if tool == "url_decode":
        s = rng.choice(["coffee%20%26%20tea", "na%C3%AFve", "100%25%20wool"])
        u = rng.choice([f"Decode {s!r}.", f"What does {s!r} decode to?"])
        args = {"text": s}
        res = json.dumps({"decoded": "coffee & tea"})
        return u, args, res
    if tool == "parse_query_string":
        url = f"https://{rng.choice(DOMAINS)}/results?q=laptop&min_price=500&max_price=1500&sort=rating"
        u = rng.choice([f"Break out the query params from {url}.",
                        f"Parse the querystring on {url}."])
        args = {"url": url}
        res = json.dumps({"q": "laptop", "min_price": "500", "max_price": "1500", "sort": "rating"})
        return u, args, res
    if tool == "build_url":
        u = rng.choice(["Construct a URL: base example.org, path /api/items, query: page=3, limit=50.",
                        "Build a URL for me with a query string."])
        args = {"base": f"https://{rng.choice(DOMAINS)}", "path": "/api/items",
                "query": {"page": 3, "limit": 50}}
        res = json.dumps({"url": args["base"] + "/api/items?page=3&limit=50"})
        return u, args, res
    if tool == "scrape_table":
        url = f"https://{rng.choice(DOMAINS)}/leaderboard"
        u = rng.choice([f"Scrape the leaderboard table on {url}.",
                        f"Extract the HTML table from {url}."])
        args = {"url": url, "table_index": rng.choice([0, 1, 2]),
                "header_row": rng.choice([True, False])}
        res = json.dumps({"rows": rng.randint(10, 500), "cols": rng.randint(3, 15)})
        return u, args, res
    if tool == "extract_emails":
        url = f"https://{rng.choice(DOMAINS)}/about"
        u = rng.choice([f"Pull email addresses from {url}.",
                        f"Find any contact emails on {url}."])
        args = {"url": url, "dedupe": rng.choice([True, False])}
        res = json.dumps({"emails": ["press@example.com", "sales@example.com", "hello@example.com"]})
        return u, args, res
    if tool == "extract_phone_numbers":
        url = f"https://{rng.choice(DOMAINS)}/contact"
        u = rng.choice([f"Find phone numbers on {url}.",
                        f"Extract all phones listed at {url}."])
        args = {"url": url, "country": rng.choice(COUNTRIES)}
        res = json.dumps({"numbers": ["+1-202-555-0145", "+44-20-7946-0958"]})
        return u, args, res
    if tool == "ping_url":
        url = _url(rng)
        u = rng.choice([f"Ping {url} for latency.", f"Time a single hit against {url}."])
        args = {"url": url, "samples": rng.choice([1, 3, 5])}
        res = json.dumps({"avg_latency_ms": rng.randint(15, 600), "loss": 0})
        return u, args, res
    if tool == "dns_lookup":
        host = rng.choice(DOMAINS)
        u = rng.choice([f"Resolve A records for {host}.",
                        f"What's the DNS for {host}?"])
        args = {"hostname": host, "record_type": rng.choice(["A", "AAAA", "MX", "TXT", "CNAME"])}
        res = json.dumps({"records": ["192.0.2." + str(rng.randint(1, 254))], "ttl": rng.choice([60, 300, 3600])})
        return u, args, res
    if tool == "whois_lookup":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Pull whois for {d}.", f"Who registered {d}?"])
        args = {"domain": d}
        res = json.dumps({"registrar": "MarkMonitor", "creation_date": "2010-03-12", "expiration": "2027-03-12"})
        return u, args, res
    if tool == "ssl_cert_info":
        d = rng.choice(DOMAINS)
        u = rng.choice([f"Inspect the SSL cert for {d}.",
                        f"When does {d}'s TLS cert expire?"])
        args = {"hostname": d, "port": 443}
        res = json.dumps({"issuer": "Let's Encrypt", "valid_until": "2026-08-15", "san_count": rng.randint(1, 25)})
        return u, args, res
    if tool == "ip_geolocate":
        ip = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        u = rng.choice([f"Where is IP {ip} located?",
                        f"Geolocate {ip} for me."])
        args = {"ip": ip}
        res = json.dumps({"country": rng.choice(["US", "DE", "JP", "BR"]), "city": "Anywhere", "lat": 40.0, "lon": -74.0})
        return u, args, res
    if tool == "mastodon_search":
        q = rng.choice(["#opensource", "fediverse", "selfhosting", "mastodev"])
        u = rng.choice([f"Search Mastodon for {q}.",
                        f"Find recent toots on {q}."])
        args = {"query": q, "instance": rng.choice(["mastodon.social", "fosstodon.org", "hachyderm.io"]),
                "limit": rng.choice([10, 25, 50])}
        res = json.dumps({"toots": rng.randint(0, 50), "instance": args["instance"]})
        return u, args, res
    if tool == "bluesky_search":
        q = rng.choice(["#bsky", "AT protocol", "did:plc"])
        u = rng.choice([f"Search Bluesky for {q}.",
                        f"Find Bluesky posts on {q}."])
        args = {"query": q, "limit": rng.choice([10, 25, 50])}
        res = json.dumps({"posts": rng.randint(0, 50), "cursor": "abc123"})
        return u, args, res
    if tool == "linkedin_company_lookup":
        co = rng.choice(["openai", "anthropic", "huggingface", "stripe", "supabase", "vercel"])
        u = rng.choice([f"Look up the LinkedIn company page for {co}.",
                        f"Get LinkedIn employee count for {co}."])
        args = {"company_slug": co}
        res = json.dumps({"name": co.title(), "employees": rng.randint(50, 50000), "industry": "Software"})
        return u, args, res
    if tool == "discord_webhook_post":
        u = rng.choice(["Post 'Build passed' to the deploys Discord channel via webhook.",
                        "Send a Discord webhook ping with the deploy summary."])
        args = {"webhook_url": f"https://discord.com/api/webhooks/{rng.randint(10**18, 10**19)}/abcdef",
                "content": "Build passed on main", "username": "ci-bot"}
        res = json.dumps({"status": 204, "delivered": True})
        return u, args, res
    if tool == "slack_post_message":
        u = rng.choice(["Post 'Standup at 10' into #team-eng on Slack.",
                        "Send a Slack message to #alerts with deploy info."])
        args = {"channel": rng.choice(["#team-eng", "#alerts", "#general"]),
                "text": "Heads up — release going out at 10am.", "as_user": True}
        res = json.dumps({"ok": True, "ts": f"{rng.randint(10**9, 10**10)}.{rng.randint(100000, 999999)}"})
        return u, args, res
    if tool == "telegram_send_message":
        u = rng.choice(["Send a Telegram message to chat 123456.",
                        "Push 'meeting started' over Telegram bot API."])
        args = {"chat_id": rng.randint(10**5, 10**9), "text": "Meeting starting now",
                "parse_mode": rng.choice(["Markdown", "HTML", "MarkdownV2"])}
        res = json.dumps({"ok": True, "message_id": rng.randint(1000, 99999)})
        return u, args, res
    if tool == "stackoverflow_search":
        q = rng.choice(["pandas merge on multiple columns", "rust async lifetimes",
                        "kubernetes pod stuck terminating", "tailwind dark mode toggle",
                        "git rebase preserve merges"])
        u = rng.choice([f"Search Stack Overflow for {q!r}.",
                        f"Find SO answers to: {q}."])
        args = {"query": q, "tags": rng.choice([["python"], ["rust"], ["kubernetes"], ["css"], ["git"]]),
                "sort": rng.choice(["votes", "newest", "relevance"])}
        res = json.dumps({"questions": rng.randint(5, 50), "top_score": rng.randint(10, 5000)})
        return u, args, res
    if tool == "producthunt_today":
        u = rng.choice(["What's launching on Product Hunt today?",
                        "Pull today's top Product Hunt launches."])
        args = {"limit": rng.choice([5, 10, 25])}
        res = json.dumps({"products": rng.randint(5, 30), "first": "AI Pencil 3"})
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
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "",
         "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
        {"role": "tool", "name": tool, "content": result},
        {"role": "assistant", "content": suffix_phrase},
    ]}


def main():
    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 500
    single_turn_count = 75
    multi_turn_count = total - single_turn_count  # 425

    n_tools = len(TOOLS)
    per_tool_target = total // n_tools  # with 68 tools => 7
    plan_tools = []
    for t in TOOLS:
        plan_tools.extend([t] * per_tool_target)
    extras_needed = total - len(plan_tools)
    extras_tools = rng.sample(TOOLS, extras_needed) if extras_needed <= n_tools else (TOOLS * ((extras_needed // n_tools) + 1))[:extras_needed]
    plan_tools.extend(extras_tools)
    assert len(plan_tools) == total
    rng.shuffle(plan_tools)

    # 14 each * 30 = 420; need 425 -> 5 phrases get +1
    suffix_assignments = []
    for phrase in SUFFIX_POOL:
        suffix_assignments.extend([phrase] * 14)
    extras_suffix = rng.sample(SUFFIX_POOL, multi_turn_count - len(suffix_assignments))
    suffix_assignments.extend(extras_suffix)
    assert len(suffix_assignments) == multi_turn_count
    rng.shuffle(suffix_assignments)

    is_single = [True] * single_turn_count + [False] * multi_turn_count
    rng.shuffle(is_single)

    samples = []
    suffix_iter = iter(suffix_assignments)
    for i, tool in enumerate(plan_tools):
        single = is_single[i]
        suffix = "" if single else next(suffix_iter)
        samples.append(make_sample(rng, tool, single, suffix))

    # blacklist scan
    bl_hits = 0
    for s in samples:
        for m in s["messages"]:
            if m.get("content"):
                for bl in BLACKLIST:
                    if bl in m["content"]:
                        bl_hits += 1
    assert bl_hits == 0, f"blacklist phrase leaked: {bl_hits}"

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
    top = max(tool_counts, key=tool_counts.get)
    print(f"max_tool_share: {tool_counts[top]/len(samples):.4f} ({top}, n={tool_counts[top]})")
    print(f"suffix_coverage: {sum(1 for v in suffix_counts.values() if v>0)}/30")
    print(f"suffix_min: {min(suffix_counts.values())}, suffix_max: {max(suffix_counts.values())}")
    print(f"blacklist_hits: {bl_hits}")


if __name__ == "__main__":
    main()
