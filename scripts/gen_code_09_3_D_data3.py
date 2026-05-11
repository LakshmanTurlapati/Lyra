# SPDX-License-Identifier: MIT
"""Web/API (73) + data science (28) entries. Wave 4 part 3."""


def register_all(add):
    # ---- web / API (73) ----
    add("web",
        "Write a function that talks to a SQLite DB through a context-managed connection.",
        "```python\nimport sqlite3\nfrom contextlib import contextmanager\n\n@contextmanager\ndef db(path: str):\n    conn = sqlite3.connect(path)\n    conn.row_factory = sqlite3.Row\n    try:\n        yield conn\n        conn.commit()\n    except Exception:\n        conn.rollback()\n        raise\n    finally:\n        conn.close()\n```\nWith `row_factory = Row` you get dict-like access on results.")

    add("web",
        "Implement a FastAPI healthcheck that pings a database.",
        "```python\nfrom fastapi import FastAPI, HTTPException, status\n\napp = FastAPI()\n\n@app.get('/health')\ndef health() -> dict:\n    try:\n        ping_db()  # implementation-specific\n    except Exception as e:\n        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e))\n    return {'status': 'ok'}\n```\nKeep this endpoint cheap; load balancers may hit it many times per second.")

    add("web",
        "Write a function that uses requests-cache to cache GETs.",
        "```python\nimport requests_cache\n\nsession = requests_cache.CachedSession('http_cache', expire_after=300)\n\ndef get(url: str) -> dict:\n    r = session.get(url, timeout=10.0)\n    r.raise_for_status()\n    return r.json()\n```\nDrop-in replacement for the requests Session API; survives across runs.")

    add("web",
        "Implement a Flask 'after_request' hook that adds security headers.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.after_request\ndef security_headers(resp):\n    resp.headers.setdefault('X-Content-Type-Options', 'nosniff')\n    resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')\n    resp.headers.setdefault('Strict-Transport-Security', 'max-age=63072000; includeSubDomains')\n    return resp\n```\nUse `setdefault` so route-specific overrides win.")

    add("web",
        "Write a function that retries on 429 honoring Retry-After.",
        "```python\nimport time\nimport httpx\n\ndef get_with_retry(url: str, attempts: int = 5) -> httpx.Response:\n    with httpx.Client(timeout=10.0) as client:\n        for i in range(attempts):\n            r = client.get(url)\n            if r.status_code != 429:\n                r.raise_for_status()\n                return r\n            wait = float(r.headers.get('retry-after', 1))\n            time.sleep(wait)\n    raise RuntimeError('rate limit -- gave up')\n```\nRespect `Retry-After`; spamming retries makes the limit worse.")

    add("web",
        "Implement a FastAPI endpoint that returns plain text.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import PlainTextResponse\n\napp = FastAPI()\n\n@app.get('/version', response_class=PlainTextResponse)\ndef version() -> str:\n    return '1.0.0'\n```\n`PlainTextResponse` skips JSON serialization and sets the right content type.")

    add("web",
        "Write a function that parses a Cookie header into a dict.",
        "```python\nfrom http.cookies import SimpleCookie\n\ndef parse_cookies(header: str) -> dict[str, str]:\n    sc: SimpleCookie = SimpleCookie()\n    sc.load(header)\n    return {k: v.value for k, v in sc.items()}\n```\nUse `http.cookies` rather than splitting strings -- it handles quoted values correctly.")

    add("web",
        "Implement a FastAPI endpoint that returns an image from disk.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import FileResponse\n\napp = FastAPI()\n\n@app.get('/logo.png')\ndef logo() -> FileResponse:\n    return FileResponse('static/logo.png', media_type='image/png')\n```\n`FileResponse` streams the file efficiently and sets ETag/Last-Modified headers.")

    add("web",
        "Write a Flask route that returns 304 if If-None-Match matches.",
        "```python\nfrom flask import Flask, Response, request\n\napp = Flask(__name__)\nETAG = 'W/\"v1\"'\n\n@app.get('/data')\ndef data():\n    if request.headers.get('If-None-Match') == ETAG:\n        return Response(status=304)\n    resp = Response('hello')\n    resp.headers['ETag'] = ETAG\n    return resp\n```\nWeak validators (`W/`) are sufficient for content-equivalence; strong validators require byte equivalence.")

    add("web",
        "Implement a function that talks to Redis with a connection pool.",
        "```python\nimport redis\n\npool = redis.ConnectionPool.from_url('redis://localhost:6379/0', max_connections=20)\n\ndef cached(key: str) -> str | None:\n    r = redis.Redis(connection_pool=pool)\n    val = r.get(key)\n    return val.decode() if val else None\n```\nReuse the pool across the process; never create a new client per call.")

    add("web",
        "Write a FastAPI endpoint with response_model that excludes None fields.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\nclass User(BaseModel):\n    id: int\n    name: str\n    nickname: str | None = None\n\napp = FastAPI()\n\n@app.get('/users/me', response_model=User, response_model_exclude_none=True)\ndef me() -> User:\n    return User(id=1, name='Ada')\n```\n`exclude_none` keeps the response shape lean for optional fields.")

    add("web",
        "Implement a Flask route that returns a paginated query result.",
        "```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\nDATA = list(range(1000))\n\n@app.get('/items')\ndef items():\n    page = int(request.args.get('page', 1))\n    size = min(int(request.args.get('size', 20)), 100)\n    start = (page - 1) * size\n    return jsonify(items=DATA[start:start + size], page=page, size=size, total=len(DATA))\n```\nAlways cap page size; clients can otherwise crash you with `size=1e9`.")

    add("web",
        "Write a function that parses an `Accept-Language` header.",
        "```python\ndef parse_accept_language(header: str) -> list[tuple[str, float]]:\n    parts = []\n    for item in header.split(','):\n        if not item.strip(): continue\n        if ';' in item:\n            lang, q = item.split(';', 1)\n            q_val = float(q.split('=', 1)[1])\n        else:\n            lang, q_val = item, 1.0\n        parts.append((lang.strip(), q_val))\n    return sorted(parts, key=lambda x: -x[1])\n```\nSort by quality descending so the highest-priority language is first.")

    add("web",
        "Implement a FastAPI endpoint that takes a list of IDs as a query parameter.",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\n\n@app.get('/items')\ndef items(ids: list[int] = Query(...)) -> dict:\n    return {'ids': ids}\n```\nFastAPI parses repeated `?ids=1&ids=2` into a list automatically.")

    add("web",
        "Write a function that creates an httpx mock-compatible test client.",
        "```python\nimport httpx\n\ndef make_test_client(transport_handler) -> httpx.Client:\n    transport = httpx.MockTransport(transport_handler)\n    return httpx.Client(transport=transport, base_url='https://test')\n```\n`MockTransport` is the supported way to test code that uses httpx without hitting the network.")

    add("web",
        "Implement a Flask route protected by an API key.",
        "```python\nimport hmac\nimport os\nfrom functools import wraps\nfrom flask import Flask, abort, request\n\nAPI_KEY = os.environ['API_KEY']\napp = Flask(__name__)\n\ndef require_key(fn):\n    @wraps(fn)\n    def inner(*a, **kw):\n        key = request.headers.get('X-API-Key', '')\n        if not hmac.compare_digest(key, API_KEY):\n            abort(401)\n        return fn(*a, **kw)\n    return inner\n\n@app.get('/private')\n@require_key\ndef private():\n    return {'ok': True}\n```\n`hmac.compare_digest` prevents timing attacks on the key comparison.")

    add("web",
        "Write a FastAPI endpoint that returns 204 No Content on delete.",
        "```python\nfrom fastapi import FastAPI, Response, status\n\napp = FastAPI()\n\n@app.delete('/items/{iid}', status_code=status.HTTP_204_NO_CONTENT)\ndef delete_item(iid: int) -> Response:\n    return Response(status_code=204)\n```\n204 is the conventional response for successful DELETE with no body.")

    add("web",
        "Implement a Flask wsgi entry point with gunicorn-friendly factory.",
        "```python\n# wsgi.py\nfrom myapp import create_app\n\napp = create_app()\n```\n`gunicorn wsgi:app` runs it. Avoid `if __name__ == '__main__': app.run()` in production -- the dev server is single-threaded.")

    add("web",
        "Write a function that reads a streaming SSE response from a server.",
        "```python\nimport httpx\n\ndef sse_lines(url: str):\n    with httpx.stream('GET', url, timeout=None, headers={'Accept': 'text/event-stream'}) as r:\n        r.raise_for_status()\n        for line in r.iter_lines():\n            if line.startswith('data: '):\n                yield line[len('data: '):]\n```\n`timeout=None` for SSE -- the connection is intentionally long-lived.")

    add("web",
        "Implement a FastAPI dependency that returns a per-request DB session.",
        "```python\nfrom fastapi import Depends, FastAPI\nfrom sqlalchemy.orm import Session\n\napp = FastAPI()\nSessionLocal = ...  # configured elsewhere\n\ndef get_db() -> Session:\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n\n@app.get('/users')\ndef list_users(db: Session = Depends(get_db)):\n    return db.query(User).all()\n```\nThe yield-based dependency cleans up the session after the response.")

    add("web",
        "Write a function that posts a Slack message via webhook.",
        "```python\nimport httpx\n\ndef notify_slack(webhook_url: str, text: str) -> None:\n    r = httpx.post(webhook_url, json={'text': text}, timeout=5.0)\n    r.raise_for_status()\n```\nKeep the timeout short; Slack's API is reliable but you don't want to block on it.")

    add("web",
        "Implement a function that uploads to S3 with boto3 and a presigned URL.",
        "```python\nimport boto3\n\ndef presigned_put(bucket: str, key: str, expires: int = 3600) -> str:\n    s3 = boto3.client('s3')\n    return s3.generate_presigned_url(\n        'put_object',\n        Params={'Bucket': bucket, 'Key': key},\n        ExpiresIn=expires,\n    )\n```\nClients PUT directly to S3 -- skip the proxying through your server.")

    add("web",
        "Write a FastAPI endpoint that returns a HEAD response with custom headers.",
        "```python\nfrom fastapi import FastAPI, Response\n\napp = FastAPI()\n\n@app.head('/data')\ndef head_data():\n    return Response(headers={'X-Total-Count': '42', 'Content-Type': 'application/json'})\n```\nHEAD must return the same headers as GET but no body.")

    add("web",
        "Implement a function that uses `aiohttp` to fetch concurrently.",
        "```python\nimport aiohttp\nimport asyncio\n\nasync def fetch_all(urls: list[str]) -> list[str]:\n    timeout = aiohttp.ClientTimeout(total=10.0)\n    async with aiohttp.ClientSession(timeout=timeout) as session:\n        async def one(url):\n            async with session.get(url) as r:\n                return await r.text()\n        return await asyncio.gather(*[one(u) for u in urls])\n```\nReuse the session; never create one per request.")

    add("web",
        "Write a FastAPI endpoint that supports both JSON and form bodies.",
        "```python\nfrom fastapi import FastAPI, Form\nfrom pydantic import BaseModel\n\nclass Body(BaseModel):\n    name: str\n\napp = FastAPI()\n\n@app.post('/json')\ndef j(b: Body) -> dict:\n    return b.model_dump()\n\n@app.post('/form')\ndef f(name: str = Form(...)) -> dict:\n    return {'name': name}\n```\nMixing JSON and form-encoded on the same path is a smell; route them separately.")

    add("web",
        "Implement a connection-pooled SQLAlchemy engine.",
        "```python\nfrom sqlalchemy import create_engine\n\nengine = create_engine(\n    'postgresql+psycopg://user:pw@host/db',\n    pool_size=10,\n    max_overflow=10,\n    pool_pre_ping=True,\n    pool_recycle=1800,\n)\n```\n`pool_pre_ping` survives idle-connection drops; `pool_recycle` keeps long-lived idle connections fresh.")

    add("web",
        "Write a Flask app that uses application factory pattern.",
        "```python\nfrom flask import Flask\n\ndef create_app(config: dict | None = None) -> Flask:\n    app = Flask(__name__)\n    app.config.update(config or {})\n    from .routes import bp\n    app.register_blueprint(bp)\n    return app\n```\nFactories make testing far easier -- you can spin up a fresh app per test.")

    add("web",
        "Implement a FastAPI endpoint that takes optional headers.",
        "```python\nfrom fastapi import FastAPI, Header\n\napp = FastAPI()\n\n@app.get('/info')\ndef info(x_request_id: str | None = Header(None), x_user_agent: str | None = Header(None)) -> dict:\n    return {'request_id': x_request_id, 'user_agent': x_user_agent}\n```\nFastAPI normalizes hyphens to underscores in parameter names.")

    add("web",
        "Write a function that proxies a request and forwards the response.",
        "```python\nimport httpx\nfrom fastapi import FastAPI, Request\nfrom fastapi.responses import Response\n\napp = FastAPI()\n\n@app.api_route('/proxy/{path:path}', methods=['GET', 'POST'])\nasync def proxy(path: str, request: Request) -> Response:\n    url = f'https://upstream.example/{path}'\n    body = await request.body()\n    async with httpx.AsyncClient(timeout=30.0) as client:\n        r = await client.request(request.method, url, content=body, headers=dict(request.headers))\n    return Response(content=r.content, status_code=r.status_code, headers=dict(r.headers))\n```\nStrip hop-by-hop headers (`Connection`, `Transfer-Encoding`, etc.) in production.")

    add("web",
        "Implement a function that downloads with content-length-based progress.",
        "```python\nimport httpx\nfrom pathlib import Path\n\ndef download_with_progress(url: str, dest: Path) -> None:\n    with httpx.stream('GET', url, timeout=60.0) as r:\n        r.raise_for_status()\n        total = int(r.headers.get('content-length', 0))\n        downloaded = 0\n        with dest.open('wb') as f:\n            for chunk in r.iter_bytes():\n                f.write(chunk); downloaded += len(chunk)\n                if total:\n                    print(f'\\r{downloaded/total*100:.1f}%', end='')\n        print()\n```\nFalls back to silence when the server doesn't send content-length.")

    add("web",
        "Write a FastAPI endpoint that returns NDJSON.",
        "```python\nimport json\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\n@app.get('/stream')\ndef stream() -> StreamingResponse:\n    def gen():\n        for i in range(1000):\n            yield json.dumps({'i': i}) + '\\n'\n    return StreamingResponse(gen(), media_type='application/x-ndjson')\n```\nNDJSON is streamable and trivial for clients to parse.")

    add("web",
        "Implement an async retry helper for httpx calls.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_with_retry(client: httpx.AsyncClient, url: str, attempts: int = 3) -> httpx.Response:\n    for i in range(attempts):\n        try:\n            r = await client.get(url, timeout=5.0)\n            r.raise_for_status()\n            return r\n        except (httpx.TransportError, httpx.HTTPStatusError):\n            if i == attempts - 1:\n                raise\n            await asyncio.sleep(2 ** i)\n    raise RuntimeError('unreachable')\n```\nExponential backoff with full jitter would be better in high-concurrency settings.")

    add("web",
        "Write a Flask route that uses `g` to memoize per-request data.",
        "```python\nfrom flask import Flask, g, jsonify\n\napp = Flask(__name__)\n\ndef current_user():\n    if 'user' not in g:\n        g.user = load_user_from_token()  # implementation-specific\n    return g.user\n\n@app.get('/me')\ndef me():\n    return jsonify(current_user())\n```\n`g` is per-request, automatically cleaned up after the response.")

    add("web",
        "Implement a function that rotates and signs an HMAC token.",
        "```python\nimport hashlib, hmac, secrets, time\n\ndef sign(secret: bytes, payload: str, ttl: int = 3600) -> str:\n    nonce = secrets.token_hex(8)\n    exp = int(time.time()) + ttl\n    body = f'{payload}|{nonce}|{exp}'\n    sig = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()\n    return f'{body}|{sig}'\n\ndef verify(secret: bytes, token: str) -> str | None:\n    body, _, sig = token.rpartition('|')\n    expected = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()\n    if not hmac.compare_digest(sig, expected): return None\n    payload, nonce, exp = body.split('|')\n    if int(exp) < time.time(): return None\n    return payload\n```\nFor production prefer a vetted library (PyJWT, itsdangerous).")

    add("web",
        "Write a FastAPI endpoint that returns an HTML response.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import HTMLResponse\n\napp = FastAPI()\n\n@app.get('/', response_class=HTMLResponse)\ndef index() -> str:\n    return '<h1>hello</h1>'\n```\nFor templating, swap to Jinja2 with `templates.TemplateResponse(...)`.")

    add("web",
        "Implement a FastAPI route with response caching headers.",
        "```python\nfrom fastapi import FastAPI, Response\n\napp = FastAPI()\n\n@app.get('/static-data')\ndef static_data(response: Response) -> dict:\n    response.headers['Cache-Control'] = 'public, max-age=3600, stale-while-revalidate=600'\n    return {'value': 42}\n```\n`stale-while-revalidate` lets caches serve stale content briefly while refetching.")

    add("web",
        "Write a function that parses a multipart upload body manually.",
        "```python\nfrom email.parser import BytesParser\nfrom email.policy import default\n\ndef parse_multipart(body: bytes, boundary: str) -> list:\n    body = b'Content-Type: multipart/form-data; boundary=' + boundary.encode() + b'\\r\\n\\r\\n' + body\n    msg = BytesParser(policy=default).parsebytes(body)\n    return [(p.get_filename(), p.get_payload(decode=True)) for p in msg.iter_parts()]\n```\nReuse the stdlib email parser; it's robust to weird boundaries.")

    add("web",
        "Implement a Flask app that serves OpenAPI spec at /openapi.json.",
        "```python\nfrom flask import Flask, jsonify\n\nSPEC = {'openapi': '3.0.0', 'info': {'title': 'app', 'version': '1.0'}, 'paths': {}}\napp = Flask(__name__)\n\n@app.get('/openapi.json')\ndef spec():\n    return jsonify(SPEC)\n```\nFastAPI does this automatically; for Flask use `flask-smorest` or `apispec` to keep it in sync with code.")

    add("web",
        "Write a function that fetches and parses a feed (Atom/RSS).",
        "```python\nimport feedparser\n\ndef latest_titles(url: str, n: int = 5) -> list[str]:\n    feed = feedparser.parse(url)\n    return [e.title for e in feed.entries[:n]]\n```\n`feedparser` handles both Atom and RSS plus the long tail of malformed feeds.")

    add("web",
        "Implement a FastAPI dependency that limits payload size.",
        "```python\nfrom fastapi import Depends, FastAPI, HTTPException, Request, status\n\nMAX = 1 << 20  # 1 MiB\n\nasync def limit_body(request: Request) -> None:\n    cl = request.headers.get('content-length')\n    if cl and int(cl) > MAX:\n        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f'body > {MAX} bytes')\n\napp = FastAPI()\n\n@app.post('/upload', dependencies=[Depends(limit_body)])\nasync def upload(request: Request):\n    return {'len': len(await request.body())}\n```\nDefence in depth -- also enforce limits at the proxy layer (nginx, ALB).")

    add("web",
        "Write a Flask app that handles 404 with a JSON body.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.errorhandler(404)\ndef not_found(_):\n    return jsonify(error='not found'), 404\n```\nGives API consumers a parseable error instead of HTML.")

    add("web",
        "Implement a function that safely parses JSON with a max size.",
        "```python\nimport json\n\ndef safe_json(body: bytes, max_bytes: int = 1 << 20) -> dict:\n    if len(body) > max_bytes:\n        raise ValueError('body too large')\n    return json.loads(body)\n```\nPrevents pathological JSON from eating memory; nothing fancier than the size check is needed for typical inputs.")

    add("web",
        "Write a function that calls a GraphQL endpoint with httpx.",
        "```python\nimport httpx\n\ndef gql(url: str, query: str, variables: dict | None = None) -> dict:\n    r = httpx.post(url, json={'query': query, 'variables': variables or {}}, timeout=10.0)\n    r.raise_for_status()\n    payload = r.json()\n    if 'errors' in payload:\n        raise RuntimeError(payload['errors'])\n    return payload['data']\n```\nGraphQL returns 200 even for errors -- check `errors` in the body.")

    add("web",
        "Implement a Flask 'before_first_request' replacement using `@app.before_serving`.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\n\nwith app.app_context():\n    init_caches()  # one-time setup\n\n@app.get('/')\ndef root():\n    return 'ok'\n```\n`before_first_request` is deprecated; do startup work at import time inside an app context.")

    add("web",
        "Write a FastAPI endpoint that serves an HLS playlist.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import Response\n\nMANIFEST = '''#EXTM3U\\n#EXT-X-VERSION:3\\n#EXT-X-TARGETDURATION:10\\n#EXT-X-MEDIA-SEQUENCE:0\\n#EXTINF:10.0,\\nseg0.ts\\n#EXT-X-ENDLIST\\n'''\n\napp = FastAPI()\n\n@app.get('/index.m3u8')\ndef playlist() -> Response:\n    return Response(content=MANIFEST, media_type='application/vnd.apple.mpegurl')\n```\nThe specific media type matters; players sniff it.")

    add("web",
        "Implement a function that uploads to GCS using google-cloud-storage.",
        "```python\nfrom google.cloud import storage\nfrom pathlib import Path\n\ndef upload_to_gcs(bucket_name: str, source: Path, dest: str) -> None:\n    client = storage.Client()\n    bucket = client.bucket(bucket_name)\n    blob = bucket.blob(dest)\n    blob.upload_from_filename(str(source))\n```\nReuse the `Client` across uploads; it owns the underlying HTTP session.")

    add("web",
        "Write a FastAPI middleware that adds CORS preflight support manually.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://app.example.com'],\n    allow_credentials=True,\n    allow_methods=['GET', 'POST'],\n    allow_headers=['Authorization', 'Content-Type'],\n)\n```\nDon't roll your own CORS -- the built-in middleware handles preflight correctly.")

    add("web",
        "Implement a function that sends a Discord webhook.",
        "```python\nimport httpx\n\ndef notify_discord(url: str, content: str) -> None:\n    r = httpx.post(url, json={'content': content[:2000]}, timeout=5.0)\n    r.raise_for_status()\n```\nDiscord caps message length at 2000 characters -- truncate or split.")

    add("web",
        "Write a function that GETs with conditional `If-Modified-Since`.",
        "```python\nimport httpx\nfrom email.utils import formatdate\n\ndef get_if_changed(url: str, last_seen: float) -> httpx.Response | None:\n    headers = {'If-Modified-Since': formatdate(last_seen, usegmt=True)}\n    r = httpx.get(url, headers=headers, timeout=10.0)\n    if r.status_code == 304:\n        return None\n    r.raise_for_status()\n    return r\n```\n`email.utils.formatdate` produces RFC 7231 dates correctly.")

    add("web",
        "Implement a Flask app with health, metrics, and ready endpoints.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/healthz')\ndef healthz():\n    return jsonify(status='ok')\n\n@app.get('/readyz')\ndef readyz():\n    return jsonify(ready=True)\n\n@app.get('/metrics')\ndef metrics():\n    return jsonify(requests_total=0)\n```\nKeep these dirt simple so they're robust during incidents.")

    add("web",
        "Write a function that calls Stripe API with idempotency key.",
        "```python\nimport secrets\nimport httpx\n\ndef create_charge(api_key: str, amount: int, currency: str) -> dict:\n    headers = {\n        'Authorization': f'Bearer {api_key}',\n        'Idempotency-Key': secrets.token_hex(16),\n    }\n    r = httpx.post(\n        'https://api.stripe.com/v1/charges',\n        data={'amount': amount, 'currency': currency},\n        headers=headers,\n        timeout=30.0,\n    )\n    r.raise_for_status()\n    return r.json()\n```\nIdempotency keys make retries safe -- Stripe returns the original response on duplicates.")

    add("web",
        "Implement a FastAPI endpoint that sets a cookie on login.",
        "```python\nfrom fastapi import FastAPI, Response\n\napp = FastAPI()\n\n@app.post('/login')\ndef login(response: Response) -> dict:\n    response.set_cookie('session', 'abc', httponly=True, samesite='lax', secure=True, max_age=86400)\n    return {'ok': True}\n```\nSecure flag matters in production; lax SameSite covers most CSRF cases for typical apps.")

    add("web",
        "Write a function that publishes a message to Kafka.",
        "```python\nimport json\nfrom confluent_kafka import Producer\n\nproducer = Producer({'bootstrap.servers': 'localhost:9092'})\n\ndef publish(topic: str, key: str, payload: dict) -> None:\n    producer.produce(topic, key=key.encode(), value=json.dumps(payload).encode())\n    producer.poll(0)\n```\nCall `producer.flush()` before process exit so buffered messages are sent.")

    add("web",
        "Implement a Flask route that serves a generated PDF.",
        "```python\nimport io\nfrom flask import Flask, Response\nfrom reportlab.pdfgen import canvas\n\napp = Flask(__name__)\n\n@app.get('/report.pdf')\ndef report():\n    buf = io.BytesIO()\n    c = canvas.Canvas(buf)\n    c.drawString(100, 750, 'Hello, world')\n    c.save()\n    return Response(buf.getvalue(), mimetype='application/pdf')\n```\nReportLab is the most established Python PDF library.")

    add("web",
        "Write a FastAPI endpoint that returns timezone-aware timestamps.",
        "```python\nimport datetime as dt\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/now')\ndef now() -> dict:\n    return {'now': dt.datetime.now(dt.timezone.utc).isoformat()}\n```\nAlways serialize timezone-aware UTC; clients localize as needed.")

    add("web",
        "Implement a function that sends an email through SMTP with TLS.",
        "```python\nimport smtplib\nfrom email.message import EmailMessage\n\ndef send_email(host: str, user: str, pw: str, to: str, subject: str, body: str) -> None:\n    msg = EmailMessage()\n    msg['From'] = user\n    msg['To'] = to\n    msg['Subject'] = subject\n    msg.set_content(body)\n    with smtplib.SMTP(host, 587) as s:\n        s.starttls()\n        s.login(user, pw)\n        s.send_message(msg)\n```\nUse `EmailMessage` (the modern API), not the legacy `email.mime` mess.")

    add("web",
        "Write a Flask route that returns 429 with Retry-After.",
        "```python\nfrom flask import Flask, Response, request\n\napp = Flask(__name__)\nHITS: dict[str, int] = {}\n\n@app.get('/limited')\ndef limited():\n    ip = request.remote_addr or 'unknown'\n    HITS[ip] = HITS.get(ip, 0) + 1\n    if HITS[ip] > 10:\n        return Response('too many', 429, {'Retry-After': '60'})\n    return {'ok': True}\n```\nIn production back this with Redis; the in-memory dict resets per process.")

    add("web",
        "Implement an async function that fans out to many APIs with bounded concurrency.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_bounded(urls: list[str], limit: int = 10) -> list[str]:\n    sem = asyncio.Semaphore(limit)\n    async with httpx.AsyncClient(timeout=10.0) as client:\n        async def one(url):\n            async with sem:\n                r = await client.get(url)\n                r.raise_for_status()\n                return r.text\n        return await asyncio.gather(*[one(u) for u in urls])\n```\nBounded concurrency prevents you from being your own DDoS.")

    add("web",
        "Write a FastAPI endpoint that returns differentiated responses by Accept header.",
        "```python\nfrom fastapi import FastAPI, Header\nfrom fastapi.responses import HTMLResponse, JSONResponse\n\napp = FastAPI()\n\n@app.get('/data')\ndef data(accept: str | None = Header(None)):\n    if accept and 'text/html' in accept:\n        return HTMLResponse('<p>hello</p>')\n    return JSONResponse({'message': 'hello'})\n```\nContent negotiation keeps the same URL useful from a browser and from code.")

    add("web",
        "Implement a function that retries network errors with `tenacity`.",
        "```python\nimport httpx\nfrom tenacity import retry, retry_if_exception_type, stop_after_delay, wait_exponential\n\n@retry(\n    retry=retry_if_exception_type(httpx.TransportError),\n    stop=stop_after_delay(30),\n    wait=wait_exponential(min=0.5, max=8),\n    reraise=True,\n)\ndef fetch(url: str) -> str:\n    r = httpx.get(url, timeout=5.0)\n    r.raise_for_status()\n    return r.text\n```\n`stop_after_delay` is wall-clock-based -- often more useful than attempt count.")

    add("web",
        "Write a Flask route that proxies to another service preserving headers.",
        "```python\nimport httpx\nfrom flask import Flask, Response, request\n\napp = Flask(__name__)\n\n@app.route('/proxy/<path:rest>', methods=['GET', 'POST', 'PUT', 'DELETE'])\ndef proxy(rest: str):\n    upstream = httpx.request(\n        request.method,\n        f'https://upstream.example/{rest}',\n        headers={k: v for k, v in request.headers if k.lower() != 'host'},\n        params=request.args, content=request.get_data(), timeout=30.0,\n    )\n    return Response(upstream.content, status=upstream.status_code, headers=dict(upstream.headers))\n```\nDrop `Host` so the upstream sets its own.")

    add("web",
        "Implement a FastAPI endpoint that requires an admin role from JWT claims.",
        "```python\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, status\nfrom fastapi.security import HTTPAuthorizationCredentials, HTTPBearer\n\nbearer = HTTPBearer()\napp = FastAPI()\n\ndef require_admin(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> dict:\n    try:\n        claims = jwt.decode(creds.credentials, 'secret', algorithms=['HS256'])\n    except jwt.PyJWTError as e:\n        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(e))\n    if 'admin' not in claims.get('roles', []):\n        raise HTTPException(status.HTTP_403_FORBIDDEN, 'admin only')\n    return claims\n\n@app.get('/admin')\ndef admin(_: dict = Depends(require_admin)) -> dict:\n    return {'ok': True}\n```\n401 vs 403 matters: bad token vs valid-but-unauthorized.")

    add("web",
        "Write a function that creates a signed S3 presigned GET URL.",
        "```python\nimport boto3\n\ndef presigned_get(bucket: str, key: str, expires: int = 600) -> str:\n    s3 = boto3.client('s3')\n    return s3.generate_presigned_url(\n        'get_object',\n        Params={'Bucket': bucket, 'Key': key},\n        ExpiresIn=expires,\n    )\n```\nUse short expirations -- the URL is a bearer credential.")

    add("web",
        "Implement a Flask route that redirects with status 307 (preserves method).",
        "```python\nfrom flask import Flask, redirect\n\napp = Flask(__name__)\n\n@app.post('/old')\ndef old_post():\n    return redirect('/new', code=307)\n```\n307 preserves method and body; 302 historically downgrades POST to GET.")

    add("web",
        "Write a function that signs a URL with HMAC for integrity.",
        "```python\nimport hashlib, hmac\nfrom urllib.parse import urlencode\n\ndef sign_url(secret: bytes, base: str, params: dict) -> str:\n    body = urlencode(sorted(params.items()))\n    sig = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()\n    return f'{base}?{body}&sig={sig}'\n```\nSort params before signing; otherwise dict ordering breaks verification.")

    add("web",
        "Implement a function that batches API calls with `asyncio.Queue`.",
        "```python\nimport asyncio\nimport httpx\n\nasync def worker(queue: asyncio.Queue, client: httpx.AsyncClient, results: list) -> None:\n    while True:\n        url = await queue.get()\n        try:\n            r = await client.get(url)\n            results.append((url, r.status_code))\n        finally:\n            queue.task_done()\n\nasync def batch(urls: list[str], n_workers: int = 5) -> list:\n    queue: asyncio.Queue = asyncio.Queue()\n    for u in urls: queue.put_nowait(u)\n    results: list = []\n    async with httpx.AsyncClient(timeout=10.0) as client:\n        workers = [asyncio.create_task(worker(queue, client, results)) for _ in range(n_workers)]\n        await queue.join()\n        for w in workers: w.cancel()\n    return results\n```\nQueue + workers caps concurrency cleanly.")

    add("web",
        "Write a FastAPI endpoint with a path enum.",
        "```python\nfrom enum import Enum\nfrom fastapi import FastAPI\n\nclass Color(str, Enum):\n    red = 'red'\n    green = 'green'\n    blue = 'blue'\n\napp = FastAPI()\n\n@app.get('/colors/{c}')\ndef show(c: Color) -> dict:\n    return {'name': c.value}\n```\nString-valued enums show up nicely in OpenAPI docs.")

    add("web",
        "Implement a function that downloads a remote URL only if newer.",
        "```python\nfrom email.utils import formatdate\nfrom pathlib import Path\nimport httpx\n\ndef sync(url: str, dest: Path) -> bool:\n    headers = {}\n    if dest.exists():\n        headers['If-Modified-Since'] = formatdate(dest.stat().st_mtime, usegmt=True)\n    r = httpx.get(url, headers=headers, timeout=30.0)\n    if r.status_code == 304:\n        return False\n    r.raise_for_status()\n    dest.write_bytes(r.content)\n    return True\n```\nReturns whether new content was downloaded.")

    add("web",
        "Write a Flask app that uses url_for to build links safely.",
        "```python\nfrom flask import Flask, jsonify, url_for\n\napp = Flask(__name__)\n\n@app.get('/users/<int:uid>')\ndef user(uid: int):\n    return jsonify(\n        id=uid,\n        self=url_for('user', uid=uid, _external=True),\n        next=url_for('user', uid=uid + 1, _external=True),\n    )\n```\n`url_for` survives blueprint reorganizations and respects `SCRIPT_NAME` for sub-path mounts.")

    add("web",
        "Implement a FastAPI dependency that extracts and validates a tenant ID.",
        "```python\nfrom fastapi import Depends, FastAPI, Header, HTTPException, status\n\ndef tenant(x_tenant: str = Header(...)) -> str:\n    if not x_tenant.isalnum():\n        raise HTTPException(status.HTTP_400_BAD_REQUEST, 'bad tenant id')\n    return x_tenant\n\napp = FastAPI()\n\n@app.get('/data')\ndef data(t: str = Depends(tenant)) -> dict:\n    return {'tenant': t}\n```\nValidate at the dependency layer so handlers don't repeat the same checks.")

    add("web",
        "Write a function that streams a large query result as CSV with FastAPI.",
        "```python\nimport csv\nimport io\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\ndef rows():\n    for i in range(1_000_000):\n        yield (i, f'name-{i}')\n\n@app.get('/export.csv')\ndef export() -> StreamingResponse:\n    def gen():\n        buf = io.StringIO()\n        w = csv.writer(buf)\n        w.writerow(['id', 'name']); yield buf.getvalue(); buf.seek(0); buf.truncate()\n        for r in rows():\n            w.writerow(r); yield buf.getvalue(); buf.seek(0); buf.truncate()\n    return StreamingResponse(gen(), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=export.csv'})\n```\nReuse a single buffer to avoid per-row allocation.")

    add("web",
        "Implement a function that tests if a server supports range requests.",
        "```python\nimport httpx\n\ndef supports_range(url: str) -> bool:\n    r = httpx.head(url, timeout=10.0)\n    r.raise_for_status()\n    return r.headers.get('accept-ranges', '').lower() == 'bytes'\n```\nUseful before attempting resumable downloads.")

    # ---- data science (28) ----
    add("data",
        "Read a CSV with pandas and show summary statistics.",
        "```python\nimport pandas as pd\n\ndef summarize(path: str) -> pd.DataFrame:\n    df = pd.read_csv(path)\n    return df.describe(include='all')\n```\n`include='all'` covers numeric and object columns.")

    add("data",
        "Write a function that loads a parquet file and selects columns.",
        "```python\nimport pandas as pd\n\ndef load_parquet(path: str, cols: list[str]) -> pd.DataFrame:\n    return pd.read_parquet(path, columns=cols)\n```\nColumn projection is pushed down to the parquet reader -- much faster than loading everything and slicing.")

    add("data",
        "Implement a function that fills missing numeric values with the median.",
        "```python\nimport pandas as pd\n\ndef fill_median(df: pd.DataFrame) -> pd.DataFrame:\n    df = df.copy()\n    nums = df.select_dtypes('number').columns\n    df[nums] = df[nums].fillna(df[nums].median())\n    return df\n```\nMedian is robust to outliers; mean is the wrong choice when you have heavy tails.")

    add("data",
        "Write a function that one-hot encodes categorical columns.",
        "```python\nimport pandas as pd\n\ndef encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:\n    return pd.get_dummies(df, columns=cols, drop_first=True)\n```\n`drop_first=True` avoids the dummy-variable trap when these features feed a linear model.")

    add("data",
        "Implement a function that computes a rolling mean.",
        "```python\nimport pandas as pd\n\ndef rolling_mean(s: pd.Series, window: int) -> pd.Series:\n    return s.rolling(window=window, min_periods=1).mean()\n```\n`min_periods=1` produces values from the first observation; default behavior leaves leading NaNs.")

    add("data",
        "Write a function that groups by a column and aggregates two metrics.",
        "```python\nimport pandas as pd\n\ndef agg(df: pd.DataFrame, group: str) -> pd.DataFrame:\n    return df.groupby(group, as_index=False).agg(total=('amount', 'sum'), n=('amount', 'count'))\n```\nNamed-aggregation syntax is more readable than the legacy dict form.")

    add("data",
        "Implement a function that pivots from long to wide format.",
        "```python\nimport pandas as pd\n\ndef to_wide(df: pd.DataFrame, idx: str, col: str, val: str) -> pd.DataFrame:\n    return df.pivot_table(index=idx, columns=col, values=val, aggfunc='sum').reset_index()\n```\n`pivot_table` aggregates duplicates; `pivot` errors on them.")

    add("data",
        "Write a function that loads JSON lines into a DataFrame.",
        "```python\nimport pandas as pd\n\ndef load_jsonl(path: str) -> pd.DataFrame:\n    return pd.read_json(path, lines=True)\n```\nFor very large files iterate with `chunksize=` to avoid loading it all at once.")

    add("data",
        "Implement a function that returns z-scores for numeric columns.",
        "```python\nimport pandas as pd\n\ndef zscore(df: pd.DataFrame) -> pd.DataFrame:\n    nums = df.select_dtypes('number')\n    return (nums - nums.mean()) / nums.std(ddof=0)\n```\nUse `ddof=0` for population std, `ddof=1` for sample.")

    add("data",
        "Write a function that filters rows where a column is in a set.",
        "```python\nimport pandas as pd\n\ndef filter_in(df: pd.DataFrame, col: str, values: set) -> pd.DataFrame:\n    return df[df[col].isin(values)]\n```\n`isin` is vectorized; chaining `==` with `|` for many values is much slower.")

    add("data",
        "Implement a function that computes pairwise correlation.",
        "```python\nimport pandas as pd\n\ndef corr(df: pd.DataFrame) -> pd.DataFrame:\n    return df.select_dtypes('number').corr()\n```\nRestrict to numeric columns; `corr` would otherwise raise on strings.")

    add("data",
        "Write a function that resamples a time series to daily frequency.",
        "```python\nimport pandas as pd\n\ndef daily(s: pd.Series) -> pd.Series:\n    return s.resample('D').mean()\n```\nThe Series index must be a DatetimeIndex; convert with `pd.to_datetime` if needed.")

    add("data",
        "Implement a function that applies a custom row-wise function efficiently.",
        "```python\nimport pandas as pd\n\ndef compute(df: pd.DataFrame) -> pd.Series:\n    # vectorized -- avoid df.apply where possible\n    return df['a'] * 2 + df['b']\n```\n`apply(axis=1)` is a Python loop in disguise -- vectorize first if at all possible.")

    add("data",
        "Write a function that creates a numpy array from nested lists.",
        "```python\nimport numpy as np\n\ndef to_array(rows: list[list[float]]) -> np.ndarray:\n    return np.asarray(rows, dtype=np.float64)\n```\n`asarray` avoids a copy if the input is already an ndarray.")

    add("data",
        "Implement a function that masks values outside [low, high].",
        "```python\nimport numpy as np\n\ndef clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:\n    return np.clip(arr, low, high)\n```\n`np.clip` is C-fast and handles broadcasting if `low`/`high` are arrays.")

    add("data",
        "Write a function that returns the indices of the n largest values.",
        "```python\nimport numpy as np\n\ndef top_n_indices(arr: np.ndarray, n: int) -> np.ndarray:\n    idx = np.argpartition(-arr, n - 1)[:n]\n    return idx[np.argsort(-arr[idx])]\n```\n`argpartition` finds the top n in O(n); the second sort orders just those n.")

    add("data",
        "Implement a function that normalizes columns of a 2D array.",
        "```python\nimport numpy as np\n\ndef l2_normalize(arr: np.ndarray) -> np.ndarray:\n    norms = np.linalg.norm(arr, axis=0, keepdims=True)\n    norms = np.where(norms == 0, 1, norms)\n    return arr / norms\n```\nGuard against zero columns by replacing the divisor with 1.")

    add("data",
        "Write a function that plots a line chart with matplotlib.",
        "```python\nimport matplotlib.pyplot as plt\n\ndef line_chart(x, y, title: str, out: str) -> None:\n    fig, ax = plt.subplots(figsize=(8, 4))\n    ax.plot(x, y)\n    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')\n    fig.tight_layout()\n    fig.savefig(out, dpi=150)\n    plt.close(fig)\n```\nClose the figure to avoid leaking memory in long-running scripts.")

    add("data",
        "Implement a function that plots a histogram.",
        "```python\nimport matplotlib.pyplot as plt\n\ndef histogram(values, bins: int, out: str) -> None:\n    fig, ax = plt.subplots()\n    ax.hist(values, bins=bins)\n    ax.set_xlabel('value'); ax.set_ylabel('count')\n    fig.savefig(out, dpi=150)\n    plt.close(fig)\n```\nPick `bins='auto'` if you don't have a strong prior on the count.")

    add("data",
        "Write a function that bins continuous values into quartiles.",
        "```python\nimport pandas as pd\n\ndef quartile(s: pd.Series) -> pd.Series:\n    return pd.qcut(s, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])\n```\n`qcut` produces equal-frequency bins; `cut` produces equal-width bins.")

    add("data",
        "Implement a function that returns top-n by group.",
        "```python\nimport pandas as pd\n\ndef top_n(df: pd.DataFrame, group: str, col: str, n: int) -> pd.DataFrame:\n    return df.sort_values(col, ascending=False).groupby(group).head(n)\n```\nSorting once and using `head(n)` is cleaner than a custom apply.")

    add("data",
        "Write a function that joins two DataFrames on a column.",
        "```python\nimport pandas as pd\n\ndef join(a: pd.DataFrame, b: pd.DataFrame, on: str, how: str = 'inner') -> pd.DataFrame:\n    return a.merge(b, on=on, how=how, validate='one_to_one')\n```\n`validate=` catches unexpected duplicates that silently change row counts.")

    add("data",
        "Implement a function that converts a date column to datetime safely.",
        "```python\nimport pandas as pd\n\ndef to_datetime(s: pd.Series) -> pd.Series:\n    return pd.to_datetime(s, errors='coerce', utc=True)\n```\n`errors='coerce'` turns unparseable values into NaT instead of raising.")

    add("data",
        "Write a function that loads CSV in chunks and aggregates.",
        "```python\nimport pandas as pd\n\ndef chunk_sum(path: str, col: str, chunksize: int = 1_000_000) -> float:\n    total = 0.0\n    for chunk in pd.read_csv(path, usecols=[col], chunksize=chunksize):\n        total += chunk[col].sum()\n    return total\n```\nProcessing chunks keeps memory bounded for files larger than RAM.")

    add("data",
        "Implement a function that creates a categorical column.",
        "```python\nimport pandas as pd\n\ndef as_cat(df: pd.DataFrame, col: str, ordered: list[str] | None = None) -> pd.DataFrame:\n    df = df.copy()\n    df[col] = pd.Categorical(df[col], categories=ordered, ordered=ordered is not None)\n    return df\n```\nOrdered categoricals preserve a meaningful sort (e.g. low < med < high).")

    add("data",
        "Write a function that exports a DataFrame to parquet with snappy compression.",
        "```python\nimport pandas as pd\n\ndef to_parquet(df: pd.DataFrame, path: str) -> None:\n    df.to_parquet(path, compression='snappy', index=False)\n```\nSnappy is the default; `zstd` compresses better at modest CPU cost.")

    add("data",
        "Implement a function that returns a DataFrame's memory usage by column.",
        "```python\nimport pandas as pd\n\ndef memory_per_col(df: pd.DataFrame) -> pd.Series:\n    return df.memory_usage(deep=True).sort_values(ascending=False)\n```\n`deep=True` accounts for object columns -- otherwise pointers are all you see.")

    add("data",
        "Write a function that downsamples a numeric column to a target count.",
        "```python\nimport pandas as pd\n\ndef downsample(s: pd.Series, target: int) -> pd.Series:\n    if len(s) <= target:\n        return s\n    return s.sample(n=target, random_state=0)\n```\nFix the seed so the sample is reproducible.")
