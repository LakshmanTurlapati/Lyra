# SPDX-License-Identifier: MIT
"""Web / API entries (~130). Wave 3 part 3."""


def register_all(add):
    add("web",
        "Write a minimal FastAPI endpoint that echoes a JSON body.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Echo(BaseModel):\n    message: str\n\n@app.post('/echo')\ndef echo(body: Echo) -> Echo:\n    return body\n```\nPydantic models give you validation, OpenAPI docs, and auto-serialization for free.")

    add("web",
        "Show how to send an HTTP GET request with a timeout using `httpx`.",
        "```python\nimport httpx\n\nresp = httpx.get('https://api.example.com/data', timeout=5.0)\nresp.raise_for_status()\ndata = resp.json()\n```\nAlways set a timeout; the default is no timeout, which can hang indefinitely.")

    add("web",
        "Write a Flask route that returns JSON with a 201 status code.",
        "```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.post('/items')\ndef create_item():\n    payload = request.get_json(force=True)\n    return jsonify({'id': 1, **payload}), 201\n```\n`force=True` will parse even if the Content-Type header is wrong; in production prefer to validate the header.")

    add("web",
        "Show how to use `httpx` for an async GET request.",
        "```python\nimport httpx\n\nasync def fetch(url: str) -> dict:\n    async with httpx.AsyncClient(timeout=10.0) as client:\n        resp = await client.get(url)\n        resp.raise_for_status()\n        return resp.json()\n```\nReuse the client across calls in real apps so connection pooling works.")

    add("web",
        "Write a FastAPI endpoint that accepts a query parameter with a default.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/search')\ndef search(q: str = '', limit: int = 10):\n    return {'query': q, 'limit': limit}\n```\nType annotations drive query-string parsing and validation.")

    add("web",
        "Show how to add CORS middleware to FastAPI.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://app.example.com'],\n    allow_methods=['GET', 'POST'],\n    allow_headers=['*'],\n)\n```\nAvoid `allow_origins=['*']` together with `allow_credentials=True`; browsers reject that combo.")

    add("web",
        "Write a function that downloads a file with `requests`, streaming to disk.",
        "```python\nimport requests\n\ndef download(url: str, dest: str) -> None:\n    with requests.get(url, stream=True, timeout=30) as r:\n        r.raise_for_status()\n        with open(dest, 'wb') as f:\n            for chunk in r.iter_content(chunk_size=64 * 1024):\n                f.write(chunk)\n```\n`stream=True` plus `iter_content` keeps memory bounded for large files.")

    add("web",
        "Write a FastAPI endpoint that uploads a file.",
        "```python\nfrom fastapi import FastAPI, UploadFile, File\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload(file: UploadFile = File(...)):\n    content = await file.read()\n    return {'filename': file.filename, 'size': len(content)}\n```\nFor large files, stream chunks with `await file.read(chunk_size)` instead of loading everything into memory.")

    add("web",
        "Show how to read JSON from a `requests.Response`, handling decode errors.",
        "```python\nimport requests\n\nresp = requests.get('https://api.example.com/data', timeout=10)\nresp.raise_for_status()\ntry:\n    data = resp.json()\nexcept ValueError:\n    raise RuntimeError(f'expected JSON, got: {resp.text[:200]!r}')\n```\nA non-JSON 200 response is a sign that an upstream proxy or login page intercepted the request.")

    add("web",
        "Write a Flask handler that returns a 404 for missing resources.",
        "```python\nfrom flask import Flask, abort, jsonify\n\napp = Flask(__name__)\n\n@app.get('/items/<int:item_id>')\ndef get_item(item_id: int):\n    item = lookup(item_id)\n    if item is None:\n        abort(404)\n    return jsonify(item)\n\n@app.errorhandler(404)\ndef not_found(_):\n    return jsonify({'error': 'not found'}), 404\n```\nCentralizing the 404 handler keeps the body shape consistent across routes.")

    add("web",
        "Show how to set a User-Agent header and basic auth in `requests`.",
        "```python\nimport requests\n\nresp = requests.get(\n    'https://api.example.com/secure',\n    headers={'User-Agent': 'MyApp/1.0'},\n    auth=('user', 'pass'),\n    timeout=10,\n)\nresp.raise_for_status()\n```\nUse `HTTPBasicAuth` for explicitness or pass a tuple for the shorthand.")

    add("web",
        "Write a FastAPI dependency that injects a database session.",
        "```python\nfrom fastapi import Depends\nfrom sqlalchemy.orm import Session\n\ndef get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n\n@app.get('/users')\ndef list_users(db: Session = Depends(get_db)):\n    return db.query(User).all()\n```\nThe `try/finally` around `yield` guarantees cleanup even on exceptions.")

    add("web",
        "Show how to retry a failing HTTP call with exponential backoff.",
        "```python\nimport time\nimport httpx\n\ndef get_with_retry(url: str, max_attempts: int = 5) -> httpx.Response:\n    delay = 1.0\n    for attempt in range(max_attempts):\n        try:\n            r = httpx.get(url, timeout=10)\n            if r.status_code < 500:\n                return r\n        except httpx.RequestError:\n            if attempt == max_attempts - 1:\n                raise\n        time.sleep(delay)\n        delay *= 2\n    return r\n```\nFor production, prefer the `tenacity` library -- it adds jitter and proper backoff policies.")

    add("web",
        "Write a FastAPI endpoint that returns paginated results.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\nITEMS = list(range(1000))\n\n@app.get('/items')\ndef list_items(page: int = 1, size: int = 20):\n    page = max(1, page)\n    size = min(max(1, size), 100)\n    start = (page - 1) * size\n    return {\n        'items': ITEMS[start:start + size],\n        'page': page,\n        'total': len(ITEMS),\n    }\n```\nAlways clamp `size` server-side; otherwise an attacker can dump the entire table in one request.")

    add("web",
        "Show how to send a POST with form-encoded data using `requests`.",
        "```python\nimport requests\n\nresp = requests.post(\n    'https://api.example.com/login',\n    data={'username': 'u', 'password': 'p'},\n    timeout=10,\n)\nresp.raise_for_status()\n```\n`data=` sends form-encoded; `json=` sends JSON. Don't pass both.")

    add("web",
        "Write a Flask endpoint that streams a long-running response.",
        "```python\nfrom flask import Flask, Response\nimport time\n\napp = Flask(__name__)\n\n@app.get('/stream')\ndef stream():\n    def gen():\n        for i in range(10):\n            yield f'event {i}\\n'\n            time.sleep(0.5)\n    return Response(gen(), mimetype='text/plain')\n```\nGenerator-based streaming keeps memory low and lets clients see partial output.")

    add("web",
        "Write a FastAPI WebSocket endpoint that echoes messages.",
        "```python\nfrom fastapi import FastAPI, WebSocket\n\napp = FastAPI()\n\n@app.websocket('/ws')\nasync def ws_echo(websocket: WebSocket):\n    await websocket.accept()\n    try:\n        while True:\n            data = await websocket.receive_text()\n            await websocket.send_text(f'echo: {data}')\n    except Exception:\n        await websocket.close()\n```\nAlways guard the loop so disconnects don't propagate as 500s.")

    add("web",
        "Show how to parse a URL into components.",
        "```python\nfrom urllib.parse import urlparse, parse_qs\n\nu = urlparse('https://example.com/path?a=1&b=2')\nprint(u.scheme, u.netloc, u.path, parse_qs(u.query))\n```\n`parse_qs` returns lists per key (since query strings can repeat keys). Use `parse_qsl` for an ordered list.")

    add("web",
        "Write a function that builds a URL from base + path + query params.",
        "```python\nfrom urllib.parse import urljoin, urlencode\n\ndef build_url(base: str, path: str, **params) -> str:\n    url = urljoin(base, path)\n    if params:\n        url += '?' + urlencode(params, doseq=True)\n    return url\n```\n`doseq=True` properly serializes list-valued parameters (`tag=a&tag=b`).")

    add("web",
        "Show how to mock an HTTP call with `responses` for testing.",
        "```python\nimport responses\nimport requests\n\n@responses.activate\ndef test_call():\n    responses.add(responses.GET, 'https://api.example.com/x', json={'ok': True}, status=200)\n    r = requests.get('https://api.example.com/x', timeout=5)\n    assert r.json() == {'ok': True}\n```\nThe `responses` library intercepts at the `requests` adapter layer -- no real network traffic.")

    add("web",
        "Write a Flask blueprint to organize related routes.",
        "```python\nfrom flask import Blueprint, jsonify\n\nbp = Blueprint('api_v1', __name__, url_prefix='/api/v1')\n\n@bp.get('/health')\ndef health():\n    return jsonify({'status': 'ok'})\n\n# In app factory:\n# app.register_blueprint(bp)\n```\nBlueprints scale a Flask app without forcing a single huge module.")

    add("web",
        "Show how to use `aiohttp` to fetch many URLs concurrently.",
        "```python\nimport asyncio\nimport aiohttp\n\nasync def fetch(session, url):\n    async with session.get(url) as r:\n        return await r.text()\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        return await asyncio.gather(*[fetch(session, u) for u in urls])\n\n# texts = asyncio.run(fetch_all(['https://example.com', 'https://example.org']))\n```\nReuse a single `ClientSession`; creating one per call wastes connection setup.")

    add("web",
        "Write a FastAPI endpoint that returns 422 with details on validation errors.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, Field\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str = Field(min_length=1, max_length=64)\n    quantity: int = Field(ge=1, le=1000)\n\n@app.post('/items')\ndef create(item: Item):\n    return item\n```\nFastAPI returns a structured 422 automatically; you only need to define the Pydantic constraints.")

    add("web",
        "Show how to set request and response middleware in FastAPI.",
        "```python\nfrom fastapi import FastAPI, Request\nimport time\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def add_timing(request: Request, call_next):\n    start = time.perf_counter()\n    response = await call_next(request)\n    response.headers['X-Process-Time'] = f'{(time.perf_counter() - start):.3f}'\n    return response\n```\nKeep middleware tight -- it runs on every request.")

    add("web",
        "Write a Flask endpoint that returns a CSV download.",
        "```python\nimport csv\nimport io\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/export.csv')\ndef export():\n    buf = io.StringIO()\n    w = csv.writer(buf)\n    w.writerow(['id', 'name'])\n    w.writerow([1, 'Alice'])\n    return Response(buf.getvalue(), mimetype='text/csv',\n                    headers={'Content-Disposition': 'attachment; filename=export.csv'})\n```\nFor large exports, use a generator and `stream_with_context`.")

    add("web",
        "Show how to validate an incoming webhook signature using HMAC.",
        "```python\nimport hmac\nimport hashlib\n\ndef verify(secret: bytes, body: bytes, signature: str) -> bool:\n    expected = hmac.new(secret, body, hashlib.sha256).hexdigest()\n    return hmac.compare_digest(expected, signature)\n```\n`hmac.compare_digest` is constant-time; `==` leaks timing info.")

    add("web",
        "Write a FastAPI endpoint that returns a redirect.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import RedirectResponse\n\napp = FastAPI()\n\n@app.get('/old')\ndef old():\n    return RedirectResponse('/new', status_code=301)\n```\nUse 301 for permanent redirects, 302/307 for temporary.")

    add("web",
        "Show how to read environment variables for app configuration.",
        "```python\nimport os\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    database_url: str\n    api_key: str = ''\n    debug: bool = False\n    class Config:\n        env_file = '.env'\n\nsettings = Settings()\n```\n`pydantic-settings` gives you typed config without scattering `os.environ` calls.")

    add("web",
        "Write a function that uses `requests.Session` for connection pooling.",
        "```python\nimport requests\n\nsession = requests.Session()\nsession.headers.update({'User-Agent': 'MyApp/1.0'})\n\ndef get(path: str) -> dict:\n    resp = session.get(f'https://api.example.com{path}', timeout=10)\n    resp.raise_for_status()\n    return resp.json()\n```\nA shared `Session` reuses TCP connections; significantly faster across many calls.")

    add("web",
        "Show how to define a path parameter in FastAPI.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/users/{user_id}')\ndef get_user(user_id: int):\n    return {'user_id': user_id}\n```\nThe type annotation drives parsing; non-int paths return 422 automatically.")

    add("web",
        "Write a function that polls an endpoint until it returns 200 or times out.",
        "```python\nimport time\nimport httpx\n\ndef poll_until_ready(url: str, timeout: float = 30.0) -> bool:\n    deadline = time.monotonic() + timeout\n    while time.monotonic() < deadline:\n        try:\n            if httpx.get(url, timeout=2).status_code == 200:\n                return True\n        except httpx.RequestError:\n            pass\n        time.sleep(1.0)\n    return False\n```\n`time.monotonic()` is immune to wall-clock jumps; `time.time()` isn't.")

    add("web",
        "Show how to define a request body and response model in FastAPI.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass CreateUser(BaseModel):\n    name: str\n    email: str\n\nclass User(BaseModel):\n    id: int\n    name: str\n    email: str\n\n@app.post('/users', response_model=User)\ndef create_user(payload: CreateUser):\n    return User(id=1, **payload.model_dump())\n```\nSeparate request and response models keep internal fields out of API surfaces.")

    add("web",
        "Write a Flask app factory pattern.",
        "```python\nfrom flask import Flask\n\ndef create_app(config: dict | None = None) -> Flask:\n    app = Flask(__name__)\n    if config:\n        app.config.update(config)\n    from .routes import bp\n    app.register_blueprint(bp)\n    return app\n```\nFactory pattern avoids module-level global state and makes testing easy.")

    add("web",
        "Show how to handle JSON serialization of datetime in Flask.",
        "```python\nfrom datetime import datetime\nfrom flask import Flask\nfrom flask.json.provider import DefaultJSONProvider\n\nclass ISOJSONProvider(DefaultJSONProvider):\n    def default(self, obj):\n        if isinstance(obj, datetime):\n            return obj.isoformat()\n        return super().default(obj)\n\napp = Flask(__name__)\napp.json = ISOJSONProvider(app)\n```\nFastAPI does this automatically via Pydantic; Flask needs a custom provider.")

    add("web",
        "Write an endpoint that returns server-sent events.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\nimport asyncio\n\napp = FastAPI()\n\n@app.get('/events')\nasync def events():\n    async def gen():\n        for i in range(5):\n            yield f'data: tick {i}\\n\\n'\n            await asyncio.sleep(1)\n    return StreamingResponse(gen(), media_type='text/event-stream')\n```\nSSE format is one or more `field: value\\n` lines per event, separated by a blank line.")

    add("web",
        "Show how to read a JWT from the Authorization header.",
        "```python\nfrom fastapi import FastAPI, Header, HTTPException\nimport jwt\n\napp = FastAPI()\nSECRET = 'change-me'\n\n@app.get('/me')\ndef me(authorization: str = Header(...)):\n    if not authorization.startswith('Bearer '):\n        raise HTTPException(401, 'missing bearer token')\n    token = authorization[7:]\n    try:\n        payload = jwt.decode(token, SECRET, algorithms=['HS256'])\n    except jwt.PyJWTError:\n        raise HTTPException(401, 'invalid token')\n    return payload\n```\nAlways pass the explicit `algorithms` list; omitting it has caused real CVEs.")

    add("web",
        "Write a function that uploads JSON via a POST request.",
        "```python\nimport httpx\n\ndef post_json(url: str, payload: dict) -> dict:\n    r = httpx.post(url, json=payload, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nUsing `json=` automatically sets `Content-Type: application/json` and serializes the dict.")

    add("web",
        "Show how to use FastAPI's `BackgroundTasks` for fire-and-forget work.",
        "```python\nfrom fastapi import FastAPI, BackgroundTasks\n\napp = FastAPI()\n\ndef send_email(addr: str, body: str) -> None:\n    pass  # do the work\n\n@app.post('/notify')\ndef notify(addr: str, tasks: BackgroundTasks):\n    tasks.add_task(send_email, addr, 'hello')\n    return {'queued': True}\n```\nFor real workloads (retries, persistence) use a queue like Celery or Arq.")

    add("web",
        "Write a Flask endpoint that requires an API key from the X-API-Key header.",
        "```python\nimport os\nfrom functools import wraps\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\ndef require_api_key(fn):\n    @wraps(fn)\n    def wrapper(*args, **kwargs):\n        if request.headers.get('X-API-Key') != os.environ.get('API_KEY'):\n            return jsonify({'error': 'unauthorized'}), 401\n        return fn(*args, **kwargs)\n    return wrapper\n\n@app.get('/secure')\n@require_api_key\ndef secure():\n    return jsonify({'ok': True})\n```\nWrap with `functools.wraps` so the decorator preserves the function's name/docstring.")

    add("web",
        "Show how to gzip a response in FastAPI.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.gzip import GZipMiddleware\n\napp = FastAPI()\napp.add_middleware(GZipMiddleware, minimum_size=1000)\n```\nResponses smaller than `minimum_size` skip compression; the CPU cost outweighs the bandwidth savings.")

    add("web",
        "Write a function that uses `httpx.AsyncClient` with retries via tenacity.",
        "```python\nimport httpx\nfrom tenacity import retry, stop_after_attempt, wait_exponential\n\n@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))\nasync def fetch(url: str) -> dict:\n    async with httpx.AsyncClient(timeout=10) as client:\n        r = await client.get(url)\n        r.raise_for_status()\n        return r.json()\n```\n`tenacity` handles backoff, jitter, and exception filtering with a single decorator.")

    add("web",
        "Show how to define a router with a tag in FastAPI.",
        "```python\nfrom fastapi import APIRouter, FastAPI\n\nrouter = APIRouter(prefix='/users', tags=['users'])\n\n@router.get('/')\ndef list_users():\n    return []\n\napp = FastAPI()\napp.include_router(router)\n```\nTags group endpoints in the OpenAPI docs; prefixes scale better than per-route paths.")

    add("web",
        "Write a function that handles GraphQL responses with errors.",
        "```python\nimport httpx\n\ndef gql(url: str, query: str, variables: dict | None = None) -> dict:\n    r = httpx.post(url, json={'query': query, 'variables': variables or {}}, timeout=10)\n    r.raise_for_status()\n    payload = r.json()\n    if payload.get('errors'):\n        raise RuntimeError(payload['errors'])\n    return payload['data']\n```\nGraphQL returns 200 even on errors; you must inspect the `errors` field.")

    add("web",
        "Show how to handle multipart form upload with `requests`.",
        "```python\nimport requests\n\nwith open('photo.jpg', 'rb') as f:\n    r = requests.post(\n        'https://api.example.com/upload',\n        files={'photo': ('photo.jpg', f, 'image/jpeg')},\n        timeout=30,\n    )\nr.raise_for_status()\n```\nThe 3-tuple `(filename, fp, content_type)` is the canonical form and avoids guessing.")

    add("web",
        "Write a Flask endpoint that returns the request's IP address.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.get('/ip')\ndef ip():\n    fwd = request.headers.get('X-Forwarded-For', '')\n    real_ip = fwd.split(',')[0].strip() if fwd else request.remote_addr\n    return jsonify({'ip': real_ip})\n```\nBehind a load balancer, `remote_addr` is the LB. Trust `X-Forwarded-For` only if you control the upstream proxy.")

    add("web",
        "Show how to register a startup/shutdown handler in FastAPI.",
        "```python\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    # startup\n    app.state.client = create_client()\n    yield\n    # shutdown\n    await app.state.client.close()\n\napp = FastAPI(lifespan=lifespan)\n```\nLifespan replaces the deprecated `@app.on_event` decorators.")

    add("web",
        "Write a FastAPI endpoint that takes a list query parameter.",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\n\n@app.get('/items')\ndef list_items(tag: list[str] = Query(default=[])):\n    return {'tags': tag}\n```\nClients pass `?tag=a&tag=b` -- `list[str]` plus `Query` collects them into a list.")

    add("web",
        "Show how to handle a custom exception in FastAPI.",
        "```python\nfrom fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\n\nclass NotFound(Exception):\n    pass\n\napp = FastAPI()\n\n@app.exception_handler(NotFound)\nasync def not_found_handler(_: Request, exc: NotFound):\n    return JSONResponse(status_code=404, content={'detail': str(exc)})\n```\nDomain exceptions in handlers, HTTP translation in handlers' handlers -- keeps business logic free of HTTP.")

    add("web",
        "Write a function that posts a Slack webhook message.",
        "```python\nimport httpx\n\ndef slack_post(webhook_url: str, text: str) -> None:\n    r = httpx.post(webhook_url, json={'text': text}, timeout=10)\n    r.raise_for_status()\n```\nKeep the webhook URL in env vars or a secret manager, never in source.")

    add("web",
        "Show how to set up a basic HTTP server with `http.server`.",
        "```python\nfrom http.server import HTTPServer, BaseHTTPRequestHandler\n\nclass Handler(BaseHTTPRequestHandler):\n    def do_GET(self):\n        self.send_response(200)\n        self.send_header('Content-Type', 'text/plain')\n        self.end_headers()\n        self.wfile.write(b'hello\\n')\n\nif __name__ == '__main__':\n    HTTPServer(('localhost', 8000), Handler).serve_forever()\n```\nGood for quick local mocks; use a real framework for anything production-bound.")

    add("web",
        "Write a Flask endpoint that returns the current server timestamp.",
        "```python\nfrom datetime import datetime, timezone\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/now')\ndef now():\n    return jsonify({'now': datetime.now(timezone.utc).isoformat()})\n```\nAlways return UTC; let the client convert to local time as needed.")

    add("web",
        "Show how to log every request in FastAPI with structured fields.",
        "```python\nimport logging\nfrom fastapi import FastAPI, Request\n\nlog = logging.getLogger(__name__)\napp = FastAPI()\n\n@app.middleware('http')\nasync def access_log(request: Request, call_next):\n    response = await call_next(request)\n    log.info('access', extra={\n        'method': request.method,\n        'path': request.url.path,\n        'status': response.status_code,\n    })\n    return response\n```\nUse `extra={}` for structured fields; configure a JSON formatter to capture them.")

    add("web",
        "Write a function that downloads JSON with a timeout and parses it safely.",
        "```python\nimport httpx\n\ndef fetch_json(url: str, timeout: float = 5.0) -> dict | list | None:\n    try:\n        r = httpx.get(url, timeout=timeout)\n        r.raise_for_status()\n        return r.json()\n    except (httpx.RequestError, ValueError):\n        return None\n```\nReturning `None` on failure forces callers to handle errors at the call site.")

    add("web",
        "Show how to add OpenAPI metadata to FastAPI.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI(\n    title='Catalog API',\n    description='Manage products and inventory.',\n    version='1.4.0',\n    contact={'name': 'API Team', 'email': 'api@example.com'},\n)\n```\nThe metadata feeds the auto-generated `/docs` and `/openapi.json`.")

    add("web",
        "Write a function that exposes a Prometheus metrics endpoint.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import PlainTextResponse\nfrom prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST\n\napp = FastAPI()\nrequests_total = Counter('requests_total', 'Total requests', ['method', 'path'])\n\n@app.get('/metrics')\ndef metrics():\n    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)\n```\nFor full coverage use `prometheus-fastapi-instrumentator` instead of hand-rolling.")

    add("web",
        "Show how to handle a graceful shutdown in a Flask + gunicorn setup.",
        "```python\nimport signal\nimport sys\n\ndef shutdown_handler(_signum, _frame):\n    # close DB pools, drain queues, etc.\n    sys.exit(0)\n\nsignal.signal(signal.SIGTERM, shutdown_handler)\n```\nGunicorn sends SIGTERM, then SIGKILL after `--graceful-timeout`. Use the window to drain.")

    add("web",
        "Write an endpoint that returns binary data with the correct Content-Type.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import Response\n\napp = FastAPI()\n\n@app.get('/pixel.png')\ndef pixel():\n    png = bytes.fromhex('89504e470d0a1a0a' '0000000d49484452' '00000001000000010806000000' '1f15c489' '0000000a49444154789c63000100000500010d0a2db4' '0000000049454e44ae426082')\n    return Response(content=png, media_type='image/png')\n```\nReturn raw bytes via `Response` to bypass JSON serialization.")

    add("web",
        "Show how to upload to S3 using boto3.",
        "```python\nimport boto3\n\ns3 = boto3.client('s3')\ns3.upload_file('local.txt', 'my-bucket', 'remote/path/local.txt')\n```\n`upload_file` handles multipart uploads automatically for large objects.")

    add("web",
        "Write a Flask endpoint that handles a webhook with idempotency.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nseen_ids: set = set()\n\n@app.post('/webhook')\ndef webhook():\n    event_id = request.headers.get('X-Event-Id')\n    if not event_id:\n        return jsonify({'error': 'missing X-Event-Id'}), 400\n    if event_id in seen_ids:\n        return jsonify({'status': 'duplicate'}), 200\n    seen_ids.add(event_id)\n    # process event\n    return jsonify({'status': 'ok'}), 200\n```\nIn-memory set is illustrative; real systems persist to Redis or a DB with TTL.")

    add("web",
        "Show how to call a SOAP service with `zeep`.",
        "```python\nfrom zeep import Client\n\nclient = Client('https://example.com/service?wsdl')\nresult = client.service.GetWeather(City='Seattle')\n```\n`zeep` introspects the WSDL so you call methods by name. SOAP feels archaic but is alive in enterprise.")

    add("web",
        "Write a function that generates a presigned S3 URL.",
        "```python\nimport boto3\n\ndef presigned_get(bucket: str, key: str, expires: int = 3600) -> str:\n    s3 = boto3.client('s3')\n    return s3.generate_presigned_url('get_object',\n                                     Params={'Bucket': bucket, 'Key': key},\n                                     ExpiresIn=expires)\n```\nKeep `expires` short for sensitive content; longer is convenient but riskier.")

    add("web",
        "Show how to send batched requests with httpx for efficiency.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_all(urls: list[str]) -> list[str]:\n    async with httpx.AsyncClient(timeout=10) as client:\n        return await asyncio.gather(*[client.get(u) for u in urls])\n\n# results = asyncio.run(fetch_all(['https://a', 'https://b']))\n```\n`asyncio.gather` runs concurrently and respects the client's connection pool.")

    add("web",
        "Write a Flask endpoint that requires a valid CSRF token.",
        "```python\nfrom flask import Flask\nfrom flask_wtf import CSRFProtect\n\napp = Flask(__name__)\napp.config['SECRET_KEY'] = 'change-me'\nCSRFProtect(app)\n```\n`flask-wtf` automatically rejects POSTs without a valid CSRF token. For APIs (no cookies) you don't need CSRF; use bearer tokens instead.")

    add("web",
        "Show how to use FastAPI's testclient to test routes.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\nclient = TestClient(app)\n\ndef test_health():\n    r = client.get('/health')\n    assert r.status_code == 200\n    assert r.json() == {'status': 'ok'}\n```\n`TestClient` calls handlers in-process; no socket required.")

    add("web",
        "Write a function that posts a Discord webhook.",
        "```python\nimport httpx\n\ndef discord_post(webhook_url: str, content: str) -> None:\n    r = httpx.post(webhook_url, json={'content': content[:2000]}, timeout=10)\n    r.raise_for_status()\n```\nDiscord caps message length at 2000 chars; truncate to avoid 400s.")

    add("web",
        "Show how to use a connection pool with `httpx.Client`.",
        "```python\nimport httpx\n\nclient = httpx.Client(\n    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),\n    timeout=httpx.Timeout(connect=5, read=30, write=10, pool=5),\n)\n```\nTune limits to match your concurrency; the defaults are conservative.")

    add("web",
        "Write a FastAPI endpoint that requires basic HTTP auth.",
        "```python\nimport secrets\nfrom fastapi import FastAPI, Depends, HTTPException, status\nfrom fastapi.security import HTTPBasic, HTTPBasicCredentials\n\napp = FastAPI()\nsecurity = HTTPBasic()\n\ndef check(credentials: HTTPBasicCredentials = Depends(security)):\n    if not (secrets.compare_digest(credentials.username, 'admin')\n            and secrets.compare_digest(credentials.password, 'secret')):\n        raise HTTPException(status.HTTP_401_UNAUTHORIZED, headers={'WWW-Authenticate': 'Basic'})\n    return credentials.username\n\n@app.get('/admin')\ndef admin(user: str = Depends(check)):\n    return {'user': user}\n```\nBasic auth over plain HTTP leaks credentials; only use it behind TLS.")

    add("web",
        "Show how to read environment-specific config in FastAPI.",
        "```python\nimport os\nfrom functools import lru_cache\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    env: str = 'dev'\n    database_url: str\n\n@lru_cache\ndef get_settings() -> Settings:\n    return Settings()\n```\n`lru_cache` makes `get_settings()` a singleton without a global.")

    add("web",
        "Write a function that paginates through a cursor-based API.",
        "```python\nimport httpx\nfrom typing import Iterator\n\ndef paginate(url: str) -> Iterator[dict]:\n    cursor = None\n    while True:\n        params = {'cursor': cursor} if cursor else {}\n        r = httpx.get(url, params=params, timeout=10)\n        r.raise_for_status()\n        data = r.json()\n        yield from data['items']\n        cursor = data.get('next_cursor')\n        if not cursor:\n            return\n```\nGenerator-based pagination plays nicely with `for` and stays memory-light.")

    add("web",
        "Show how to validate an email field with Pydantic.",
        "```python\nfrom pydantic import BaseModel, EmailStr\n\nclass Signup(BaseModel):\n    email: EmailStr\n    name: str\n```\n`EmailStr` requires `pip install pydantic[email]`; backed by `email-validator`.")

    add("web",
        "Write a Flask endpoint that downloads a remote file and returns it.",
        "```python\nimport requests\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/proxy')\ndef proxy():\n    url = 'https://example.com/file.bin'\n    upstream = requests.get(url, stream=True, timeout=30)\n    return Response(upstream.iter_content(chunk_size=8192),\n                    content_type=upstream.headers.get('Content-Type', 'application/octet-stream'))\n```\nProxy carefully -- whitelist allowed URLs to avoid SSRF.")

    add("web",
        "Show how to add request ID propagation in FastAPI.",
        "```python\nimport uuid\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def request_id(request: Request, call_next):\n    rid = request.headers.get('X-Request-Id', str(uuid.uuid4()))\n    response = await call_next(request)\n    response.headers['X-Request-Id'] = rid\n    return response\n```\nLog the rid on every record; it makes cross-service tracing trivial.")

    add("web",
        "Write a function that calls an OAuth2 token endpoint.",
        "```python\nimport httpx\n\ndef get_token(client_id: str, client_secret: str, token_url: str) -> str:\n    r = httpx.post(\n        token_url,\n        data={'grant_type': 'client_credentials'},\n        auth=(client_id, client_secret),\n        timeout=10,\n    )\n    r.raise_for_status()\n    return r.json()['access_token']\n```\nClient credentials goes via Basic auth to the token endpoint; cache the token until just before its expiry.")

    add("web",
        "Show how to send a header on every request via httpx event hooks.",
        "```python\nimport httpx\n\ndef add_header(request):\n    request.headers['X-App'] = 'lyra'\n\nclient = httpx.Client(event_hooks={'request': [add_header]})\n```\nEvent hooks centralize cross-cutting concerns (auth, tracing) without wrapping every call.")

    add("web",
        "Write a function that publishes to RabbitMQ via `pika`.",
        "```python\nimport json\nimport pika\n\ndef publish(queue: str, payload: dict) -> None:\n    conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))\n    try:\n        ch = conn.channel()\n        ch.queue_declare(queue=queue, durable=True)\n        ch.basic_publish(\n            exchange='',\n            routing_key=queue,\n            body=json.dumps(payload).encode(),\n            properties=pika.BasicProperties(delivery_mode=2),\n        )\n    finally:\n        conn.close()\n```\n`delivery_mode=2` makes the message persistent; otherwise broker restart drops it.")

    add("web",
        "Show how to consume from a Redis Stream with `redis-py`.",
        "```python\nimport redis\n\nr = redis.Redis()\nlast_id = '0'\nwhile True:\n    resp = r.xread({'events': last_id}, block=1000, count=10)\n    for stream, msgs in resp or []:\n        for msg_id, fields in msgs:\n            handle(fields)\n            last_id = msg_id\n```\nStreams are append-only logs with consumer groups; better than pub/sub for at-least-once.")

    add("web",
        "Write a Flask endpoint that serves a static asset with cache headers.",
        "```python\nfrom flask import Flask, send_from_directory\n\napp = Flask(__name__)\n\n@app.get('/assets/<path:name>')\ndef asset(name: str):\n    response = send_from_directory('static', name, max_age=31536000)\n    response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'\n    return response\n```\n`immutable` is safe only when assets have content-hashed filenames.")

    add("web",
        "Show how to call an API with retries and circuit breaking via tenacity.",
        "```python\nimport httpx\nfrom tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type\n\n@retry(\n    retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),\n    stop=stop_after_attempt(3),\n    wait=wait_exponential(min=1, max=10),\n    reraise=True,\n)\ndef call(url: str) -> dict:\n    r = httpx.get(url, timeout=5)\n    r.raise_for_status()\n    return r.json()\n```\n`reraise=True` lets the original exception escape after the last attempt -- critical for upstream observability.")

    add("web",
        "Write a function that checks API health by calling /health.",
        "```python\nimport httpx\n\ndef is_healthy(base_url: str) -> bool:\n    try:\n        return httpx.get(f'{base_url}/health', timeout=2).status_code == 200\n    except httpx.RequestError:\n        return False\n```\nKeep health checks cheap and dependency-free; they should run constantly.")

    add("web",
        "Show how to use FastAPI to mount static files.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='static'), name='static')\n```\nFor production, prefer serving static files from nginx or a CDN; FastAPI's static handler is fine for dev.")

    add("web",
        "Write a Flask endpoint that returns YAML.",
        "```python\nimport yaml\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/config.yaml')\ndef config():\n    data = {'env': 'prod', 'replicas': 3}\n    return Response(yaml.safe_dump(data), mimetype='application/x-yaml')\n```\nUse `safe_dump`; never `yaml.dump` on untrusted data structures.")

    add("web",
        "Show how to mock httpx calls in tests with `respx`.",
        "```python\nimport httpx\nimport respx\n\n@respx.mock\ndef test_call():\n    route = respx.get('https://api.example.com/x').respond(200, json={'ok': True})\n    r = httpx.get('https://api.example.com/x')\n    assert r.json() == {'ok': True}\n    assert route.called\n```\n`respx` is to httpx what `responses` is to `requests`.")

    add("web",
        "Write a function that authenticates with HTTP digest auth.",
        "```python\nimport httpx\n\ndef get_with_digest(url: str, user: str, password: str) -> httpx.Response:\n    return httpx.get(url, auth=httpx.DigestAuth(user, password), timeout=10)\n```\nDigest auth requires a challenge round-trip but doesn't send the password in plaintext like Basic does.")

    add("web",
        "Show how to test an async FastAPI endpoint.",
        "```python\nimport pytest\nimport httpx\nfrom myapp import app\n\n@pytest.mark.asyncio\nasync def test_async_route():\n    async with httpx.AsyncClient(app=app, base_url='http://test') as client:\n        r = await client.get('/async-route')\n        assert r.status_code == 200\n```\nUse `httpx.AsyncClient(app=app)` to call ASGI apps without spinning up a server.")

    add("web",
        "Write a function that posts to Sentry's HTTP API to capture an event.",
        "```python\nimport sentry_sdk\n\nsentry_sdk.init(dsn='https://key@sentry.io/123')\n\ntry:\n    risky_thing()\nexcept Exception as exc:\n    sentry_sdk.capture_exception(exc)\n    raise\n```\nLet the SDK do the heavy lifting; talking to the HTTP ingress directly is rare.")

    add("web",
        "Show how to handle file streaming download in Flask.",
        "```python\nfrom flask import Flask, send_file\n\napp = Flask(__name__)\n\n@app.get('/file')\ndef file():\n    return send_file('big.bin', as_attachment=True, download_name='big.bin')\n```\n`send_file` uses sendfile() under WSGI servers that support it -- zero-copy from kernel.")

    add("web",
        "Write a function that uploads to Google Cloud Storage.",
        "```python\nfrom google.cloud import storage\n\ndef upload_gcs(bucket: str, src: str, dst: str) -> None:\n    client = storage.Client()\n    blob = client.bucket(bucket).blob(dst)\n    blob.upload_from_filename(src)\n```\nUse application-default credentials in cloud environments; locally `gcloud auth application-default login` sets them up.")

    add("web",
        "Show how to handle an OPTIONS preflight in Flask without flask-cors.",
        "```python\nfrom flask import Flask, make_response\n\napp = Flask(__name__)\n\n@app.route('/api', methods=['OPTIONS', 'GET'])\ndef api():\n    if request.method == 'OPTIONS':\n        resp = make_response()\n        resp.headers['Access-Control-Allow-Origin'] = '*'\n        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST'\n        return resp\n    return {'data': 1}\n```\n`flask-cors` handles this less verbosely; useful to know the underlying mechanism.")

    add("web",
        "Write an endpoint that paginates results from a SQL query.",
        "```python\nfrom fastapi import FastAPI, Query\nfrom sqlalchemy.orm import Session\n\napp = FastAPI()\n\n@app.get('/users')\ndef list_users(db: Session = Depends(get_db),\n               page: int = Query(1, ge=1),\n               size: int = Query(20, ge=1, le=100)):\n    offset = (page - 1) * size\n    users = db.query(User).offset(offset).limit(size).all()\n    total = db.query(User).count()\n    return {'items': users, 'page': page, 'total': total}\n```\nFor large tables, prefer keyset (cursor) pagination -- `OFFSET` gets slow at the tail.")

    add("web",
        "Show how to define a typed dependency in FastAPI.",
        "```python\nfrom typing import Annotated\nfrom fastapi import FastAPI, Depends\n\napp = FastAPI()\n\ndef pagination(page: int = 1, size: int = 20) -> dict:\n    return {'page': page, 'size': size}\n\n@app.get('/items')\ndef list_items(p: Annotated[dict, Depends(pagination)]):\n    return p\n```\n`Annotated` is the modern, less-magic way to write dependencies.")

    add("web",
        "Write a function that pings a list of hosts concurrently.",
        "```python\nimport asyncio\nimport httpx\n\nasync def ping(client: httpx.AsyncClient, host: str) -> tuple[str, bool]:\n    try:\n        r = await client.head(host, timeout=2)\n        return host, r.status_code < 500\n    except httpx.RequestError:\n        return host, False\n\nasync def ping_all(hosts: list[str]) -> dict:\n    async with httpx.AsyncClient() as client:\n        results = await asyncio.gather(*[ping(client, h) for h in hosts])\n    return dict(results)\n```\nUse HEAD to avoid downloading full bodies; use GET if HEAD is unsupported.")

    add("web",
        "Show how to use FastAPI to enforce request size limits.",
        "```python\nfrom fastapi import FastAPI, Request, HTTPException\n\nMAX_SIZE = 1_000_000\napp = FastAPI()\n\n@app.middleware('http')\nasync def limit_size(request: Request, call_next):\n    if int(request.headers.get('content-length', 0)) > MAX_SIZE:\n        raise HTTPException(413, 'payload too large')\n    return await call_next(request)\n```\nDo size enforcement at the reverse-proxy too; the app middleware is a backstop.")

    add("web",
        "Write a function that scrapes a page with `httpx` and `selectolax`.",
        "```python\nimport httpx\nfrom selectolax.parser import HTMLParser\n\ndef titles(url: str) -> list[str]:\n    r = httpx.get(url, timeout=10)\n    r.raise_for_status()\n    return [n.text() for n in HTMLParser(r.text).css('h2.title')]\n```\n`selectolax` is much faster than BeautifulSoup; use it when you scrape at scale.")

    add("web",
        "Show how to wire a Flask app to gunicorn with a WSGI entrypoint.",
        "```python\n# wsgi.py\nfrom myapp import create_app\n\napp = create_app()\n\n# gunicorn -w 4 -b 0.0.0.0:8000 wsgi:app\n```\nUse `-w` (workers) of about `2 * CPU + 1` as a starting point; tune by load testing.")

    add("web",
        "Write a function that posts JSON with a Bearer token.",
        "```python\nimport httpx\n\ndef call_api(url: str, token: str, payload: dict) -> dict:\n    r = httpx.post(url, json=payload, headers={'Authorization': f'Bearer {token}'}, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nRotate tokens periodically; keep a refresh mechanism so calls don't fail at expiry.")

    add("web",
        "Show how to handle SSE on the client with httpx.",
        "```python\nimport httpx\n\nwith httpx.stream('GET', 'https://example.com/events', timeout=None) as r:\n    for line in r.iter_lines():\n        if line.startswith('data: '):\n            print(line[6:])\n```\n`timeout=None` is appropriate for long-lived streams; otherwise reads will time out.")

    add("web",
        "Write a function that returns a paginated DynamoDB scan.",
        "```python\nimport boto3\n\ndef scan_all(table_name: str):\n    table = boto3.resource('dynamodb').Table(table_name)\n    kwargs: dict = {}\n    while True:\n        resp = table.scan(**kwargs)\n        yield from resp.get('Items', [])\n        if 'LastEvaluatedKey' not in resp:\n            return\n        kwargs['ExclusiveStartKey'] = resp['LastEvaluatedKey']\n```\nGenerator hides the pagination from callers and keeps memory bounded.")

    add("web",
        "Show how to do template rendering with Jinja2 in Flask.",
        "```python\nfrom flask import Flask, render_template\n\napp = Flask(__name__)\n\n@app.get('/')\ndef home():\n    return render_template('home.html', name='World')\n```\nKeep templates in `templates/`; Flask discovers them automatically.")

    add("web",
        "Write a function that uses `httpx` proxies for outbound calls.",
        "```python\nimport httpx\n\nclient = httpx.Client(proxies='http://proxy.local:3128', timeout=10)\nresp = client.get('https://api.example.com/data')\n```\nFor a system-wide proxy, `HTTPS_PROXY` env var works without code changes.")

    add("web",
        "Show how to expose an OpenAPI client from FastAPI's spec.",
        "```python\n# Run the app, then download:\n# curl http://localhost:8000/openapi.json -o openapi.json\n# Generate a client:\n# openapi-generator-cli generate -i openapi.json -g python -o ./client\n```\nThis gives you a typed client without hand-writing one. Pin the generator version in CI.")

    add("web",
        "Write a function that publishes a message to Kafka via `confluent-kafka`.",
        "```python\nfrom confluent_kafka import Producer\n\nproducer = Producer({'bootstrap.servers': 'localhost:9092'})\n\ndef publish(topic: str, key: bytes, value: bytes) -> None:\n    producer.produce(topic, key=key, value=value)\n    producer.flush()\n```\n`flush()` ensures delivery before the function returns; for high throughput batch by interval instead.")

    add("web",
        "Show how to handle a DELETE request in FastAPI.",
        "```python\nfrom fastapi import FastAPI, status\n\napp = FastAPI()\n\n@app.delete('/items/{item_id}', status_code=status.HTTP_204_NO_CONTENT)\ndef delete_item(item_id: int):\n    delete_from_db(item_id)\n    return None\n```\n204 No Content is the conventional success status for DELETE.")

    add("web",
        "Write a function that queries Elasticsearch.",
        "```python\nfrom elasticsearch import Elasticsearch\n\nes = Elasticsearch('http://localhost:9200')\n\ndef search(query: str) -> list[dict]:\n    resp = es.search(index='docs', query={'match': {'body': query}})\n    return [hit['_source'] for hit in resp['hits']['hits']]\n```\nWrap calls behind a thin function; the ES client's API is stable but verbose.")

    add("web",
        "Show how to read a Cookie from a FastAPI request.",
        "```python\nfrom fastapi import FastAPI, Cookie\n\napp = FastAPI()\n\n@app.get('/whoami')\ndef whoami(session: str | None = Cookie(default=None)):\n    return {'session': session}\n```\nFor signed cookies use `itsdangerous` or framework-provided session middleware.")

    add("web",
        "Write a Flask endpoint that returns a 429 with a Retry-After header.",
        "```python\nfrom flask import Flask, make_response\n\napp = Flask(__name__)\n\n@app.get('/limited')\ndef limited():\n    if rate_limited():\n        resp = make_response({'error': 'rate limited'}, 429)\n        resp.headers['Retry-After'] = '30'\n        return resp\n    return {'ok': True}\n```\nWell-behaved clients honor `Retry-After`; ignoring it is a sign of bad citizenship.")

    add("web",
        "Show how to expose an HTTP endpoint that returns a Pydantic-validated nested model.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Address(BaseModel):\n    city: str\n    country: str\n\nclass User(BaseModel):\n    name: str\n    address: Address\n\n@app.get('/user', response_model=User)\ndef get_user():\n    return User(name='Lakshman', address=Address(city='Seattle', country='US'))\n```\nNested Pydantic models give you OpenAPI schemas for the entire object graph.")

    add("web",
        "Write a function that downloads a tarball and extracts it.",
        "```python\nimport io\nimport tarfile\nimport httpx\n\ndef fetch_and_extract(url: str, dest: str) -> None:\n    r = httpx.get(url, timeout=60)\n    r.raise_for_status()\n    with tarfile.open(fileobj=io.BytesIO(r.content), mode='r:gz') as tar:\n        tar.extractall(dest)\n```\nFor untrusted archives, validate paths with `tarfile.data_filter` (Python 3.12+) to prevent path traversal.")

    add("web",
        "Show how to use FastAPI to return XML.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import Response\n\napp = FastAPI()\n\n@app.get('/feed.xml')\ndef feed():\n    body = '<rss><channel><title>Hello</title></channel></rss>'\n    return Response(content=body, media_type='application/xml')\n```\nBuild XML strings carefully -- escape user input or use `xml.etree.ElementTree`.")

    add("web",
        "Write a function that uses long polling to wait for a job.",
        "```python\nimport time\nimport httpx\n\ndef wait_for_job(job_id: str, timeout: float = 60.0) -> dict:\n    deadline = time.monotonic() + timeout\n    while time.monotonic() < deadline:\n        r = httpx.get(f'https://api.example.com/jobs/{job_id}', timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        if body['status'] == 'done':\n            return body\n        time.sleep(2)\n    raise TimeoutError(job_id)\n```\nWebSockets / SSE / pubsub are better than polling for low-latency notifications.")

    add("web",
        "Show how to publish to a Redis pub/sub channel.",
        "```python\nimport redis\n\nr = redis.Redis()\nr.publish('events', 'hello')\n```\nPub/sub is fire-and-forget; subscribers must be connected at publish time. For at-least-once delivery use Streams.")

    add("web",
        "Write a function that accepts a webhook with an HMAC signature.",
        "```python\nimport hmac\nimport hashlib\nfrom fastapi import FastAPI, Request, HTTPException\n\napp = FastAPI()\nSECRET = b'shh'\n\n@app.post('/webhook')\nasync def webhook(request: Request):\n    body = await request.body()\n    signature = request.headers.get('X-Signature', '')\n    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()\n    if not hmac.compare_digest(expected, signature):\n        raise HTTPException(401, 'bad signature')\n    return {'ok': True}\n```\nAlways read the raw body (`request.body()`) before parsing -- you need the bytes the sender signed.")

    add("web",
        "Show how to set up SQLAlchemy async with FastAPI.",
        "```python\nfrom sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker\n\nengine = create_async_engine('postgresql+asyncpg://user:pass@host/db')\nSessionLocal = async_sessionmaker(engine, expire_on_commit=False)\n\nasync def get_db():\n    async with SessionLocal() as session:\n        yield session\n```\n`expire_on_commit=False` avoids accidental lazy-load round-trips after commit.")

    add("web",
        "Write a function that uses `dnspython` to resolve a hostname.",
        "```python\nimport dns.resolver\n\ndef resolve(host: str) -> list[str]:\n    return [r.address for r in dns.resolver.resolve(host, 'A')]\n```\nGreat for tools that need DNS independent of the OS resolver (e.g., custom nameservers).")

    add("web",
        "Show how to log slow requests in FastAPI.",
        "```python\nimport time\nimport logging\nfrom fastapi import FastAPI, Request\n\nlog = logging.getLogger(__name__)\napp = FastAPI()\nSLOW_MS = 500\n\n@app.middleware('http')\nasync def log_slow(request: Request, call_next):\n    start = time.perf_counter()\n    response = await call_next(request)\n    elapsed_ms = (time.perf_counter() - start) * 1000\n    if elapsed_ms > SLOW_MS:\n        log.warning('slow', extra={'path': request.url.path, 'ms': elapsed_ms})\n    return response\n```\nSlow logs are gold for diagnosing latency regressions.")

    add("web",
        "Write a function that consumes from an SQS queue.",
        "```python\nimport boto3\n\nsqs = boto3.client('sqs')\n\ndef consume(queue_url: str):\n    while True:\n        resp = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10, WaitTimeSeconds=20)\n        for msg in resp.get('Messages', []):\n            handle(msg)\n            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=msg['ReceiptHandle'])\n```\nLong polling (`WaitTimeSeconds`) reduces empty receives and saves cost.")

    add("web",
        "Show how to handle an HTTP/2 request with httpx.",
        "```python\nimport httpx\n\nwith httpx.Client(http2=True) as client:\n    r = client.get('https://www.cloudflare.com/')\n    print(r.http_version)  # 'HTTP/2'\n```\nHTTP/2 multiplexes requests over one connection; `http2=True` requires the `httpx[http2]` extra.")

    add("web",
        "Write a function that emits structured JSON logs.",
        "```python\nimport json\nimport logging\nimport sys\n\nclass JsonFormatter(logging.Formatter):\n    def format(self, record):\n        payload = {\n            'level': record.levelname,\n            'msg': record.getMessage(),\n            'logger': record.name,\n        }\n        if record.exc_info:\n            payload['exc'] = self.formatException(record.exc_info)\n        for k, v in record.__dict__.get('extra', {}).items():\n            payload[k] = v\n        return json.dumps(payload)\n\nhandler = logging.StreamHandler(sys.stdout)\nhandler.setFormatter(JsonFormatter())\nlogging.basicConfig(level='INFO', handlers=[handler])\n```\nFor anything beyond toy use, prefer `structlog` -- it's batteries-included.")

    add("web",
        "Show how to define rate limiting in FastAPI with slowapi.",
        "```python\nfrom fastapi import FastAPI, Request\nfrom slowapi import Limiter\nfrom slowapi.util import get_remote_address\n\napp = FastAPI()\nlimiter = Limiter(key_func=get_remote_address)\napp.state.limiter = limiter\n\n@app.get('/limited')\n@limiter.limit('5/minute')\ndef limited(request: Request):\n    return {'ok': True}\n```\nFor multi-instance deployments point `slowapi` at Redis as the storage backend.")

    add("web",
        "Write a function that publishes a heartbeat to InfluxDB.",
        "```python\nfrom influxdb_client import InfluxDBClient, Point\nfrom influxdb_client.client.write_api import SYNCHRONOUS\n\nclient = InfluxDBClient(url='http://localhost:8086', token='t', org='o')\nwrite_api = client.write_api(write_options=SYNCHRONOUS)\n\ndef heartbeat(service: str) -> None:\n    point = Point('heartbeat').tag('service', service).field('alive', 1)\n    write_api.write(bucket='ops', record=point)\n```\nReuse the client across calls; tag cardinality matters for storage cost.")

    add("web",
        "Show how to add request tracing with OpenTelemetry in FastAPI.",
        "```python\nfrom fastapi import FastAPI\nfrom opentelemetry.instrumentation.fastapi import FastAPIInstrumentor\n\napp = FastAPI()\nFastAPIInstrumentor.instrument_app(app)\n```\nOne-line instrumentation sends spans to whatever exporter you configure (OTLP, Jaeger, Zipkin).")

    add("web",
        "Write a Flask endpoint with a JSON request body and validation via marshmallow.",
        "```python\nfrom flask import Flask, request, jsonify\nfrom marshmallow import Schema, fields, ValidationError\n\nclass CreateUser(Schema):\n    name = fields.Str(required=True)\n    age = fields.Int(required=True, validate=lambda x: x >= 0)\n\napp = Flask(__name__)\n\n@app.post('/users')\ndef create_user():\n    try:\n        data = CreateUser().load(request.get_json())\n    except ValidationError as exc:\n        return jsonify(exc.messages), 422\n    return jsonify(data), 201\n```\nOr just use Pydantic with Flask -- it works fine outside FastAPI too.")

    add("web",
        "Show how to consume a paginated REST API with `requests`.",
        "```python\nimport requests\n\ndef all_items(base: str):\n    page = 1\n    while True:\n        r = requests.get(f'{base}/items', params={'page': page}, timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        yield from body['items']\n        if not body.get('has_next'):\n            return\n        page += 1\n```\nGenerator-based; the caller decides when to stop iterating.")

    add("web",
        "Write a function that sends an email via SMTP.",
        "```python\nimport smtplib\nfrom email.message import EmailMessage\n\ndef send_email(host: str, port: int, sender: str, to: str, subject: str, body: str) -> None:\n    msg = EmailMessage()\n    msg['From'] = sender\n    msg['To'] = to\n    msg['Subject'] = subject\n    msg.set_content(body)\n    with smtplib.SMTP(host, port) as smtp:\n        smtp.starttls()\n        smtp.send_message(msg)\n```\nFor real apps, use a managed sender (SendGrid, SES) -- direct SMTP gets blacklisted easily.")

    add("web",
        "Show how to run a periodic task with FastAPI's lifespan.",
        "```python\nimport asyncio\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\nasync def heartbeat():\n    while True:\n        # do periodic work\n        await asyncio.sleep(60)\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    task = asyncio.create_task(heartbeat())\n    try:\n        yield\n    finally:\n        task.cancel()\n\napp = FastAPI(lifespan=lifespan)\n```\nFor anything serious, use APScheduler or a separate worker process; the lifespan task dies with the worker.")

    add("web",
        "Write a function that uploads a multipart file to a generic API.",
        "```python\nimport httpx\n\ndef upload(url: str, path: str, field: str = 'file') -> dict:\n    with open(path, 'rb') as f:\n        r = httpx.post(url, files={field: f}, timeout=30)\n    r.raise_for_status()\n    return r.json()\n```\nThe context manager around `open` makes sure the file handle is released even on errors.")

    add("web",
        "Show how to set httpx to follow redirects.",
        "```python\nimport httpx\n\nresp = httpx.get('https://example.com/short', follow_redirects=True, timeout=10)\nprint(resp.url)  # final URL after redirects\n```\nDefault is `follow_redirects=False` -- be explicit so behavior doesn't change with future versions.")

    add("web",
        "Write a Flask endpoint that returns custom error JSON.",
        "```python\nfrom flask import Flask, jsonify\nfrom werkzeug.exceptions import HTTPException\n\napp = Flask(__name__)\n\n@app.errorhandler(HTTPException)\ndef handle_http_error(exc: HTTPException):\n    return jsonify({'error': exc.name, 'detail': exc.description}), exc.code\n```\nCentralized error JSON keeps the API surface uniform.")

    add("web",
        "Show how to use Pydantic to coerce form fields to strict types.",
        "```python\nfrom pydantic import BaseModel, StrictInt\n\nclass Body(BaseModel):\n    quantity: StrictInt\n```\nStrict types reject string-coerced inputs (`'1'`) -- useful when the API demands real ints.")

    add("web",
        "Write a function that benchmarks an HTTP endpoint with concurrent requests.",
        "```python\nimport asyncio\nimport time\nimport httpx\n\nasync def bench(url: str, n: int = 100, concurrency: int = 10) -> float:\n    sem = asyncio.Semaphore(concurrency)\n    async with httpx.AsyncClient(timeout=10) as client:\n        async def one():\n            async with sem:\n                await client.get(url)\n        start = time.perf_counter()\n        await asyncio.gather(*[one() for _ in range(n)])\n        return time.perf_counter() - start\n```\nA semaphore caps concurrency without spawning more tasks than needed.")

    add("web",
        "Show how to implement bearer-token auth as a FastAPI dependency.",
        "```python\nfrom fastapi import FastAPI, Depends, HTTPException\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\n\napp = FastAPI()\nbearer = HTTPBearer()\n\ndef get_user(token: HTTPAuthorizationCredentials = Depends(bearer)) -> str:\n    if token.credentials != 'expected':\n        raise HTTPException(401, 'invalid token')\n    return 'lakshman'\n\n@app.get('/me')\ndef me(user: str = Depends(get_user)):\n    return {'user': user}\n```\nIn production, validate against your identity provider, not a constant.")

    add("web",
        "Write a function that connects to a Postgres database with `psycopg`.",
        "```python\nimport psycopg\n\ndef list_users():\n    with psycopg.connect('postgresql://user:pass@host/db') as conn:\n        with conn.cursor() as cur:\n            cur.execute('SELECT id, name FROM users LIMIT 100')\n            return cur.fetchall()\n```\n`psycopg` 3 supports both sync and async; the context managers handle close/rollback automatically.")

    add("web",
        "Show how to handle conditional GET with ETag in Flask.",
        "```python\nimport hashlib\nfrom flask import Flask, request, make_response\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    body = b'hello'\n    etag = hashlib.md5(body).hexdigest()\n    if request.headers.get('If-None-Match') == etag:\n        return '', 304\n    resp = make_response(body)\n    resp.headers['ETag'] = etag\n    return resp\n```\nETag-based conditional GET saves bandwidth for large unchanged resources.")

    add("web",
        "Write a function that uses `aiofiles` for non-blocking file IO in FastAPI.",
        "```python\nimport aiofiles\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/log')\nasync def get_log():\n    async with aiofiles.open('app.log', 'r') as f:\n        return {'log': await f.read()}\n```\nBlocking file IO inside an async handler stalls the event loop -- `aiofiles` runs IO in a thread pool.")
