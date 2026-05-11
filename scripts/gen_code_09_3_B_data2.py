# SPDX-License-Identifier: MIT
"""Web / API entries: Flask, FastAPI, async, requests, httpx (~125 total)."""


def register_all(add):
    # ---- 3-msg ----
    add("web",
        "Show a minimal Flask app with a single GET endpoint returning JSON.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/health')\ndef health():\n    return jsonify(status='ok')\n\nif __name__ == '__main__':\n    app.run(debug=True)\n```\n`@app.get` (Flask 2.0+) is cleaner than `@app.route('/health', methods=['GET'])`.")

    add("web",
        "Write a FastAPI endpoint that accepts a JSON body and returns it back.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str\n    qty: int\n\n@app.post('/echo')\ndef echo(item: Item) -> Item:\n    return item\n```\nFastAPI uses Pydantic for automatic validation; the type annotation is the schema.")

    add("web",
        "Write an async function that fetches a URL with httpx.",
        "```python\nimport httpx\n\nasync def fetch(url: str) -> str:\n    async with httpx.AsyncClient(timeout=10) as client:\n        resp = await client.get(url)\n        resp.raise_for_status()\n        return resp.text\n```\nAlways set a timeout. `raise_for_status` converts 4xx/5xx into exceptions.")

    add("web",
        "Show how to make a POST request with JSON body using requests.",
        "```python\nimport requests\n\nresp = requests.post('https://api.example.com/items', json={'name': 'x'}, timeout=10)\nresp.raise_for_status()\nprint(resp.json())\n```\nUse `json=` not `data=` -- it sets Content-Type and serializes for you.")

    add("web",
        "Write a Flask route that handles a 404 with JSON response.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.errorhandler(404)\ndef not_found(_):\n    return jsonify(error='not found'), 404\n```\nReturning a tuple `(body, status)` lets Flask set the status code.")

    add("web",
        "Write a FastAPI dependency that checks an API-key header.",
        "```python\nfrom fastapi import FastAPI, Header, HTTPException, Depends\n\napp = FastAPI()\n\ndef require_key(x_api_key: str = Header(...)):\n    if x_api_key != 'secret':\n        raise HTTPException(status_code=401, detail='bad key')\n\n@app.get('/secure', dependencies=[Depends(require_key)])\ndef secure():\n    return {'ok': True}\n```\n`Header(...)` makes the header required; the dependency raises 401 on mismatch.")

    add("web",
        "Show how to upload a file with the requests library.",
        "```python\nimport requests\n\nwith open('report.pdf', 'rb') as f:\n    resp = requests.post('https://api.example.com/upload', files={'file': f}, timeout=30)\nresp.raise_for_status()\n```\n`files=` triggers multipart/form-data automatically.")

    add("web",
        "Write a small aiohttp client that fetches multiple URLs concurrently.",
        "```python\nimport asyncio\nimport aiohttp\n\nasync def fetch(session, url):\n    async with session.get(url) as resp:\n        resp.raise_for_status()\n        return await resp.text()\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        return await asyncio.gather(*(fetch(session, u) for u in urls))\n```\nReuse the session across requests -- creating a new one per request defeats connection pooling.")

    add("web",
        "Write a Flask route that streams a large file response.",
        "```python\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\ndef gen():\n    for i in range(1_000_000):\n        yield f'line {i}\\n'\n\n@app.get('/big')\ndef big():\n    return Response(gen(), mimetype='text/plain')\n```\nA generator passed to `Response` streams the body without buffering everything in memory.")

    add("web",
        "Write a FastAPI websocket echo handler.",
        "```python\nfrom fastapi import FastAPI, WebSocket\n\napp = FastAPI()\n\n@app.websocket('/ws')\nasync def echo(ws: WebSocket):\n    await ws.accept()\n    try:\n        while True:\n            msg = await ws.receive_text()\n            await ws.send_text(msg)\n    except Exception:\n        await ws.close()\n```\nAlways `accept()` first. Wrap the loop in try/except so disconnect doesn't propagate uncaught.")

    add("web",
        "Write a function that uses requests with retries via urllib3 Retry.",
        "```python\nimport requests\nfrom requests.adapters import HTTPAdapter\nfrom urllib3.util.retry import Retry\n\ndef session_with_retries() -> requests.Session:\n    s = requests.Session()\n    retry = Retry(total=5, backoff_factor=0.5, status_forcelist=[502, 503, 504])\n    s.mount('https://', HTTPAdapter(max_retries=retry))\n    s.mount('http://', HTTPAdapter(max_retries=retry))\n    return s\n```\nMount on both schemes. Backoff doubles between attempts.")

    add("web",
        "Write a Flask blueprint for a versioned API.",
        "```python\nfrom flask import Blueprint, jsonify\n\nv1 = Blueprint('v1', __name__, url_prefix='/api/v1')\n\n@v1.get('/users')\ndef users():\n    return jsonify(items=[])\n```\nBlueprints group related endpoints. Register with `app.register_blueprint(v1)`.")

    add("web",
        "Write a FastAPI background task that runs after the response is sent.",
        "```python\nfrom fastapi import FastAPI, BackgroundTasks\n\napp = FastAPI()\n\ndef notify(addr: str, msg: str) -> None:\n    print(f'send to {addr}: {msg}')\n\n@app.post('/order')\ndef order(bg: BackgroundTasks, addr: str):\n    bg.add_task(notify, addr, 'thanks')\n    return {'ok': True}\n```\nThe response returns immediately; the task runs after the response is flushed.")

    add("web",
        "Show how to add CORS to a FastAPI app.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://example.com'],\n    allow_methods=['*'],\n    allow_headers=['*'],\n)\n```\nAvoid `allow_origins=['*']` in production unless you really mean 'any origin'.")

    add("web",
        "Write a Flask route that reads a query parameter with a default.",
        "```python\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n@app.get('/search')\ndef search():\n    q = request.args.get('q', default='', type=str)\n    page = request.args.get('page', default=1, type=int)\n    return {'q': q, 'page': page}\n```\n`type=` casts and falls back to default on failure.")

    add("web",
        "Write a FastAPI path-parameter route with validation.",
        "```python\nfrom fastapi import FastAPI, Path\n\napp = FastAPI()\n\n@app.get('/items/{item_id}')\ndef get_item(item_id: int = Path(..., ge=1, le=10000)):\n    return {'id': item_id}\n```\n`Path(..., ge=..., le=...)` enforces numeric ranges before the body runs.")

    add("web",
        "Use httpx to download a binary file in chunks.",
        "```python\nimport httpx\nfrom pathlib import Path\n\ndef download(url: str, dest: Path) -> None:\n    with httpx.stream('GET', url, timeout=30) as r:\n        r.raise_for_status()\n        with dest.open('wb') as f:\n            for chunk in r.iter_bytes(chunk_size=64 * 1024):\n                f.write(chunk)\n```\n`stream` keeps memory bounded for very large files.")

    add("web",
        "Write a Flask route that returns CSV using StringIO.",
        "```python\nimport csv\nfrom io import StringIO\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/users.csv')\ndef users_csv():\n    buf = StringIO()\n    w = csv.writer(buf)\n    w.writerow(['id', 'name'])\n    w.writerow([1, 'alice'])\n    return Response(buf.getvalue(), mimetype='text/csv')\n```\nFor very large CSVs, use a generator and a streaming Response instead.")

    add("web",
        "Write a FastAPI app that returns custom JSON via ORJSONResponse.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import ORJSONResponse\n\napp = FastAPI(default_response_class=ORJSONResponse)\n\n@app.get('/data')\ndef data():\n    return {'big': list(range(10_000))}\n```\norjson is several times faster than the stdlib for large payloads.")

    add("web",
        "Show how to use httpx OAuth bearer auth.",
        "```python\nimport httpx\n\nresp = httpx.get(\n    'https://api.example.com/me',\n    headers={'Authorization': 'Bearer abc123'},\n    timeout=10,\n)\nresp.raise_for_status()\nprint(resp.json())\n```\nFor more complex flows, use `httpx_auth` or a custom `httpx.Auth` subclass.")

    add("web",
        "Write a FastAPI middleware that logs request duration.",
        "```python\nimport time\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def timing(request: Request, call_next):\n    start = time.monotonic()\n    response = await call_next(request)\n    dt = time.monotonic() - start\n    response.headers['x-process-time'] = f'{dt:.4f}'\n    return response\n```\n`time.monotonic` is the right clock for elapsed-time measurements.")

    add("web",
        "Write a Flask after_request hook that adds a header to every response.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.after_request\ndef add_header(resp):\n    resp.headers['x-app-version'] = '1.2.3'\n    return resp\n```\nAlways return the response from after_request hooks.")

    add("web",
        "Use aiohttp on the server to expose a simple JSON endpoint.",
        "```python\nfrom aiohttp import web\n\nasync def hello(request):\n    return web.json_response({'msg': 'hi'})\n\napp = web.Application()\napp.add_routes([web.get('/', hello)])\n\nif __name__ == '__main__':\n    web.run_app(app, port=8080)\n```\nNo decorators required -- `add_routes` keeps the routing table explicit.")

    add("web",
        "Show how to send a multipart upload using httpx.",
        "```python\nimport httpx\n\nfiles = {'photo': ('p.jpg', open('p.jpg', 'rb'), 'image/jpeg')}\nresp = httpx.post('https://api.example.com/upload', files=files, timeout=30)\nresp.raise_for_status()\n```\nThe tuple form `(filename, fileobj, mimetype)` lets you control all three fields.")

    add("web",
        "Write a Flask route that requires a JSON content type.",
        "```python\nfrom flask import Flask, request, abort\n\napp = Flask(__name__)\n\n@app.post('/items')\ndef create():\n    if not request.is_json:\n        abort(415)\n    data = request.get_json()\n    return data, 201\n```\n415 'Unsupported Media Type' is the right code when the content type doesn't match.")

    add("web",
        "Use FastAPI's Depends with a database session.",
        "```python\nfrom contextlib import contextmanager\nfrom fastapi import FastAPI, Depends\n\napp = FastAPI()\n\nclass Session:\n    def close(self): pass\n\ndef get_db():\n    db = Session()\n    try:\n        yield db\n    finally:\n        db.close()\n\n@app.get('/users')\ndef list_users(db: Session = Depends(get_db)):\n    return {'count': 0}\n```\nThe `yield` pattern guarantees cleanup runs after the request finishes.")

    add("web",
        "Write a small URL shortener handler in Flask using an in-memory dict.",
        "```python\nimport secrets\nfrom flask import Flask, request, redirect, jsonify, abort\n\napp = Flask(__name__)\nstore: dict[str, str] = {}\n\n@app.post('/shorten')\ndef shorten():\n    url = request.json['url']\n    code = secrets.token_urlsafe(6)\n    store[code] = url\n    return jsonify(code=code)\n\n@app.get('/<code>')\ndef expand(code):\n    if code not in store:\n        abort(404)\n    return redirect(store[code])\n```\nReplace the dict with Redis or a DB for any real use.")

    add("web",
        "Write a function that uses requests' Session with persistent cookies.",
        "```python\nimport requests\n\ndef login_and_fetch(login_url: str, target: str, creds: dict) -> str:\n    s = requests.Session()\n    s.post(login_url, data=creds, timeout=10).raise_for_status()\n    r = s.get(target, timeout=10)\n    r.raise_for_status()\n    return r.text\n```\nThe Session keeps cookies across calls; closes connections cleanly on GC.")

    add("web",
        "Show how to define a Pydantic model with a field alias.",
        "```python\nfrom pydantic import BaseModel, Field\n\nclass User(BaseModel):\n    user_id: int = Field(alias='id')\n    name: str\n```\nUseful when the wire format and the Python field name differ. Pass `by_alias=True` to `model_dump` to round-trip.")

    add("web",
        "Write a FastAPI startup event that opens a connection pool.",
        "```python\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    app.state.pool = {'placeholder': True}\n    yield\n    app.state.pool.clear()\n\napp = FastAPI(lifespan=lifespan)\n```\nLifespan context replaced the older `@app.on_event('startup')` pattern.")

    add("web",
        "Write a small JSON-RPC style endpoint in Flask.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\nMETHODS = {\n    'add': lambda a, b: a + b,\n    'mul': lambda a, b: a * b,\n}\n\n@app.post('/rpc')\ndef rpc():\n    body = request.get_json()\n    fn = METHODS.get(body['method'])\n    if not fn:\n        return jsonify(error='unknown method'), 400\n    return jsonify(result=fn(*body.get('params', [])))\n```\nA dispatch table keeps it simple. Real JSON-RPC has more spec to follow (id, error codes).")

    add("web",
        "Write an async function that times out an httpx call.",
        "```python\nimport httpx\n\nasync def safe_get(url: str) -> str | None:\n    try:\n        async with httpx.AsyncClient(timeout=2.0) as c:\n            r = await c.get(url)\n            r.raise_for_status()\n            return r.text\n    except (httpx.TimeoutException, httpx.HTTPStatusError):\n        return None\n```\nReturning `None` on failure is fine for callers that already handle nullability.")

    add("web",
        "Show how to define query parameters with FastAPI.",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\n\n@app.get('/search')\ndef search(q: str = Query(..., min_length=2), page: int = 1):\n    return {'q': q, 'page': page}\n```\n`Query(..., min_length=2)` makes `q` required and validates length.")

    add("web",
        "Write a Flask app that uses Werkzeug's secure_filename for uploads.",
        "```python\nfrom flask import Flask, request\nfrom werkzeug.utils import secure_filename\nfrom pathlib import Path\n\napp = Flask(__name__)\nUPLOAD = Path('uploads'); UPLOAD.mkdir(exist_ok=True)\n\n@app.post('/upload')\ndef upload():\n    f = request.files['file']\n    name = secure_filename(f.filename)\n    f.save(UPLOAD / name)\n    return {'saved': name}\n```\n`secure_filename` strips path separators -- otherwise you risk directory traversal attacks.")

    add("web",
        "Write a small webhook receiver that verifies an HMAC signature.",
        "```python\nimport hmac, hashlib\nfrom flask import Flask, request, abort\n\napp = Flask(__name__)\nSECRET = b'topsecret'\n\n@app.post('/hook')\ndef hook():\n    sig = request.headers.get('x-signature', '')\n    expected = hmac.new(SECRET, request.data, hashlib.sha256).hexdigest()\n    if not hmac.compare_digest(sig, expected):\n        abort(401)\n    return {'ok': True}\n```\nUse `hmac.compare_digest` to avoid timing-attack leaks.")

    add("web",
        "Show how to make a GraphQL request with httpx.",
        "```python\nimport httpx\n\nQUERY = '{ user(id: 1) { name } }'\nresp = httpx.post(\n    'https://api.example.com/graphql',\n    json={'query': QUERY},\n    timeout=10,\n)\nresp.raise_for_status()\nprint(resp.json()['data'])\n```\nGraphQL servers expect a JSON body with `query` and optional `variables`/`operationName`.")

    add("web",
        "Write a FastAPI app that returns 204 No Content on delete.",
        "```python\nfrom fastapi import FastAPI, status\nfrom fastapi.responses import Response\n\napp = FastAPI()\n\n@app.delete('/items/{item_id}', status_code=status.HTTP_204_NO_CONTENT)\ndef delete(item_id: int) -> Response:\n    return Response(status_code=204)\n```\n204 explicitly means 'success, no body' -- don't return a JSON object with it.")

    add("web",
        "Write a function that paginates through a JSON API using next-page links.",
        "```python\nimport requests\n\ndef paginate(url: str, params=None):\n    while url:\n        r = requests.get(url, params=params, timeout=10)\n        r.raise_for_status()\n        data = r.json()\n        yield from data['items']\n        url = data.get('next')\n        params = None\n```\nGenerator pattern keeps memory low even for huge result sets.")

    add("web",
        "Write a Flask route protected by basic auth.",
        "```python\nfrom flask import Flask, request, abort\nimport secrets\n\napp = Flask(__name__)\n\n@app.before_request\ndef check_auth():\n    auth = request.authorization\n    if not auth or not (\n        secrets.compare_digest(auth.username, 'admin')\n        and secrets.compare_digest(auth.password, 'pw')\n    ):\n        abort(401)\n```\n`secrets.compare_digest` is constant-time. Don't roll your own equality check.")

    add("web",
        "Write a FastAPI endpoint that returns a redirect.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import RedirectResponse\n\napp = FastAPI()\n\n@app.get('/old')\ndef old():\n    return RedirectResponse(url='/new', status_code=308)\n```\n308 preserves the request method on redirect (unlike 302 which downgrades to GET).")

    add("web",
        "Show how to use httpx's transport for a unit test.",
        "```python\nimport httpx\n\ndef handler(request):\n    return httpx.Response(200, json={'ok': True})\n\ntransport = httpx.MockTransport(handler)\nclient = httpx.Client(transport=transport)\nassert client.get('https://x').json() == {'ok': True}\n```\n`MockTransport` lets you swap the network for a function -- great for tests without HTTP servers.")

    add("web",
        "Write a small Flask server-sent-events endpoint.",
        "```python\nimport time\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\ndef events():\n    for i in range(5):\n        yield f'data: {i}\\n\\n'\n        time.sleep(1)\n\n@app.get('/sse')\ndef sse():\n    return Response(events(), mimetype='text/event-stream')\n```\nSSE messages are 'data: ...\\n\\n' separated. Keep the connection open with a generator.")

    add("web",
        "Use FastAPI to upload and read a file.",
        "```python\nfrom fastapi import FastAPI, UploadFile, File\n\napp = FastAPI()\n\n@app.post('/upload')\nasync def upload(f: UploadFile = File(...)):\n    content = await f.read()\n    return {'name': f.filename, 'size': len(content)}\n```\n`UploadFile` is async and spools to disk for large uploads.")

    add("web",
        "Write a Flask error handler that converts exceptions to JSON.",
        "```python\nfrom flask import Flask, jsonify\nfrom werkzeug.exceptions import HTTPException\n\napp = Flask(__name__)\n\n@app.errorhandler(HTTPException)\ndef handle(e):\n    return jsonify(error=e.description), e.code\n\n@app.errorhandler(Exception)\ndef handle_unexpected(e):\n    return jsonify(error='internal'), 500\n```\nTwo handlers: one for HTTP errors (preserve status), one catch-all that hides details.")

    add("web",
        "Write a Pydantic model with a validator that normalizes an email.",
        "```python\nfrom pydantic import BaseModel, field_validator\n\nclass User(BaseModel):\n    email: str\n    @field_validator('email')\n    @classmethod\n    def lower(cls, v: str) -> str:\n        return v.strip().lower()\n```\nPydantic 2 uses `field_validator` (was `validator` in v1).")

    add("web",
        "Send POSTed JSON and parse the response with httpx.",
        "```python\nimport httpx\n\nresp = httpx.post('https://api.example.com/items', json={'name': 'a'}, timeout=10)\nresp.raise_for_status()\nitem = resp.json()\nprint(item['id'])\n```\nThe symmetry with requests is intentional; httpx supports both sync and async with the same API.")

    add("web",
        "Write a small async producer using asyncio.Queue.",
        "```python\nimport asyncio\n\nasync def producer(q: asyncio.Queue, n: int) -> None:\n    for i in range(n):\n        await q.put(i)\n    await q.put(None)\n\nasync def consumer(q: asyncio.Queue) -> None:\n    while True:\n        x = await q.get()\n        if x is None:\n            return\n        print(x)\n```\nSentinel `None` signals shutdown -- standard async producer/consumer pattern.")

    add("web",
        "Use FastAPI to return a streaming JSON-lines response.",
        "```python\nimport json\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\ndef gen():\n    for i in range(1000):\n        yield json.dumps({'i': i}) + '\\n'\n\n@app.get('/lines')\ndef lines():\n    return StreamingResponse(gen(), media_type='application/x-ndjson')\n```\nNDJSON / JSON-lines is friendlier than a giant array for clients that want to stream-parse.")

    add("web",
        "Write a Flask test client snippet that posts JSON and checks status.",
        "```python\nfrom myapp import app\n\ndef test_create():\n    client = app.test_client()\n    resp = client.post('/items', json={'name': 'x'})\n    assert resp.status_code == 201\n    assert resp.get_json() == {'id': 1, 'name': 'x'}\n```\n`test_client()` skips the network -- it dispatches WSGI calls in-process.")

    add("web",
        "Use httpx async client with a connection limit.",
        "```python\nimport httpx, asyncio\n\nasync def main(urls):\n    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)\n    async with httpx.AsyncClient(limits=limits, timeout=10) as c:\n        return await asyncio.gather(*(c.get(u) for u in urls))\n```\nLimits keep you from overwhelming the target host (or your own file descriptors).")

    add("web",
        "Write a FastAPI handler that returns a Pydantic response model.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\nclass UserOut(BaseModel):\n    id: int\n    name: str\n\napp = FastAPI()\n\n@app.get('/users/{user_id}', response_model=UserOut)\ndef get_user(user_id: int):\n    return {'id': user_id, 'name': 'alice', 'password_hash': 'should not leak'}\n```\n`response_model` strips fields not declared in the model -- a defense against accidental leaks.")

    add("web",
        "Write a Flask app that serves static files from a custom path.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__, static_folder='public', static_url_path='/static')\n```\nIn production, terminate static traffic at nginx/CDN; only fall through to Flask for dynamic routes.")

    add("web",
        "Use the `requests` library to check if a URL is reachable.",
        "```python\nimport requests\n\ndef is_reachable(url: str, timeout: float = 5.0) -> bool:\n    try:\n        return requests.head(url, timeout=timeout, allow_redirects=True).ok\n    except requests.RequestException:\n        return False\n```\nHEAD is cheaper than GET; `allow_redirects=True` follows 3xx.")

    add("web",
        "Show how to add gzip compression in FastAPI.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.gzip import GZipMiddleware\n\napp = FastAPI()\napp.add_middleware(GZipMiddleware, minimum_size=1024)\n```\nDon't compress tiny bodies -- the overhead wins.")

    add("web",
        "Write an async function that races multiple endpoints and returns the first response.",
        "```python\nimport asyncio, httpx\n\nasync def race(urls):\n    async with httpx.AsyncClient(timeout=5) as c:\n        tasks = [asyncio.create_task(c.get(u)) for u in urls]\n        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)\n        for t in pending: t.cancel()\n        return next(iter(done)).result()\n```\nCancel pending tasks to avoid leaks.")

    add("web",
        "Write a route in Flask that returns paginated JSON.",
        "```python\nfrom flask import Flask, request\n\napp = Flask(__name__)\nDATA = list(range(100))\n\n@app.get('/items')\ndef items():\n    page = int(request.args.get('page', 1))\n    size = int(request.args.get('size', 10))\n    start = (page - 1) * size\n    return {'page': page, 'size': size, 'items': DATA[start:start+size]}\n```\nIn production validate `page`/`size` ranges and clamp size to a sane max.")

    add("web",
        "Write a Pydantic model that validates a URL.",
        "```python\nfrom pydantic import BaseModel, HttpUrl\n\nclass Webhook(BaseModel):\n    url: HttpUrl\n```\n`HttpUrl` enforces scheme, host, and well-formedness automatically.")

    add("web",
        "Write a FastAPI app with a global exception handler.",
        "```python\nfrom fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\nclass DomainError(Exception):\n    def __init__(self, msg: str): self.msg = msg\n\n@app.exception_handler(DomainError)\nasync def handle_domain(request: Request, exc: DomainError):\n    return JSONResponse(status_code=400, content={'error': exc.msg})\n```\nDefine a project-specific exception type and register one handler -- much cleaner than scattering try/excepts.")

    add("web",
        "Show how to use httpx with mTLS client certificates.",
        "```python\nimport httpx\n\nclient = httpx.Client(\n    cert=('client.crt', 'client.key'),\n    verify='ca.crt',\n    timeout=10,\n)\nresp = client.get('https://api.example.com/secure')\nresp.raise_for_status()\n```\nThe `cert=` tuple holds your cert + key; `verify=` points at the CA bundle.")

    add("web",
        "Write a Flask form that accepts urlencoded input.",
        "```python\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n@app.post('/login')\ndef login():\n    user = request.form.get('user', '')\n    pw = request.form.get('password', '')\n    return {'user': user, 'pw_len': len(pw)}\n```\n`request.form` is for application/x-www-form-urlencoded and multipart; `request.json` is for JSON.")

    add("web",
        "Add a custom JSON encoder to Flask.",
        "```python\nimport json\nimport datetime as dt\nfrom flask import Flask\nfrom flask.json.provider import DefaultJSONProvider\n\nclass MyJSON(DefaultJSONProvider):\n    def default(self, o):\n        if isinstance(o, dt.datetime):\n            return o.isoformat()\n        return super().default(o)\n\napp = Flask(__name__)\napp.json = MyJSON(app)\n```\nFlask 2.2+ uses providers instead of subclassing JSONEncoder.")

    add("web",
        "Write a FastAPI route that returns a file download.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import FileResponse\n\napp = FastAPI()\n\n@app.get('/download')\ndef download():\n    return FileResponse('report.pdf', filename='report.pdf', media_type='application/pdf')\n```\nThe `filename` argument sets Content-Disposition so browsers prompt to save.")

    add("web",
        "Use httpx with a custom retry and exponential backoff via a transport wrapper.",
        "```python\nimport httpx, time\n\nclass RetryTransport(httpx.HTTPTransport):\n    def __init__(self, retries=3, **kw):\n        super().__init__(**kw)\n        self.retries = retries\n    def handle_request(self, request):\n        for i in range(self.retries):\n            try:\n                resp = super().handle_request(request)\n                if resp.status_code < 500:\n                    return resp\n            except httpx.RequestError:\n                if i == self.retries - 1: raise\n            time.sleep(0.2 * (2 ** i))\n        return resp\n\nclient = httpx.Client(transport=RetryTransport(retries=4))\n```\nWrapping the transport keeps the retry logic transparent to user code.")

    add("web",
        "Write a Flask route that returns YAML.",
        "```python\nimport yaml\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/cfg.yaml')\ndef cfg():\n    return Response(yaml.safe_dump({'k': 1}), mimetype='application/yaml')\n```\n`safe_dump` -- not `dump` -- to avoid arbitrary Python object serialization.")

    add("web",
        "Use FastAPI to read raw request body bytes.",
        "```python\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.post('/raw')\nasync def raw(request: Request):\n    body = await request.body()\n    return {'len': len(body)}\n```\nUse for webhooks where the signature is computed over the raw bytes.")

    add("web",
        "Show how to handle redirects manually with requests.",
        "```python\nimport requests\n\nresp = requests.get('http://x.test', allow_redirects=False, timeout=10)\nif resp.is_redirect:\n    print(resp.headers['Location'])\n```\nDisable auto-redirects when you need to inspect each hop.")

    add("web",
        "Write a Flask blueprint that handles 405 errors with JSON.",
        "```python\nfrom flask import Blueprint, jsonify\n\napi = Blueprint('api', __name__)\n\n@api.errorhandler(405)\ndef method_not_allowed(_):\n    return jsonify(error='method not allowed'), 405\n```\nBlueprint-scoped error handlers only fire for routes within the blueprint.")

    add("web",
        "Use FastAPI dependencies to inject a request-scoped logger.",
        "```python\nimport logging, uuid\nfrom fastapi import FastAPI, Depends\n\napp = FastAPI()\n\ndef logger():\n    return logging.LoggerAdapter(logging.getLogger('app'), {'rid': uuid.uuid4().hex})\n\n@app.get('/work')\ndef work(log = Depends(logger)):\n    log.info('handling')\n    return {'ok': True}\n```\nThe adapter attaches a per-request id you can include in your log format.")

    add("web",
        "Write an httpx call that uses HTTP/2.",
        "```python\nimport httpx\n\nwith httpx.Client(http2=True, timeout=10) as c:\n    resp = c.get('https://h2.example.com/')\n    print(resp.http_version)\n    print(resp.text[:100])\n```\nHTTP/2 needs the `h2` extra: `pip install 'httpx[http2]'`.")

    add("web",
        "Write a Flask route that uses g for per-request state.",
        "```python\nfrom flask import Flask, g, request\n\napp = Flask(__name__)\n\n@app.before_request\ndef setup():\n    g.user = request.headers.get('x-user', 'anon')\n\n@app.get('/me')\ndef me():\n    return {'user': g.user}\n```\n`g` is request-scoped; safer than module globals which would leak between requests.")

    add("web",
        "Show how to define a Pydantic v2 union with discriminator.",
        "```python\nfrom typing import Annotated, Literal, Union\nfrom pydantic import BaseModel, Field\n\nclass Cat(BaseModel):\n    type: Literal['cat']\n    purr: bool\n\nclass Dog(BaseModel):\n    type: Literal['dog']\n    bark: bool\n\nclass Owner(BaseModel):\n    pet: Annotated[Union[Cat, Dog], Field(discriminator='type')]\n```\nThe discriminator makes parsing fast and unambiguous.")

    add("web",
        "Write a FastAPI cookie-setting endpoint.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\n@app.post('/login')\ndef login():\n    resp = JSONResponse({'ok': True})\n    resp.set_cookie('session', 'abc', httponly=True, secure=True, samesite='lax')\n    return resp\n```\n`httponly`, `secure`, and `samesite` matter for session cookies.")

    add("web",
        "Write a Flask app that uses Limiter for rate limiting.",
        "```python\nfrom flask import Flask\nfrom flask_limiter import Limiter\nfrom flask_limiter.util import get_remote_address\n\napp = Flask(__name__)\nlimiter = Limiter(key_func=get_remote_address, app=app, default_limits=['100/minute'])\n\n@app.get('/api/data')\n@limiter.limit('10/minute')\ndef data():\n    return {'ok': True}\n```\n`get_remote_address` keys by client IP. Behind a proxy, configure trusted proxies first.")

    add("web",
        "Write an httpx client that follows redirects and shares cookies.",
        "```python\nimport httpx\n\nwith httpx.Client(follow_redirects=True, timeout=10) as c:\n    c.get('https://x.test/login')\n    r = c.get('https://x.test/me')\n    print(r.json())\n```\nThe Client retains cookies by default. Use `follow_redirects=True` if you need it (the default is False in httpx).")

    add("web",
        "Write a function that returns the OpenAPI schema from a FastAPI app.",
        "```python\nimport json\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/health')\ndef health(): return {'ok': True}\n\nschema = app.openapi()\nprint(json.dumps(schema, indent=2)[:200])\n```\n`app.openapi()` returns a dict; persist or diff it across deploys to track API changes.")

    add("web",
        "Use Flask-CORS to enable CORS for a specific blueprint.",
        "```python\nfrom flask import Blueprint, jsonify\nfrom flask_cors import CORS\n\napi = Blueprint('api', __name__)\nCORS(api, resources={r'/*': {'origins': 'https://app.example.com'}})\n\n@api.get('/me')\ndef me():\n    return jsonify(user='alice')\n```\nScope CORS to only the routes that need it.")

    add("web",
        "Write an httpx call that uses a SOCKS5 proxy.",
        "```python\nimport httpx\n\nclient = httpx.Client(proxy='socks5://127.0.0.1:1080', timeout=10)\nresp = client.get('https://example.com')\nprint(resp.status_code)\n```\nNeeds the `socksio` extra: `pip install 'httpx[socks]'`.")

    add("web",
        "Write a FastAPI dependency that paginates a query.",
        "```python\nfrom fastapi import FastAPI, Query, Depends\n\napp = FastAPI()\n\ndef pagination(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):\n    return {'offset': (page - 1) * size, 'limit': size}\n\n@app.get('/items')\ndef items(p: dict = Depends(pagination)):\n    return p\n```\nReusable across endpoints. The validation lives in one place.")

    add("web",
        "Write a small Flask oauth2 callback handler.",
        "```python\nfrom flask import Flask, request\n\napp = Flask(__name__)\n\n@app.get('/oauth/callback')\ndef cb():\n    code = request.args.get('code')\n    state = request.args.get('state')\n    if not code:\n        return {'error': 'missing code'}, 400\n    return {'code': code, 'state': state}\n```\nIn real apps validate `state` against the value you stored before the redirect.")

    add("web",
        "Use httpx to send a request with custom event hooks.",
        "```python\nimport httpx\n\ndef log_request(request):\n    print('->', request.method, request.url)\n\ndef log_response(resp):\n    print('<-', resp.status_code)\n\nclient = httpx.Client(event_hooks={'request': [log_request], 'response': [log_response]})\nclient.get('https://example.com')\n```\nGreat for observability; keep hooks fast since they run inline.")

    add("web",
        "Write a FastAPI app that uses orjson via custom response class.",
        "```python\nimport orjson\nfrom fastapi import FastAPI\nfrom fastapi.responses import ORJSONResponse\n\napp = FastAPI(default_response_class=ORJSONResponse)\n\n@app.get('/')\ndef root():\n    return {'utc': '2026-01-01T00:00:00Z'}\n```\norjson handles datetime, numpy, and dataclasses natively without custom encoders.")

    add("web",
        "Write a Flask route that proxies a request to another service.",
        "```python\nimport requests\nfrom flask import Flask, request, Response\n\napp = Flask(__name__)\nUPSTREAM = 'https://api.upstream.test'\n\n@app.route('/proxy/<path:path>', methods=['GET', 'POST'])\ndef proxy(path):\n    r = requests.request(\n        request.method,\n        f'{UPSTREAM}/{path}',\n        headers={k: v for k, v in request.headers if k.lower() != 'host'},\n        data=request.get_data(),\n        timeout=10,\n    )\n    return Response(r.content, status=r.status_code, headers=dict(r.headers))\n```\nStrip the Host header so the upstream sees its own. For production proxying, use nginx -- it's a much harder problem than it looks.")

    add("web",
        "Show how to configure Pydantic Settings from env vars.",
        "```python\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    db_url: str\n    debug: bool = False\n    class Config:\n        env_file = '.env'\n\nsettings = Settings()\n```\nReads from env or .env file. `pydantic_settings` is the v2 successor to `pydantic.BaseSettings`.")

    add("web",
        "Write an aiohttp client that uses connection pooling.",
        "```python\nimport asyncio, aiohttp\n\nasync def main(urls):\n    conn = aiohttp.TCPConnector(limit=20)\n    async with aiohttp.ClientSession(connector=conn) as s:\n        return await asyncio.gather(*(s.get(u) for u in urls))\n```\n`limit=20` caps simultaneous connections per host.")

    add("web",
        "Use FastAPI's TestClient to verify a status code.",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\ndef test_root():\n    client = TestClient(app)\n    r = client.get('/')\n    assert r.status_code == 200\n```\n`TestClient` is built on httpx -- same interface as a real client.")

    add("web",
        "Write a Flask route that returns a 422 with structured field errors.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.post('/users')\ndef create():\n    data = request.get_json() or {}\n    errors = {}\n    if not data.get('name'): errors['name'] = 'required'\n    if not data.get('email'): errors['email'] = 'required'\n    if errors:\n        return jsonify(errors=errors), 422\n    return jsonify(ok=True), 201\n```\n422 communicates the request was understood but the entity was invalid.")

    add("web",
        "Show how to use httpx async with backoff library.",
        "```python\nimport httpx, backoff\n\n@backoff.on_exception(backoff.expo, httpx.HTTPError, max_tries=5)\nasync def fetch(url: str) -> str:\n    async with httpx.AsyncClient(timeout=10) as c:\n        r = await c.get(url)\n        r.raise_for_status()\n        return r.text\n```\nThe decorator handles retry + jittered backoff; keep your function focused on the happy path.")

    # ---- 5-msg follow-ups ----
    add("web",
        "Show how to validate a JWT in a FastAPI dependency.",
        "```python\nfrom fastapi import FastAPI, HTTPException, Depends\nfrom fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\nimport jwt\n\nSECRET = 'changeme'\nbearer = HTTPBearer()\napp = FastAPI()\n\ndef current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):\n    try:\n        return jwt.decode(creds.credentials, SECRET, algorithms=['HS256'])\n    except jwt.PyJWTError:\n        raise HTTPException(status_code=401, detail='invalid token')\n\n@app.get('/me')\ndef me(user: dict = Depends(current_user)):\n    return user\n```\n`HTTPBearer` extracts the token from Authorization headers; the dependency returns the decoded claims.",
        "Why use HS256 vs RS256?",
        "HS256 (HMAC) uses a single shared secret -- both signer and verifier need it. Fine for monolithic apps where the same service issues and validates. RS256 (RSA) uses a private/public key pair: only the issuer holds the private key, and any number of verifiers can hold the public key. Use RS256 for distributed systems, third-party verifiers, or any case where key rotation crosses service boundaries.")

    add("web",
        "Write a FastAPI app with database transactions per request.",
        "```python\nfrom fastapi import FastAPI, Depends\nfrom sqlalchemy.orm import Session\nfrom sqlalchemy import create_engine\n\nengine = create_engine('sqlite:///app.db')\napp = FastAPI()\n\ndef get_db():\n    with Session(engine) as session:\n        with session.begin():\n            yield session\n\n@app.post('/items')\ndef create(name: str, db: Session = Depends(get_db)):\n    return {'name': name}\n```\nThe nested `with session.begin()` opens a transaction that commits on success or rolls back on exception.",
        "What happens if I forget the begin() block?",
        "SQLAlchemy 2.0 sessions are autocommit-off by default, so without `begin()` (or `session.commit()`) writes never get flushed to the database -- you'd lose data silently. The context-manager pattern is the recommended idiom because it guarantees commit-on-success / rollback-on-exception with no manual try/except.")

    add("web",
        "Write an aiohttp server with graceful shutdown.",
        "```python\nimport asyncio\nfrom aiohttp import web\n\nasync def hello(request):\n    return web.Response(text='hi')\n\nasync def on_shutdown(app):\n    print('cleanup')\n\napp = web.Application()\napp.add_routes([web.get('/', hello)])\napp.on_shutdown.append(on_shutdown)\n\nweb.run_app(app, port=8080)\n```\n`on_shutdown` runs when the process catches SIGINT/SIGTERM.",
        "How do I drain in-flight requests during shutdown?",
        "aiohttp's `run_app` does this automatically -- it stops accepting new connections, waits for in-flight handlers to finish (up to `shutdown_timeout`, default 60s), then closes the loop. You can lengthen the timeout via `web.run_app(app, shutdown_timeout=120)`. For load-balanced deployments, also unregister from the LB before sending SIGTERM so traffic stops arriving.")

    add("web",
        "Write a Flask app with structured logging using JSON output.",
        "```python\nimport json, logging, sys\nfrom flask import Flask, request, g\n\nclass JsonFormatter(logging.Formatter):\n    def format(self, record):\n        return json.dumps({\n            'level': record.levelname,\n            'msg': record.getMessage(),\n            'logger': record.name,\n        })\n\nh = logging.StreamHandler(sys.stdout); h.setFormatter(JsonFormatter())\nlogging.getLogger().addHandler(h); logging.getLogger().setLevel(logging.INFO)\n\napp = Flask(__name__)\n@app.before_request\ndef log_req():\n    logging.info(f'{request.method} {request.path}')\n```\nJSON logs are easier for log aggregators (Loki, CloudWatch, Datadog) to parse.",
        "Should I include the trace id in the logs too?",
        "Yes -- it's the single most useful field for correlating logs with traces. Generate or extract a trace id (W3C `traceparent` header is the standard) per request, store it on `g`, and add it to the JSON output. Once you wire it up, debugging distributed problems becomes dramatically easier.")

    add("web",
        "Write a FastAPI middleware that adds request IDs.",
        "```python\nimport uuid\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def add_rid(request: Request, call_next):\n    rid = request.headers.get('x-request-id') or uuid.uuid4().hex\n    request.state.rid = rid\n    response = await call_next(request)\n    response.headers['x-request-id'] = rid\n    return response\n```\nIf upstream already supplies an id, keep it; otherwise generate one.",
        "What about propagating that id to outbound requests?",
        "Pull `request.state.rid` in your handlers and add it to outbound calls' headers (e.g. `headers={'x-request-id': rid}`). For a cleaner pattern, use a context variable (`contextvars.ContextVar`) populated in the middleware and read inside an httpx auth/transport hook -- that way handlers don't have to remember.")

    add("web",
        "Write a Flask handler that uses Pydantic for request body validation.",
        "```python\nfrom flask import Flask, request, jsonify\nfrom pydantic import BaseModel, ValidationError\n\nclass CreateItem(BaseModel):\n    name: str\n    qty: int\n\napp = Flask(__name__)\n\n@app.post('/items')\ndef create():\n    try:\n        body = CreateItem.model_validate(request.get_json())\n    except ValidationError as e:\n        return jsonify(errors=e.errors()), 422\n    return jsonify(body.model_dump()), 201\n```\nFlask doesn't have built-in Pydantic integration, so we wire it manually.",
        "Should I switch to FastAPI for this?",
        "Probably -- FastAPI gives you Pydantic validation, OpenAPI docs, dependency injection, and async support all together. The migration cost is moderate if your Flask app is mostly REST. Stay on Flask if you have heavy non-API surface (server-rendered templates, Flask-Login, etc.) or a large existing extension footprint.")

    add("web",
        "Show how to use httpx with mTLS and a custom CA.",
        "```python\nimport httpx\n\nclient = httpx.Client(\n    cert=('client.pem', 'client.key'),\n    verify='ca.pem',\n    timeout=10,\n)\nr = client.get('https://internal.example/api')\nr.raise_for_status()\n```\nClient cert authenticates you to the server; the CA bundle authenticates the server to you.",
        "What's the failure mode if the cert is expired?",
        "httpx (via httpcore -> ssl) will raise an `httpx.ConnectError` wrapping an `ssl.SSLError` like `certificate verify failed: certificate has expired`. Wrap calls in try/except and surface the underlying SSL error message in your logs -- generic 'connection failed' is impossible to debug otherwise.")

    add("web",
        "Write a Flask route that returns ETag headers.",
        "```python\nimport hashlib\nfrom flask import Flask, request, Response\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    body = b'{\"k\": 1}'\n    etag = hashlib.md5(body).hexdigest()\n    if request.headers.get('If-None-Match') == etag:\n        return '', 304\n    resp = Response(body, mimetype='application/json')\n    resp.headers['ETag'] = etag\n    return resp\n```\nReturning 304 Not Modified saves bandwidth when the client has the same version.",
        "Why MD5 here and not SHA-256?",
        "ETag is purely a cache-coherence value -- it doesn't need cryptographic strength. MD5 is shorter, faster, and collisions are statistically negligible across a single resource's history. Many CDNs use shorter hashes still (xxhash). Avoid MD5 only when an attacker controls the input and could craft a collision against a security boundary.")

    add("web",
        "Write a FastAPI app exposing Prometheus metrics.",
        "```python\nfrom fastapi import FastAPI\nfrom prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST\nfrom fastapi.responses import Response\n\nrequests_total = Counter('http_requests_total', 'Total requests', ['method', 'path'])\napp = FastAPI()\n\n@app.middleware('http')\nasync def count(request, call_next):\n    requests_total.labels(request.method, request.url.path).inc()\n    return await call_next(request)\n\n@app.get('/metrics')\ndef metrics():\n    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)\n```\nFor full coverage, also track latencies via Histogram and active in-flight via Gauge.",
        "How do I avoid path-cardinality blowing up?",
        "Don't label by full path -- label by the route template. Use FastAPI's matched route: `request.scope.get('route').path` (when present) gives `/items/{item_id}` instead of `/items/12345`. High-cardinality labels turn Prometheus into a memory hog and slow queries dramatically.")

    add("web",
        "Write a FastAPI app that integrates with SQLModel.",
        "```python\nfrom fastapi import FastAPI\nfrom sqlmodel import SQLModel, Field, Session, create_engine, select\n\nclass Hero(SQLModel, table=True):\n    id: int | None = Field(default=None, primary_key=True)\n    name: str\n\nengine = create_engine('sqlite:///heroes.db')\nSQLModel.metadata.create_all(engine)\napp = FastAPI()\n\n@app.post('/heroes')\ndef create(hero: Hero):\n    with Session(engine) as s:\n        s.add(hero); s.commit(); s.refresh(hero)\n        return hero\n\n@app.get('/heroes')\ndef list_heroes():\n    with Session(engine) as s:\n        return s.exec(select(Hero)).all()\n```\nSQLModel unifies SQLAlchemy and Pydantic so the DB model and request model are the same class.",
        "Is there a downside to using the same model for both?",
        "Yes -- you eventually need to diverge: hide internal columns (created_at, owner_id) from the API, or accept different shape on input vs output (no id on POST, id required on GET). The fix is separate models with shared mixins, e.g. `class HeroBase(SQLModel)` (shared fields), `class Hero(HeroBase, table=True)`, `class HeroPublic(HeroBase)`, `class HeroCreate(HeroBase)`.")

    add("web",
        "Write a small Flask app that uses Flask-Caching with Redis.",
        "```python\nfrom flask import Flask\nfrom flask_caching import Cache\n\napp = Flask(__name__)\ncache = Cache(app, config={'CACHE_TYPE': 'RedisCache', 'CACHE_REDIS_URL': 'redis://localhost:6379/0'})\n\n@app.get('/expensive/<int:n>')\n@cache.cached(timeout=60)\ndef expensive(n):\n    return {'result': sum(range(n))}\n```\nThe path is included in the cache key automatically.",
        "How do I invalidate the cache when data changes?",
        "Use `cache.delete('view//expensive/42')` for a specific entry, or pattern-key the result with `make_cache_key` and call `cache.delete_memoized(expensive, 42)`. For broader invalidation, use a versioned key prefix that you bump on writes -- old entries simply expire instead of being deleted.")

    add("web",
        "Write a FastAPI app with rate limiting via SlowAPI.",
        "```python\nfrom fastapi import FastAPI, Request\nfrom slowapi import Limiter\nfrom slowapi.util import get_remote_address\n\nlimiter = Limiter(key_func=get_remote_address)\napp = FastAPI()\napp.state.limiter = limiter\n\n@app.get('/api')\n@limiter.limit('5/minute')\ndef api(request: Request):\n    return {'ok': True}\n```\nSlowAPI requires the `request: Request` parameter so it can pull the IP.",
        "How do I rate-limit per user instead of per IP?",
        "Replace `get_remote_address` with a custom key function that returns the user id from a token: `lambda req: req.state.user.id if hasattr(req.state, 'user') else get_remote_address(req)`. Make sure auth runs before rate limiting -- otherwise unauthenticated requests get keyed by IP and authenticated by id, which can be exploited.")

    add("web",
        "Write an aiohttp client that posts JSON with retry on 5xx.",
        "```python\nimport asyncio, aiohttp\n\nasync def post_with_retry(url, payload, attempts=3):\n    backoff = 0.5\n    for i in range(attempts):\n        try:\n            async with aiohttp.ClientSession() as s:\n                async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:\n                    if r.status < 500:\n                        return await r.json()\n        except aiohttp.ClientError:\n            if i == attempts - 1:\n                raise\n        await asyncio.sleep(backoff)\n        backoff *= 2\n    raise RuntimeError('exhausted retries')\n```\nClassic exponential backoff with a per-call ClientSession; pull the session out if you batch many calls.",
        "Should I really retry POSTs?",
        "Only if the request is idempotent (or the server provides an idempotency key). Retrying a POST that creates a row twice produces duplicate rows. The fix: include `Idempotency-Key: <uuid>` and have the server dedupe. For non-idempotent operations, bubble the error up instead of retrying blindly.")

    add("web",
        "Show how to add OpenTelemetry tracing to a FastAPI app.",
        "```python\nfrom fastapi import FastAPI\nfrom opentelemetry import trace\nfrom opentelemetry.instrumentation.fastapi import FastAPIInstrumentor\nfrom opentelemetry.sdk.trace import TracerProvider\nfrom opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter\n\ntrace.set_tracer_provider(TracerProvider())\ntrace.get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))\n\napp = FastAPI()\nFastAPIInstrumentor.instrument_app(app)\n\n@app.get('/work')\ndef work():\n    return {'ok': True}\n```\nThe instrumentor wraps every route in a span automatically.",
        "How do I export to a real tracing backend?",
        "Replace `ConsoleSpanExporter` with an OTLP exporter: `from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter; OTLPSpanExporter(endpoint='https://otel-collector:4317')`. Most modern backends (Tempo, Honeycomb, Datadog) accept OTLP directly. Use `BatchSpanProcessor` in production to amortize export cost.")

    add("web",
        "Write a Flask route that uses Marshmallow for schema validation.",
        "```python\nfrom flask import Flask, request, jsonify\nfrom marshmallow import Schema, fields, ValidationError\n\nclass ItemSchema(Schema):\n    name = fields.Str(required=True)\n    qty = fields.Int(required=True)\n\napp = Flask(__name__)\nschema = ItemSchema()\n\n@app.post('/items')\ndef create():\n    try:\n        data = schema.load(request.get_json() or {})\n    except ValidationError as e:\n        return jsonify(errors=e.messages), 422\n    return jsonify(data), 201\n```\nMarshmallow predates Pydantic; both work, Pydantic v2 is faster for raw validation.",
        "Pydantic vs Marshmallow -- which would you pick today?",
        "Pydantic v2 -- it's faster (Rust core), better-typed, and now has the broader ecosystem (FastAPI, OpenAI SDKs, settings management). Marshmallow still has nice features like fully customizable serialization paths and tightly integrated with SQLAlchemy via marshmallow-sqlalchemy. For new projects, Pydantic is the default; keep Marshmallow if you have heavy investment in its plugins.")

    add("web",
        "Write an httpx async streaming download with progress.",
        "```python\nimport httpx, asyncio\nfrom pathlib import Path\n\nasync def download(url: str, dest: Path):\n    async with httpx.AsyncClient(timeout=None) as c:\n        async with c.stream('GET', url) as r:\n            r.raise_for_status()\n            total = int(r.headers.get('content-length', 0))\n            done = 0\n            with dest.open('wb') as f:\n                async for chunk in r.aiter_bytes(chunk_size=64 * 1024):\n                    f.write(chunk); done += len(chunk)\n                    if total:\n                        print(f'\\r{done/total:.1%}', end='')\n```\nThe progress assumes the server returned Content-Length; otherwise show bytes transferred.",
        "What if the connection drops halfway?",
        "Use HTTP Range to resume: track `done` to disk, then on retry add `Range: bytes=<done>-` and append to the file instead of overwriting. Verify the server supports range with `Accept-Ranges: bytes` in the initial HEAD response. Many CDNs do; some upstream services don't, in which case you must restart from zero.")

    add("web",
        "Write a Flask app with Sentry error reporting.",
        "```python\nimport sentry_sdk\nfrom sentry_sdk.integrations.flask import FlaskIntegration\nfrom flask import Flask\n\nsentry_sdk.init(\n    dsn='https://example@sentry.io/123',\n    integrations=[FlaskIntegration()],\n    traces_sample_rate=0.1,\n)\n\napp = Flask(__name__)\n\n@app.get('/')\ndef root():\n    raise RuntimeError('boom')\n```\nThe integration captures unhandled exceptions automatically.",
        "Should I sample at 0.1 or capture everything?",
        "10% transaction sampling is a common starting point: enough data to spot regressions, not so much that you blow the Sentry quota. Always capture errors at 100% (`sample_rate=1.0`); only traces are sampled. If your traffic is tiny or you have a tight feedback loop, bump to 1.0 traces; if you serve millions of requests/min, dial down further or use head-based sampling tied to error states.")

    add("web",
        "Write a small async pub/sub using asyncio queues.",
        "```python\nimport asyncio\nfrom collections import defaultdict\n\nclass Bus:\n    def __init__(self) -> None:\n        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)\n    def subscribe(self, topic: str) -> asyncio.Queue:\n        q: asyncio.Queue = asyncio.Queue()\n        self._subs[topic].append(q)\n        return q\n    async def publish(self, topic: str, msg) -> None:\n        for q in self._subs[topic]:\n            await q.put(msg)\n```\nIn-process only -- for cross-process use Redis pub/sub or NATS.",
        "How do I handle slow subscribers without blocking the publisher?",
        "Either give each queue a `maxsize` and use `q.put_nowait(msg)` with QueueFull handled (drop or DLQ), or wrap each subscriber in a task and let the publisher fire-and-forget. Backpressure choice depends on semantics: lossy realtime systems (metrics) drop; reliable systems (orders) must block or persist.")

    add("web",
        "Use FastAPI to expose an async endpoint that streams Server-Sent Events.",
        "```python\nimport asyncio\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\nasync def gen():\n    for i in range(5):\n        yield f'data: tick {i}\\n\\n'\n        await asyncio.sleep(1)\n\n@app.get('/stream')\nasync def stream():\n    return StreamingResponse(gen(), media_type='text/event-stream')\n```\nThe blank line after `data:` separates events. Use `event: name\\n` to type messages.",
        "How would I detect when the client disconnects?",
        "Pass `request: Request` to the endpoint, then in the generator check `await request.is_disconnected()` between yields and break out if True. This is important for long-running streams -- without it, you keep generating events into a closed socket and waste CPU.")
