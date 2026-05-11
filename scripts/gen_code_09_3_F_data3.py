# SPDX-License-Identifier: MIT
"""Web part 2 (50) + Data science part 1 (50). Wave 6 part 3."""


def register_all(add):
    # ---- 50 more web entries ----
    add("web",
        "Write a FastAPI route that validates email format with EmailStr.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, EmailStr\n\nclass Subscribe(BaseModel):\n    email: EmailStr\n\napp = FastAPI()\n\n@app.post('/subscribe')\ndef subscribe(body: Subscribe):\n    return {'email': body.email}\n```\n`EmailStr` requires the `email-validator` package. It catches common typos that bare regex won't.")

    add("web",
        "Implement a Flask route that returns YAML.",
        "```python\nimport yaml\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/config')\ndef config():\n    body = yaml.safe_dump({'env': 'prod', 'workers': 4})\n    return Response(body, mimetype='application/yaml')\n```\n`safe_dump` is fine for trusted output. `yaml.dump` defaults are safe but `safe_dump` is explicit and prevents accidental dumps of arbitrary Python objects.")

    add("web",
        "Write a FastAPI route that times out an upstream call.",
        "```python\nimport asyncio, httpx\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.get('/upstream')\nasync def upstream():\n    try:\n        async with httpx.AsyncClient(timeout=2.0) as c:\n            r = await c.get('https://example.com/slow')\n        return r.json()\n    except (httpx.TimeoutException, asyncio.TimeoutError):\n        raise HTTPException(504, 'upstream timeout')\n```\nMap upstream timeouts to 504 so callers can distinguish 'we failed' from 'upstream failed'.")

    add("web",
        "Implement Flask logout endpoint.",
        "```python\nfrom flask import Flask, session\n\napp = Flask(__name__)\n\n@app.post('/logout')\ndef logout():\n    session.clear()\n    return {'ok': True}\n```\n`session.clear()` removes all keys; the cookie is updated on the response automatically.")

    add("web",
        "Write a FastAPI app with HTTPS redirect middleware.",
        "```python\nfrom fastapi import FastAPI\nfrom starlette.middleware.httpsredirect import HTTPSRedirectMiddleware\n\napp = FastAPI()\napp.add_middleware(HTTPSRedirectMiddleware)\n```\nBehind a load balancer that terminates TLS, you generally want this **off** -- the LB already handles it. Use only if uvicorn directly faces the internet.")

    add("web",
        "Implement a FastAPI middleware adding security headers.",
        "```python\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def security_headers(request: Request, call_next):\n    resp = await call_next(request)\n    resp.headers['X-Content-Type-Options'] = 'nosniff'\n    resp.headers['X-Frame-Options'] = 'DENY'\n    resp.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'\n    return resp\n```\nFor CSP, build the policy string explicitly per route family -- a one-size-fits-all CSP usually breaks something.")

    add("web",
        "Write a FastAPI request body that accepts a list.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\nclass Item(BaseModel):\n    name: str\n\napp = FastAPI()\n\n@app.post('/items/bulk')\ndef bulk(items: list[Item]):\n    return {'count': len(items)}\n```\nFastAPI handles `list[Model]` in the body the same way as a single model -- declare the type and you're done.")

    add("web",
        "Implement a Flask endpoint that returns ETags for caching.",
        "```python\nimport hashlib\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    body = jsonify({'items': [1, 2, 3]})\n    etag = hashlib.md5(body.data).hexdigest()\n    if request.headers.get('If-None-Match') == etag:\n        return '', 304\n    body.headers['ETag'] = etag\n    return body\n```\nReturning 304 saves bandwidth and is mandatory if you set `Cache-Control: must-revalidate`.")

    add("web",
        "Write a FastAPI app with lifespan that initializes httpx client.",
        "```python\nfrom contextlib import asynccontextmanager\nimport httpx\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    app.state.http = httpx.AsyncClient(timeout=10)\n    yield\n    await app.state.http.aclose()\n\napp = FastAPI(lifespan=lifespan)\n\n@app.get('/proxy')\nasync def proxy():\n    r = await app.state.http.get('https://example.com')\n    return {'status': r.status_code}\n```\nReusing one client across requests gives connection pooling and HTTP/2 multiplexing.")

    add("web",
        "Implement a FastAPI route that accepts query list parameter.",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\n\n@app.get('/search')\ndef search(tags: list[str] = Query(default=[])):\n    return {'tags': tags}\n```\nClient passes `?tags=a&tags=b`. FastAPI collects multiple values into a list.")

    add("web",
        "Write a Flask endpoint with X-Total-Count header.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\nDB = list(range(123))\n\n@app.get('/items')\ndef items():\n    resp = jsonify(DB[:20])\n    resp.headers['X-Total-Count'] = str(len(DB))\n    return resp\n```\nCommon convention for paginated lists; some clients (admin uis, old REST tools) rely on it.")

    add("web",
        "Implement FastAPI request that returns 201 with Location header.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\n@app.post('/items', status_code=201)\ndef create(item: dict):\n    new_id = 42\n    return JSONResponse(\n        content={'id': new_id, **item},\n        status_code=201,\n        headers={'Location': f'/items/{new_id}'},\n    )\n```\nReturning 201 + Location is REST-canonical; clients that follow Location can fetch the canonical representation.")

    add("web",
        "Write a Flask route handling JSON parse errors gracefully.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.post('/echo')\ndef echo():\n    data = request.get_json(silent=True)\n    if data is None:\n        return jsonify(error='invalid JSON'), 400\n    return jsonify(data)\n```\n`silent=True` returns None instead of raising; we then produce a friendly 400 instead of Flask's default 500.")

    add("web",
        "Implement FastAPI streaming generator response.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\ndef big_csv():\n    yield 'id,name\\n'\n    for i in range(1_000_000):\n        yield f'{i},user{i}\\n'\n\n@app.get('/users.csv')\ndef users():\n    return StreamingResponse(big_csv(), media_type='text/csv')\n```\nStreaming keeps memory flat regardless of dataset size. Add `Content-Disposition` to trigger a download dialog.")

    add("web",
        "Write FastAPI dependency overrides for testing.",
        "```python\nfrom fastapi import Depends, FastAPI\nfrom fastapi.testclient import TestClient\n\ndef get_db():\n    return 'real'\n\napp = FastAPI()\n\n@app.get('/who')\ndef who(db: str = Depends(get_db)):\n    return {'db': db}\n\ndef test_who():\n    app.dependency_overrides[get_db] = lambda: 'fake'\n    try:\n        c = TestClient(app)\n        assert c.get('/who').json() == {'db': 'fake'}\n    finally:\n        app.dependency_overrides.clear()\n```\nAlways clear overrides in `finally` -- otherwise test ordering matters.")

    add("web",
        "Implement an httpx call with bearer auth.",
        "```python\nimport httpx\n\nclass BearerAuth(httpx.Auth):\n    def __init__(self, token: str):\n        self.token = token\n    def auth_flow(self, request):\n        request.headers['Authorization'] = f'Bearer {self.token}'\n        yield request\n\nwith httpx.Client(auth=BearerAuth('xxx')) as c:\n    r = c.get('https://api.example.com/me')\n```\nA reusable auth object is cleaner than threading `headers=` through every call.")

    add("web",
        "Write a FastAPI route returning 422 with custom validator.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, field_validator\n\nclass Order(BaseModel):\n    quantity: int\n\n    @field_validator('quantity')\n    @classmethod\n    def positive(cls, v: int) -> int:\n        if v <= 0:\n            raise ValueError('must be positive')\n        return v\n\napp = FastAPI()\n\n@app.post('/orders')\ndef create(order: Order):\n    return order\n```\nField validators raising `ValueError` produce 422 responses with the message included.")

    add("web",
        "Implement a Flask blueprint with URL prefix.",
        "```python\nfrom flask import Blueprint, jsonify, Flask\n\napi = Blueprint('api', __name__, url_prefix='/api/v1')\n\n@api.get('/ping')\ndef ping():\n    return jsonify(pong=True)\n\napp = Flask(__name__)\napp.register_blueprint(api)\n```\nFinal URL is `/api/v1/ping`. Easy way to version APIs without rewriting every route.")

    add("web",
        "Write a FastAPI endpoint that accepts and returns binary data.",
        "```python\nfrom fastapi import FastAPI, Body\nfrom fastapi.responses import Response\n\napp = FastAPI()\n\n@app.post('/echo-bytes')\ndef echo(data: bytes = Body(...)):\n    return Response(content=data, media_type='application/octet-stream')\n```\nFor true streaming, use `request: Request` and iterate `request.stream()` to avoid loading the whole body.")

    add("web",
        "Implement Flask login that hashes passwords with bcrypt.",
        "```python\nimport bcrypt\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nUSERS = {'alice': bcrypt.hashpw(b'secret', bcrypt.gensalt())}\n\n@app.post('/login')\ndef login():\n    data = request.get_json()\n    user = data['user']; pw = data['password'].encode()\n    if user in USERS and bcrypt.checkpw(pw, USERS[user]):\n        return jsonify(ok=True)\n    return jsonify(error='bad creds'), 401\n```\n`checkpw` is constant-time. Never compare hashes with `==`.")

    add("web",
        "Write a FastAPI route that handles OPTIONS preflight manually.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://app.example.com'],\n    allow_methods=['GET', 'POST'],\n    allow_headers=['Authorization', 'Content-Type'],\n    max_age=600,\n)\n```\nThe middleware handles OPTIONS automatically. `max_age=600` caches the preflight response for 10 minutes.")

    add("web",
        "Implement an httpx call with custom transport for testing.",
        "```python\nimport httpx\n\ndef handler(request):\n    return httpx.Response(200, json={'ok': True})\n\ntransport = httpx.MockTransport(handler)\nwith httpx.Client(transport=transport) as c:\n    r = c.get('http://anything')\n    assert r.json() == {'ok': True}\n```\n`MockTransport` is the official way to test code that uses httpx without spinning up a server.")

    add("web",
        "Write a FastAPI request that uses HTTPX response_model_exclude_none.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\nclass User(BaseModel):\n    id: int\n    nickname: str | None = None\n\napp = FastAPI()\n\n@app.get('/u', response_model=User, response_model_exclude_none=True)\ndef u():\n    return User(id=1)\n```\n`exclude_none` strips null fields from the response -- cleaner JSON, smaller payloads.")

    add("web",
        "Implement Flask request size limit.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\napp.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # 1MB\n```\nFlask returns 413 for oversized requests automatically. Always set this for public endpoints -- otherwise a single big body can OOM your worker.")

    add("web",
        "Write FastAPI endpoint with explicit response status codes.",
        "```python\nfrom fastapi import FastAPI, status\n\napp = FastAPI()\n\n@app.delete('/items/{id}', status_code=status.HTTP_204_NO_CONTENT)\ndef delete(id: int):\n    return None\n```\n204 must have an empty body. Returning `None` from a 204-coded endpoint is correct.")

    add("web",
        "Implement Flask file upload with validation.",
        "```python\nfrom pathlib import Path\nfrom flask import Flask, request, jsonify\n\nALLOWED = {'.png', '.jpg', '.jpeg'}\napp = Flask(__name__)\n\n@app.post('/upload')\ndef upload():\n    f = request.files.get('file')\n    if not f:\n        return jsonify(error='no file'), 400\n    if Path(f.filename).suffix.lower() not in ALLOWED:\n        return jsonify(error='bad type'), 400\n    f.save(f'/tmp/{f.filename}')\n    return jsonify(ok=True)\n```\nFor production, also sniff the file's actual magic bytes -- extension can be spoofed.")

    add("web",
        "Write a FastAPI route that returns content based on Accept header.",
        "```python\nfrom fastapi import FastAPI, Header\nfrom fastapi.responses import JSONResponse, PlainTextResponse\n\napp = FastAPI()\n\n@app.get('/data')\ndef data(accept: str = Header(default='application/json')):\n    if 'text/plain' in accept:\n        return PlainTextResponse('hello')\n    return JSONResponse({'msg': 'hello'})\n```\nClassic content negotiation. Most APIs settle on JSON-only and skip this.")

    add("web",
        "Implement Flask after_request hook that adds CORS headers.",
        "```python\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.after_request\ndef add_cors(resp):\n    resp.headers['Access-Control-Allow-Origin'] = 'https://app.example.com'\n    resp.headers['Access-Control-Allow-Credentials'] = 'true'\n    return resp\n```\nManual CORS is fragile -- prefer `flask-cors`. Keep this only for tightly-scoped legacy use cases.")

    add("web",
        "Write a FastAPI app that mounts a static files directory.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.staticfiles import StaticFiles\n\napp = FastAPI()\napp.mount('/static', StaticFiles(directory='static'), name='static')\n```\nFor production, serve static via a CDN or nginx -- ASGI servers serving static is wasteful.")

    add("web",
        "Implement a FastAPI route returning 410 Gone for deprecated endpoints.",
        "```python\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.get('/old-thing')\ndef old():\n    raise HTTPException(410, 'this endpoint was removed; use /new-thing')\n```\n410 (vs 404) tells clients the resource is gone permanently and they should stop calling.")

    add("web",
        "Write a Flask endpoint that returns Server-Timing header.",
        "```python\nimport time\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    t0 = time.perf_counter()\n    # do work\n    elapsed = (time.perf_counter() - t0) * 1000\n    resp = app.make_response({'ok': True})\n    resp.headers['Server-Timing'] = f'app;dur={elapsed:.1f}'\n    return resp\n```\nBrowser DevTools shows Server-Timing in the network tab -- great for debugging slow endpoints from the client side.")

    add("web",
        "Implement an httpx context manager wrapper for retries.",
        "```python\nimport httpx\nfrom contextlib import contextmanager\n\n@contextmanager\ndef retrying_client(retries: int = 3):\n    transport = httpx.HTTPTransport(retries=retries)\n    with httpx.Client(transport=transport, timeout=10) as c:\n        yield c\n```\nhttpx's transport-level `retries=` only retries connection failures, not 5xx. Layer tenacity on top for response-based retries.")

    add("web",
        "Write a FastAPI route generating a presigned S3 URL stub.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.post('/uploads/sign')\ndef sign(filename: str):\n    # In real code, call boto3.client('s3').generate_presigned_url(...)\n    return {'url': f'https://bucket.s3/upload/{filename}?sig=stub'}\n```\nPresigned URLs let the client upload directly to S3, sidestepping your servers and saving bandwidth.")

    add("web",
        "Implement Flask graceful shutdown via signal handler.",
        "```python\nimport signal, sys\nfrom flask import Flask\n\napp = Flask(__name__)\n\ndef on_term(signum, frame):\n    print('flushing metrics...')\n    sys.exit(0)\n\nsignal.signal(signal.SIGTERM, on_term)\n```\nWith production WSGI servers (gunicorn, uwsgi) the server itself handles graceful shutdown; this manual handler is for development.")

    add("web",
        "Write a FastAPI endpoint that streams JSON lines.",
        "```python\nimport json\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\ndef gen():\n    for i in range(1000):\n        yield json.dumps({'i': i}) + '\\n'\n\n@app.get('/items.jsonl')\ndef items():\n    return StreamingResponse(gen(), media_type='application/x-ndjson')\n```\nNDJSON / JSONL is parser-friendly for streamed records -- one valid JSON value per line.")

    add("web",
        "Implement an httpx call that follows redirects manually.",
        "```python\nimport httpx\n\ndef follow(url: str, max_redirects: int = 5) -> httpx.Response:\n    with httpx.Client(follow_redirects=False, timeout=10) as c:\n        for _ in range(max_redirects):\n            r = c.get(url)\n            if 300 <= r.status_code < 400 and 'location' in r.headers:\n                url = r.headers['location']\n                continue\n            return r\n        raise RuntimeError('too many redirects')\n```\nUseful when you want to log every hop or strip credentials on cross-origin redirects.")

    add("web",
        "Write a FastAPI dependency that returns request user from JWT.",
        "```python\nimport jwt\nfrom fastapi import Depends, FastAPI, HTTPException, Header\n\nSECRET = 'change-me'\napp = FastAPI()\n\ndef current_user(authorization: str = Header(...)) -> dict:\n    if not authorization.startswith('Bearer '):\n        raise HTTPException(401)\n    try:\n        return jwt.decode(authorization[7:], SECRET, algorithms=['HS256'])\n    except jwt.PyJWTError:\n        raise HTTPException(401)\n\n@app.get('/me')\ndef me(user: dict = Depends(current_user)):\n    return user\n```\nPin `algorithms=['HS256']` -- unrestricted decode is a known critical CVE shape.")

    add("web",
        "Implement Flask route returning ETag for conditional requests.",
        "```python\nimport hashlib\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.get('/static-data')\ndef static_data():\n    body = jsonify(data='hello')\n    etag = '\"' + hashlib.md5(body.data).hexdigest() + '\"'\n    if request.if_none_match.contains(etag.strip('\"')):\n        return '', 304\n    body.set_etag(etag.strip('\"'))\n    return body\n```\nFlask has built-in `set_etag`/`if_none_match` helpers. The quoted form is wire-correct per RFC 7232.")

    add("web",
        "Write a FastAPI app exposing Prometheus metrics.",
        "```python\nfrom fastapi import FastAPI\nfrom prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST\nfrom fastapi.responses import Response\n\napp = FastAPI()\nrequests_total = Counter('requests_total', 'Total HTTP requests', ['path'])\n\n@app.middleware('http')\nasync def count(request, call_next):\n    requests_total.labels(path=request.url.path).inc()\n    return await call_next(request)\n\n@app.get('/metrics')\ndef metrics():\n    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)\n```\nIn production use `prometheus-fastapi-instrumentator` for the standard set of histograms.")

    add("web",
        "Implement Flask-SQLAlchemy session usage.",
        "```python\nfrom flask import Flask\nfrom flask_sqlalchemy import SQLAlchemy\n\napp = Flask(__name__)\napp.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'\ndb = SQLAlchemy(app)\n\nclass User(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n    email = db.Column(db.String(255), unique=True)\n```\nFlask-SQLAlchemy auto-manages the session per request. For SQLAlchemy 2.0 typed models, prefer raw SQLAlchemy.")

    add("web",
        "Write a FastAPI route guarded by Pydantic Settings.",
        "```python\nfrom fastapi import FastAPI, HTTPException\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    feature_x_enabled: bool = False\n\nsettings = Settings()\napp = FastAPI()\n\n@app.get('/x')\ndef x():\n    if not settings.feature_x_enabled:\n        raise HTTPException(404)\n    return {'ok': True}\n```\n`pydantic-settings` reads from env vars by default. Feature-flag dark launches without code changes.")

    add("web",
        "Implement a FastAPI background scheduler with apscheduler.",
        "```python\nfrom contextlib import asynccontextmanager\nfrom apscheduler.schedulers.asyncio import AsyncIOScheduler\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    sched = AsyncIOScheduler()\n    sched.add_job(lambda: print('tick'), 'interval', seconds=10)\n    sched.start()\n    yield\n    sched.shutdown()\n\napp = FastAPI(lifespan=lifespan)\n```\nFor multi-worker deployments, schedulers must run in exactly one process (or use a lock) -- otherwise jobs fire N times.")

    add("web",
        "Write a Flask CSRF protection setup.",
        "```python\nfrom flask import Flask\nfrom flask_wtf.csrf import CSRFProtect\n\napp = Flask(__name__)\napp.config['SECRET_KEY'] = 'change-me'\nCSRFProtect(app)\n```\n`SECRET_KEY` is required for CSRF token signing. Forms must include `{{ csrf_token() }}` in the template.")

    add("web",
        "Implement FastAPI route accepting form-data.",
        "```python\nfrom fastapi import FastAPI, Form\n\napp = FastAPI()\n\n@app.post('/login')\ndef login(username: str = Form(...), password: str = Form(...)):\n    return {'user': username}\n```\nUse `Form(...)` (not Pydantic) for `application/x-www-form-urlencoded` bodies.")

    add("web",
        "Write a Flask custom JSON encoder for datetime.",
        "```python\nfrom datetime import datetime\nfrom flask import Flask\nfrom flask.json.provider import DefaultJSONProvider\n\nclass MyProvider(DefaultJSONProvider):\n    def default(self, o):\n        if isinstance(o, datetime):\n            return o.isoformat()\n        return super().default(o)\n\napp = Flask(__name__)\napp.json = MyProvider(app)\n```\nFlask 2.3+ uses providers. The previous `app.json_encoder` API was removed.")

    add("web",
        "Implement a FastAPI route with custom status_code per outcome.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\nDB = {1: 'alice'}\n\n@app.get('/users/{id}')\ndef get(id: int):\n    if id not in DB:\n        return JSONResponse({'error': 'not found'}, status_code=404)\n    return {'name': DB[id]}\n```\nReturning a `JSONResponse` lets you control the status code per call without raising.")

    add("web",
        "Write a Flask CLI command.",
        "```python\nimport click\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.cli.command('seed')\n@click.option('--count', default=10)\ndef seed(count: int):\n    print(f'seeding {count} rows')\n```\nRun with `flask seed --count 100`. Great for one-off ops tasks (DB migrations, backfills).")

    add("web",
        "Implement FastAPI app with custom OpenAPI servers list.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI(\n    servers=[\n        {'url': 'https://api.example.com', 'description': 'prod'},\n        {'url': 'https://staging.example.com', 'description': 'staging'},\n    ],\n)\n```\nThe Swagger UI lets users switch between environments without changing the URL bar.")

    add("web",
        "Write a FastAPI app with multiple routers grouped by feature.",
        "```python\nfrom fastapi import APIRouter, FastAPI\n\nusers = APIRouter(prefix='/users', tags=['users'])\nitems = APIRouter(prefix='/items', tags=['items'])\n\n@users.get('/')\ndef list_users(): return []\n\n@items.get('/')\ndef list_items(): return []\n\napp = FastAPI()\napp.include_router(users)\napp.include_router(items)\n```\nPer-feature routers keep large apps navigable; tags drive Swagger grouping.")

    # ---- 50 data science entries ----
    add("data",
        "Read a CSV file with pandas.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.head())\n```\nFor mixed-type columns, pass `dtype={'col': str}` to avoid silent type coercion. `low_memory=False` prevents chunked-typing surprises on wide files.")

    add("data",
        "Filter a pandas DataFrame by column condition.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'age': [20, 30, 40], 'name': ['a', 'b', 'c']})\nadults = df[df['age'] >= 30]\nprint(adults)\n```\nBoolean indexing returns a copy by default. For chained filters, use `df.query('age >= 30 and city == \"NY\"')` -- often more readable.")

    add("data",
        "Group a DataFrame and compute mean per group.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','a','b','b'], 'val': [1, 2, 3, 4]})\nprint(df.groupby('cat')['val'].mean())\n```\n`.groupby(...).agg(...)` is the right tool when you need multiple aggregations or named outputs.")

    add("data",
        "Pivot a long DataFrame to wide.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'date': ['2024-01','2024-02']*2, 'metric': ['a','a','b','b'], 'val': [1,2,3,4]})\nwide = df.pivot(index='date', columns='metric', values='val')\nprint(wide)\n```\n`pivot` requires unique (index, columns) pairs; use `pivot_table` with `aggfunc=` for duplicates.")

    add("data",
        "Melt a wide DataFrame to long.",
        "```python\nimport pandas as pd\n\nwide = pd.DataFrame({'date': ['2024-01'], 'a': [1], 'b': [2]})\nlong = wide.melt(id_vars='date', var_name='metric', value_name='val')\nprint(long)\n```\nLong format is required by most plotting libraries (seaborn, plotnine).")

    add("data",
        "Merge two DataFrames on a key.",
        "```python\nimport pandas as pd\n\nleft = pd.DataFrame({'id': [1,2], 'x': ['a','b']})\nright = pd.DataFrame({'id': [1,3], 'y': [10, 20]})\njoined = left.merge(right, on='id', how='left')\nprint(joined)\n```\nDefault is inner join. Always be explicit about `how=`; the default has bitten enough engineers that explicitness is worth the keystrokes.")

    add("data",
        "Sort a DataFrame by multiple columns.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [3,1,2], 'b': [1,2,3]})\nprint(df.sort_values(['a','b'], ascending=[True, False]))\n```\nPass a list to `ascending=` to control direction per column.")

    add("data",
        "Drop rows with any null values.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1, None, 3]})\nclean = df.dropna()\nprint(clean)\n```\nUse `dropna(subset=['col'])` to only drop based on specific columns. Default behavior drops if any column is null.")

    add("data",
        "Fill missing values with column mean.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'a': [1.0, None, 3.0]})\ndf['a'] = df['a'].fillna(df['a'].mean())\n```\nSimple imputation. For modeling, add a `was_missing` indicator column so the model can learn from missingness too.")

    add("data",
        "Apply a function to a DataFrame column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'name': ['alice', 'bob']})\ndf['name_upper'] = df['name'].str.upper()\nprint(df)\n```\nUse `.str.<method>` rather than `.apply(str.upper)` -- vectorized string ops are 10x+ faster.")

    add("data",
        "Compute rolling average over a time series.",
        "```python\nimport pandas as pd\n\nidx = pd.date_range('2024-01-01', periods=10)\ns = pd.Series(range(10), index=idx)\nprint(s.rolling(3).mean())\n```\nRolling windows respect index order. For time-based windows, use `rolling('7D')` instead of integer windows.")

    add("data",
        "Resample a time series to monthly frequency.",
        "```python\nimport pandas as pd\n\nidx = pd.date_range('2024-01-01', periods=90, freq='D')\ns = pd.Series(range(90), index=idx)\nprint(s.resample('ME').sum())\n```\n`'ME'` is month-end; pandas 2.2+ migrated from the old `'M'` alias. `'MS'` for month-start.")

    add("data",
        "Read a Parquet file.",
        "```python\nimport pandas as pd\n\ndf = pd.read_parquet('data.parquet')\n```\nParquet preserves dtypes, compresses well, and reads faster than CSV. Standard format for analytical data warehouses.")

    add("data",
        "Write a DataFrame to Parquet with compression.",
        "```python\nimport pandas as pd\n\ndf.to_parquet('out.parquet', compression='zstd')\n```\nzstd is usually 2-5x faster to read/write than gzip with similar compression ratio. Default is snappy which is the fastest but largest.")

    add("data",
        "Create a numpy array of zeros.",
        "```python\nimport numpy as np\n\na = np.zeros((3, 4), dtype=np.float32)\nprint(a.shape, a.dtype)\n```\nAlways specify `dtype=` when memory matters. Default is float64; float32 halves memory for ML workloads with negligible accuracy loss.")

    add("data",
        "Reshape a numpy array.",
        "```python\nimport numpy as np\n\na = np.arange(12)\nb = a.reshape(3, 4)\nprint(b)\n```\nUse `-1` for one dimension to infer: `a.reshape(-1, 4)` works regardless of total size, as long as it's divisible by 4.")

    add("data",
        "Compute element-wise product of two numpy arrays.",
        "```python\nimport numpy as np\n\na = np.array([1, 2, 3])\nb = np.array([4, 5, 6])\nprint(a * b)\n```\n`*` is element-wise; `@` is matrix multiplication. The `np.dot` function does both depending on inputs -- prefer the operators for clarity.")

    add("data",
        "Compute the matrix inverse with numpy.",
        "```python\nimport numpy as np\n\nA = np.array([[1, 2], [3, 4]], dtype=float)\nA_inv = np.linalg.inv(A)\nprint(A_inv)\n```\nFor solving `Ax = b`, prefer `np.linalg.solve(A, b)` over `inv(A) @ b` -- it's faster and numerically more stable.")

    add("data",
        "Generate random samples from a normal distribution.",
        "```python\nimport numpy as np\n\nrng = np.random.default_rng(42)\nx = rng.normal(loc=0, scale=1, size=1000)\nprint(x.mean(), x.std())\n```\n`default_rng` is the modern API; legacy `np.random.normal` uses a global state and is harder to reason about.")

    add("data",
        "Plot a line chart with matplotlib.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\nfig, ax = plt.subplots()\nax.plot(x, np.sin(x))\nax.set_xlabel('x'); ax.set_ylabel('sin(x)')\nfig.tight_layout()\nfig.savefig('out.png', dpi=150)\n```\nAlways grab `fig, ax` explicitly. `plt.plot` global state is fine in notebooks but breaks in scripts and tests.")

    add("data",
        "Plot a histogram.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nrng = np.random.default_rng(0)\ndata = rng.normal(size=1000)\nfig, ax = plt.subplots()\nax.hist(data, bins=30)\nfig.savefig('hist.png')\n```\n`bins='auto'` chooses the bin count via Sturges/Freedman-Diaconis -- a sensible default if you don't have a specific reason.")

    add("data",
        "Create a scatter plot colored by category.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nrng = np.random.default_rng(1)\nx, y = rng.normal(size=(2, 100))\ncats = rng.integers(0, 3, 100)\nfig, ax = plt.subplots()\nsc = ax.scatter(x, y, c=cats, cmap='viridis')\nfig.colorbar(sc, ax=ax, label='category')\nfig.savefig('scatter.png')\n```\nFor categorical hue use seaborn -- it gives discrete legends rather than a continuous bar.")

    add("data",
        "Compute summary statistics of a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.describe())\nprint(df.describe(include='object'))  # for non-numeric\n```\n`describe(include='all')` shows both. `include='object'` gives count, unique, top, freq for string columns.")

    add("data",
        "One-hot encode a categorical column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'color': ['red','blue','red','green']})\nencoded = pd.get_dummies(df, columns=['color'], dtype=int)\nprint(encoded)\n```\nUse `drop_first=True` to avoid the dummy trap in linear models. For tree-based models it doesn't matter.")

    add("data",
        "Standardize features with sklearn.",
        "```python\nfrom sklearn.preprocessing import StandardScaler\nimport numpy as np\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(100, 3))\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\n```\nFit on train only; call `transform` on test. Otherwise you leak test statistics into training.")

    add("data",
        "Train a linear regression with sklearn.",
        "```python\nfrom sklearn.linear_model import LinearRegression\nimport numpy as np\n\nX = np.array([[1], [2], [3], [4]])\ny = np.array([2, 4, 6, 8])\nmodel = LinearRegression().fit(X, y)\nprint(model.coef_, model.intercept_)\n```\nFor regularization or large data, `Ridge`, `Lasso`, or `SGDRegressor` are usually better choices.")

    add("data",
        "Compute correlation matrix.",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nprint(df.corr(numeric_only=True))\n```\n`corr()` defaults to Pearson; pass `method='spearman'` for rank-based correlation that's robust to outliers.")

    add("data",
        "Plot a correlation heatmap.",
        "```python\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport pandas as pd\n\ndf = pd.read_csv('data.csv')\nfig, ax = plt.subplots(figsize=(8, 6))\nsns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', center=0, ax=ax)\nfig.savefig('corr.png')\n```\n`center=0` makes the diverging colormap show positive vs negative correlation visually.")

    add("data",
        "Split a dataset for train/test.",
        "```python\nfrom sklearn.model_selection import train_test_split\nimport numpy as np\n\nX = np.arange(100).reshape(-1, 1)\ny = np.arange(100)\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)\n```\nAlways set `random_state` for reproducibility. For classification use `stratify=y` to preserve class balance.")

    add("data",
        "Cross-validate a classifier.",
        "```python\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import cross_val_score\nimport numpy as np\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(100, 4))\ny = (X.sum(axis=1) > 0).astype(int)\nscores = cross_val_score(LogisticRegression(), X, y, cv=5)\nprint(scores.mean(), scores.std())\n```\nReport mean and std together -- a high mean with high std is much weaker evidence than a slightly lower mean with low std.")

    add("data",
        "Compute the confusion matrix for predictions.",
        "```python\nfrom sklearn.metrics import confusion_matrix\n\ny_true = [0, 1, 1, 0, 1]\ny_pred = [0, 1, 0, 0, 1]\nprint(confusion_matrix(y_true, y_pred))\n```\nRows are true labels, columns are predicted. `ConfusionMatrixDisplay.from_predictions` plots it nicely.")

    add("data",
        "Compute precision, recall, and F1.",
        "```python\nfrom sklearn.metrics import classification_report\n\nprint(classification_report([0,1,1,0], [0,1,0,0]))\n```\n`classification_report` gives per-class metrics plus macro/weighted averages. Use macro for class-imbalanced problems.")

    add("data",
        "Compute the ROC AUC score.",
        "```python\nfrom sklearn.metrics import roc_auc_score\nimport numpy as np\n\ny_true = np.array([0, 0, 1, 1])\ny_score = np.array([0.1, 0.4, 0.35, 0.8])\nprint(roc_auc_score(y_true, y_score))\n```\nPass probabilities (not class labels) to `y_score`. AUC is threshold-independent -- great for ranking quality, not for calibration.")

    add("data",
        "Plot a learning curve.",
        "```python\nfrom sklearn.model_selection import learning_curve\nfrom sklearn.linear_model import LogisticRegression\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(200, 4)); y = (X.sum(axis=1) > 0).astype(int)\nsizes, train, test = learning_curve(LogisticRegression(), X, y, cv=5)\nfig, ax = plt.subplots()\nax.plot(sizes, train.mean(axis=1), label='train')\nax.plot(sizes, test.mean(axis=1), label='val')\nax.legend(); fig.savefig('lc.png')\n```\nGap between train and val tells you bias vs variance: large gap = high variance, both low = high bias.")

    add("data",
        "Plot multiple subplots.",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\nfig, axes = plt.subplots(2, 2, figsize=(8, 6))\nfor i, ax in enumerate(axes.flat):\n    ax.plot(np.random.default_rng(i).normal(size=100))\n    ax.set_title(f'panel {i}')\nfig.tight_layout()\nfig.savefig('panels.png')\n```\n`axes.flat` iterates regardless of layout. `tight_layout()` keeps titles from overlapping the figure above.")

    add("data",
        "Read JSON into a DataFrame.",
        "```python\nimport pandas as pd\n\ndf = pd.read_json('data.json', lines=True)\n```\n`lines=True` reads NDJSON (one JSON value per line) which is the canonical format for streamed records.")

    add("data",
        "Set DataFrame index from a column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'date': pd.date_range('2024-01-01', periods=3), 'val': [1,2,3]})\ndf = df.set_index('date')\nprint(df.loc['2024-01-02'])\n```\n`set_index` makes time-series operations natural and label-based slicing fast.")

    add("data",
        "Compute percentile of a column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'val': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})\nprint(df['val'].quantile([0.25, 0.5, 0.75, 0.95]))\n```\nPass a list to get multiple percentiles at once. Default interpolation is `linear`.")

    add("data",
        "Group by and aggregate with multiple functions.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'cat': ['a','a','b'], 'val': [1, 2, 10]})\nout = df.groupby('cat').agg(\n    mean_val=('val', 'mean'),\n    max_val=('val', 'max'),\n    n=('val', 'size'),\n)\nprint(out)\n```\nNamed aggregation (introduced in pandas 0.25) is cleaner than the dict-of-dicts approach.")

    add("data",
        "Apply z-score normalization to a column.",
        "```python\nimport pandas as pd\n\ndf = pd.DataFrame({'val': [10, 20, 30, 40, 50]})\ndf['z'] = (df['val'] - df['val'].mean()) / df['val'].std()\nprint(df)\n```\nFor cross-validation, fit the mean/std on training only and apply to validation.")

    add("data",
        "Compute exponential moving average.",
        "```python\nimport pandas as pd\n\ns = pd.Series(range(10))\nprint(s.ewm(span=3).mean())\n```\nEWM gives more weight to recent values. `span` is the equivalent simple-MA window size.")

    add("data",
        "Plot a bar chart with seaborn.",
        "```python\nimport pandas as pd\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\ndf = pd.DataFrame({'cat': ['a','b','c'], 'val': [10, 20, 15]})\nfig, ax = plt.subplots()\nsns.barplot(data=df, x='cat', y='val', ax=ax)\nfig.savefig('bar.png')\n```\nPass `hue=` for grouped bars. Seaborn auto-computes confidence intervals when there are repeated values per category.")

    add("data",
        "Save a numpy array to a file.",
        "```python\nimport numpy as np\n\na = np.arange(100).reshape(10, 10)\nnp.save('a.npy', a)\nb = np.load('a.npy')\n```\n`.npy` preserves dtype and shape losslessly. `np.savez_compressed` for multiple arrays in one file.")

    add("data",
        "Compute the sum across a numpy axis.",
        "```python\nimport numpy as np\n\nA = np.arange(12).reshape(3, 4)\nprint(A.sum(axis=0))  # column sums\nprint(A.sum(axis=1))  # row sums\n```\n`axis=0` collapses rows (giving column-wise output), `axis=1` collapses columns. Always state explicitly -- `axis` is a constant source of off-by-one.")

    add("data",
        "Use numpy boolean masking.",
        "```python\nimport numpy as np\n\na = np.array([1, 2, 3, 4, 5])\nprint(a[a > 2])\n```\nMore concise than `a[np.where(a > 2)]`. The mask must have the same shape as the indexed array.")

    add("data",
        "Compute pairwise distances with scipy.",
        "```python\nimport numpy as np\nfrom scipy.spatial.distance import pdist, squareform\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(5, 3))\nD = squareform(pdist(X, metric='euclidean'))\nprint(D)\n```\n`pdist` returns a condensed (upper-triangular) distance vector; `squareform` expands to the full N x N matrix.")

    add("data",
        "Cluster data with KMeans.",
        "```python\nfrom sklearn.cluster import KMeans\nimport numpy as np\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(100, 2))\nkm = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)\nprint(km.cluster_centers_)\n```\nSet `n_init=10` (or 'auto') to run multiple initializations. Single init can land in a bad local minimum.")

    add("data",
        "Reduce dimensionality with PCA.",
        "```python\nfrom sklearn.decomposition import PCA\nimport numpy as np\n\nrng = np.random.default_rng(0)\nX = rng.normal(size=(100, 10))\npca = PCA(n_components=2).fit(X)\nprint(pca.explained_variance_ratio_)\nX_2d = pca.transform(X)\n```\nStandardize first if features are on different scales -- otherwise the largest-variance features dominate the components.")
