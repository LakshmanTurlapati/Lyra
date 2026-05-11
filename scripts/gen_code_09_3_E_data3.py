# SPDX-License-Identifier: MIT
"""Web/API (75) + data science (25). Wave 5 part 3."""


def register_all(add):
    # ----- web / API (75) -----
    add("web",
        "Write a FastAPI route that uploads multiple files at once.",
        "```python\nfrom fastapi import FastAPI, UploadFile\n\napp = FastAPI()\n\n@app.post('/uploads')\nasync def uploads(files: list[UploadFile]) -> dict:\n    out = []\n    for f in files:\n        data = await f.read()\n        out.append({'name': f.filename, 'size': len(data)})\n    return {'files': out}\n```\nFor large batches, stream each file to disk in chunks instead of buffering in memory.")

    add("web",
        "How do I unit-test a FastAPI app with TestClient?",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app\n\nclient = TestClient(app)\n\ndef test_health():\n    r = client.get('/healthz')\n    assert r.status_code == 200\n    assert r.json() == {'status': 'ok'}\n```\n`TestClient` wraps `httpx`; same surface as the production client so test code transfers easily.")

    add("web",
        "Write a Flask route that returns SSE events.",
        "```python\nimport time, json\nfrom flask import Flask, Response, stream_with_context\n\napp = Flask(__name__)\n\n@app.get('/events')\ndef events():\n    @stream_with_context\n    def gen():\n        for i in range(10):\n            yield f'data: {json.dumps({\"i\": i})}\\n\\n'\n            time.sleep(1)\n    return Response(gen(), mimetype='text/event-stream')\n```\nDisable buffering at any reverse proxy or the events arrive in bursts.")

    add("web",
        "How do I parse a multipart form with FastAPI including JSON fields?",
        "```python\nimport json\nfrom fastapi import FastAPI, File, Form, UploadFile\nfrom pydantic import BaseModel\n\nclass Meta(BaseModel):\n    name: str\n    tags: list[str] = []\n\napp = FastAPI()\n\n@app.post('/upload')\ndef upload(meta: str = Form(...), file: UploadFile = File(...)) -> dict:\n    parsed = Meta.model_validate(json.loads(meta))\n    return {'name': parsed.name, 'file': file.filename}\n```\nMixed multipart + JSON is awkward in any framework -- send the JSON payload as a Form field.")

    add("web",
        "Write a FastAPI dependency that opens a Redis connection per request.",
        "```python\nimport redis.asyncio as redis\nfrom fastapi import Depends, FastAPI\n\npool = redis.ConnectionPool.from_url('redis://localhost:6379/0')\n\nasync def get_redis():\n    client = redis.Redis(connection_pool=pool)\n    try:\n        yield client\n    finally:\n        await client.aclose()\n\napp = FastAPI()\n\n@app.get('/cached/{key}')\nasync def cached(key: str, r=Depends(get_redis)) -> dict:\n    val = await r.get(key)\n    return {'value': val.decode() if val else None}\n```\nShared pool, per-request client; cleanup in `finally`.")

    add("web",
        "How do I cache a FastAPI response in Redis?",
        "```python\nimport json\nimport redis.asyncio as redis\nfrom fastapi import FastAPI\n\nr = redis.Redis()\napp = FastAPI()\n\n@app.get('/products/{pid}')\nasync def get_product(pid: int) -> dict:\n    key = f'product:{pid}'\n    if cached := await r.get(key):\n        return json.loads(cached)\n    product = await fetch_from_db(pid)\n    await r.set(key, json.dumps(product), ex=60)\n    return product\n```\nAlways set a TTL (`ex=`); unbounded keys are how Redis quietly OOMs.")

    add("web",
        "Write a FastAPI route that returns 304 if If-None-Match matches.",
        "```python\nimport hashlib\nfrom fastapi import FastAPI, Header, Response\n\napp = FastAPI()\n\n@app.get('/items/{pid}')\ndef item(pid: int, if_none_match: str | None = Header(None)):\n    body = f'item-{pid}'.encode()\n    etag = hashlib.sha256(body).hexdigest()\n    if if_none_match == etag:\n        return Response(status_code=304)\n    return Response(content=body, headers={'ETag': etag, 'Content-Type': 'text/plain'})\n```\nETags shave bandwidth for clients that re-fetch unchanged resources.")

    add("web",
        "How do I do dependency overrides for FastAPI tests?",
        "```python\nfrom fastapi.testclient import TestClient\nfrom myapp import app, get_db\n\ndef fake_db():\n    yield InMemoryDb()\n\napp.dependency_overrides[get_db] = fake_db\nclient = TestClient(app)\n```\nReset between tests with `app.dependency_overrides.clear()` so suites stay isolated.")

    add("web",
        "Write a Flask app that returns a 429 with Retry-After when rate-limited.",
        "```python\nfrom flask import Flask, jsonify\nfrom flask_limiter import Limiter\nfrom flask_limiter.errors import RateLimitExceeded\nfrom flask_limiter.util import get_remote_address\n\napp = Flask(__name__)\nlimiter = Limiter(get_remote_address, app=app)\n\n@app.errorhandler(RateLimitExceeded)\ndef ratelimit_handler(e):\n    return (\n        jsonify(error='rate limited', detail=str(e.description)),\n        429,\n        {'Retry-After': str(int(e.reset_at - e.now)) if hasattr(e, 'reset_at') else '60'},\n    )\n\n@app.get('/api/items')\n@limiter.limit('5/minute')\ndef items():\n    return jsonify(items=[])\n```\n`Retry-After` lets well-behaved clients back off correctly.")

    add("web",
        "How do I keep secrets out of code in a Flask app?",
        "Use environment variables loaded by `pydantic-settings` or `python-dotenv`:\n```python\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    secret_key: str\n    database_url: str\n    class Config:\n        env_file = '.env'\n\nsettings = Settings()\napp.config['SECRET_KEY'] = settings.secret_key\n```\nGit-ignore `.env`; ship a checked-in `.env.example` documenting the required keys.")

    add("web",
        "Write a FastAPI custom exception handler.",
        "```python\nfrom fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\n\nclass NotEnoughQuotaError(Exception):\n    pass\n\napp = FastAPI()\n\n@app.exception_handler(NotEnoughQuotaError)\nasync def quota_handler(request: Request, exc: NotEnoughQuotaError):\n    return JSONResponse(\n        status_code=402,\n        content={'error': 'quota_exhausted', 'message': str(exc)},\n    )\n```\nMap domain exceptions to HTTP responses centrally so handlers don't repeat the mapping.")

    add("web",
        "How do I add tags and summaries to FastAPI docs?",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI(title='My API', version='1.0.0')\n\n@app.get('/items/', tags=['items'], summary='List items', description='Returns the catalog')\ndef list_items() -> list:\n    return []\n```\nGood OpenAPI metadata is what makes auto-generated docs actually useful for clients.")

    add("web",
        "Write a FastAPI app that mounts a sub-application.",
        "```python\nfrom fastapi import FastAPI\n\napi_v1 = FastAPI()\n\n@api_v1.get('/users')\ndef users():\n    return []\n\napp = FastAPI()\napp.mount('/v1', api_v1)\n```\nUseful for keeping versioned APIs cleanly separated; each sub-app has its own OpenAPI doc.")

    add("web",
        "How do I run a recurring background job in a FastAPI app?",
        "```python\nimport asyncio\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\nasync def heartbeat():\n    while True:\n        print('tick')\n        await asyncio.sleep(60)\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    task = asyncio.create_task(heartbeat())\n    yield\n    task.cancel()\n\napp = FastAPI(lifespan=lifespan)\n```\nFor anything beyond toy scope, use APScheduler or a real job runner -- in-process tasks die with the worker.")

    add("web",
        "Write a Flask app factory.",
        "```python\nfrom flask import Flask\n\ndef create_app(config: dict | None = None) -> Flask:\n    app = Flask(__name__)\n    app.config.from_mapping(config or {})\n    from .routes import bp\n    app.register_blueprint(bp)\n    return app\n```\nThe app-factory pattern makes testing and multiple configs trivial.")

    add("web",
        "How do I parse a JSON body without Pydantic in FastAPI?",
        "```python\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.post('/raw')\nasync def raw(request: Request) -> dict:\n    body = await request.json()\n    return {'echo': body}\n```\nUseful for opaque payloads, but you lose validation; prefer Pydantic for anything client-facing.")

    add("web",
        "Write a FastAPI app that records request and response bodies.",
        "```python\nimport logging\nfrom fastapi import FastAPI, Request\n\nlog = logging.getLogger(__name__)\napp = FastAPI()\n\n@app.middleware('http')\nasync def log_io(request: Request, call_next):\n    body = await request.body()\n    log.info('REQ %s %s body=%s', request.method, request.url.path, body[:200])\n    resp = await call_next(request)\n    return resp\n```\nNever log full bodies in production -- they contain secrets. Truncate or redact.")

    add("web",
        "How do I emit Prometheus metrics from a FastAPI app?",
        "Use `prometheus-fastapi-instrumentator`:\n```python\nfrom fastapi import FastAPI\nfrom prometheus_fastapi_instrumentator import Instrumentator\n\napp = FastAPI()\nInstrumentator().instrument(app).expose(app)\n```\nThe `/metrics` endpoint is added automatically; scrape it with your Prometheus server.")

    add("web",
        "Write a Flask blueprint that serializes SQLAlchemy models with Marshmallow.",
        "```python\nfrom flask import Blueprint, jsonify\nfrom marshmallow import Schema, fields\nfrom .models import User\n\nclass UserSchema(Schema):\n    id = fields.Int()\n    email = fields.Email()\n    created_at = fields.DateTime()\n\nbp = Blueprint('users', __name__, url_prefix='/users')\nschema = UserSchema(many=True)\n\n@bp.get('/')\ndef list_users():\n    return jsonify(schema.dump(User.query.all()))\n```\nMarshmallow keeps serialization separate from the model; Pydantic does the same job in FastAPI.")

    add("web",
        "How do I run database migrations with Alembic?",
        "```bash\nalembic init migrations          # one-time scaffold\nalembic revision --autogenerate -m 'add users table'\nalembic upgrade head\n```\nAlways review autogenerated migrations -- column type changes and renames need manual help.")

    add("web",
        "Write a FastAPI endpoint that returns a redirect.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import RedirectResponse\n\napp = FastAPI()\n\n@app.get('/old')\ndef old() -> RedirectResponse:\n    return RedirectResponse(url='/new', status_code=308)\n```\n301/308 are permanent (cached by browsers and crawlers); 302/307 are temporary.")

    add("web",
        "How do I configure CORS with credentials in Flask?",
        "```python\nfrom flask import Flask\nfrom flask_cors import CORS\n\napp = Flask(__name__)\nCORS(app, origins=['https://example.com'], supports_credentials=True)\n```\nWhen `supports_credentials=True`, the origin list cannot be `'*'` -- the browser will reject the response.")

    add("web",
        "Write a FastAPI app that returns localized error messages.",
        "```python\nfrom fastapi import FastAPI, Header, HTTPException\n\nMESSAGES = {\n    'en': {'not_found': 'Item not found'},\n    'es': {'not_found': 'Articulo no encontrado'},\n}\n\napp = FastAPI()\n\n@app.get('/items/{pid}')\ndef item(pid: int, accept_language: str = Header('en')):\n    lang = accept_language.split(',')[0].split('-')[0]\n    if pid > 100:\n        raise HTTPException(404, MESSAGES.get(lang, MESSAGES['en'])['not_found'])\n    return {'id': pid}\n```\nFor real i18n use Babel or `python-i18n`; this is the principle, not a production solution.")

    add("web",
        "How do I implement OAuth2 password flow in FastAPI?",
        "```python\nfrom fastapi import Depends, FastAPI, HTTPException\nfrom fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm\n\noauth2 = OAuth2PasswordBearer(tokenUrl='token')\napp = FastAPI()\n\n@app.post('/token')\ndef login(form: OAuth2PasswordRequestForm = Depends()) -> dict:\n    if form.username != 'alice' or form.password != 'secret':\n        raise HTTPException(401, 'bad creds')\n    return {'access_token': 'fake', 'token_type': 'bearer'}\n\n@app.get('/me')\ndef me(token: str = Depends(oauth2)) -> dict:\n    return {'token': token[:8] + '...'}\n```\nReplace `'fake'` with a signed JWT and a real user lookup.")

    add("web",
        "Write a FastAPI endpoint that returns an HTML response.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import HTMLResponse\n\napp = FastAPI()\n\n@app.get('/', response_class=HTMLResponse)\ndef root() -> str:\n    return '<h1>Hello</h1>'\n```\nFor real templates use Jinja2 (`fastapi.templating.Jinja2Templates`) and never f-string user input into HTML.")

    add("web",
        "How do I do service-to-service auth with mTLS in Python?",
        "```python\nimport httpx\n\nclient = httpx.Client(\n    cert=('/etc/certs/client.crt', '/etc/certs/client.key'),\n    verify='/etc/certs/ca.crt',\n    base_url='https://internal.example.com',\n    timeout=5.0,\n)\nresp = client.get('/api/items')\nresp.raise_for_status()\n```\nKeys belong on disk with 0600 permissions and out of source control. Rotate via your secrets manager.")

    add("web",
        "Write a FastAPI app with a custom OpenAPI title and version.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI(\n    title='Inventory API',\n    description='Manage SKUs, stock, and warehouses.',\n    version='2.1.3',\n    contact={'name': 'Platform team', 'email': 'platform@example.com'},\n    license_info={'name': 'MIT'},\n)\n```\nGood metadata makes the auto-generated `/docs` page useful to consumers.")

    add("web",
        "How do I serialize Decimal correctly in a FastAPI response?",
        "```python\nfrom decimal import Decimal\nfrom pydantic import BaseModel\nfrom fastapi import FastAPI\n\nclass Money(BaseModel):\n    amount: Decimal\n    currency: str\n\napp = FastAPI()\n\n@app.get('/price', response_model=Money)\ndef price():\n    return Money(amount=Decimal('19.95'), currency='USD')\n```\nPydantic preserves precision (it serialises `Decimal` to a string in v2 by default). Never round monetary values to float.")

    add("web",
        "Write a FastAPI route that proxies a download from S3.",
        "```python\nimport boto3\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\ns3 = boto3.client('s3')\napp = FastAPI()\n\n@app.get('/download/{key}')\ndef download(key: str):\n    obj = s3.get_object(Bucket='my-bucket', Key=key)\n    return StreamingResponse(\n        obj['Body'].iter_chunks(),\n        media_type=obj['ContentType'],\n        headers={'Content-Disposition': f'attachment; filename=\"{key}\"'},\n    )\n```\nFor public URLs, prefer pre-signed URLs and let the client hit S3 directly -- saves your bandwidth.")

    add("web",
        "How do I convert a SQLAlchemy model to a Pydantic model?",
        "```python\nfrom pydantic import BaseModel, ConfigDict\n\nclass UserOut(BaseModel):\n    model_config = ConfigDict(from_attributes=True)\n    id: int\n    email: str\n\nuser = User(id=1, email='a@b.com')\nUserOut.model_validate(user).model_dump()\n```\n`from_attributes=True` lets Pydantic read ORM objects via attribute access. Use it as the response_model directly.")

    add("web",
        "Write a Flask route that streams a query result without materialising it.",
        "```python\nimport json\nfrom flask import Response, stream_with_context\n\n@app.get('/users/stream')\ndef stream_users():\n    @stream_with_context\n    def gen():\n        first = True\n        yield '['\n        for u in User.query.yield_per(500):\n            if not first:\n                yield ','\n            first = False\n            yield json.dumps({'id': u.id, 'email': u.email})\n        yield ']'\n    return Response(gen(), mimetype='application/json')\n```\n`yield_per(N)` keeps memory bounded; otherwise SQLAlchemy buffers the entire result set.")

    add("web",
        "How do I add a startup script that waits for the database in a container?",
        "```python\nimport os, time\nimport psycopg\nfrom psycopg import OperationalError\n\nurl = os.environ['DATABASE_URL']\nfor attempt in range(30):\n    try:\n        psycopg.connect(url, connect_timeout=2).close()\n        print('db ready')\n        break\n    except OperationalError:\n        print(f'attempt {attempt}: db not ready, sleeping')\n        time.sleep(1)\nelse:\n    raise SystemExit('db never became ready')\n```\nDocker/Compose `depends_on` doesn't actually wait for ready -- only running. A wait loop is the right pattern.")

    add("web",
        "Write a FastAPI middleware that adds security headers.",
        "```python\nfrom fastapi import FastAPI, Request\n\napp = FastAPI()\n\n@app.middleware('http')\nasync def security_headers(request: Request, call_next):\n    resp = await call_next(request)\n    resp.headers['X-Content-Type-Options'] = 'nosniff'\n    resp.headers['X-Frame-Options'] = 'DENY'\n    resp.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'\n    resp.headers['Referrer-Policy'] = 'no-referrer'\n    return resp\n```\nCSP belongs here too, but it's app-specific -- enable it after testing the policy carefully.")

    add("web",
        "How do I implement long polling with FastAPI?",
        "```python\nimport asyncio\nfrom fastapi import FastAPI\n\napp = FastAPI()\nlatest = {'rev': 0, 'data': None}\n\n@app.get('/poll')\nasync def poll(since: int = 0, timeout: int = 25):\n    deadline = asyncio.get_running_loop().time() + timeout\n    while asyncio.get_running_loop().time() < deadline:\n        if latest['rev'] > since:\n            return {'rev': latest['rev'], 'data': latest['data']}\n        await asyncio.sleep(0.5)\n    return {'rev': since, 'data': None}\n```\nWebSockets or SSE scale better, but long polling works through any HTTP infrastructure.")

    add("web",
        "Write a FastAPI route that returns a 422 when validation fails for a custom rule.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, field_validator\n\nclass Order(BaseModel):\n    qty: int\n    @field_validator('qty')\n    @classmethod\n    def positive(cls, v: int) -> int:\n        if v <= 0:\n            raise ValueError('qty must be positive')\n        return v\n\napp = FastAPI()\n\n@app.post('/orders')\ndef create(o: Order):\n    return {'qty': o.qty}\n```\nPydantic raises `ValidationError`, FastAPI maps it to a 422 with field-level detail automatically.")

    add("web",
        "How do I stream a large response from a Flask view?",
        "```python\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/big')\ndef big():\n    def gen():\n        for i in range(1_000_000):\n            yield f'{i}\\n'\n    return Response(gen(), mimetype='text/plain')\n```\nFlask's WSGI mode iterates the generator lazily; the response is streamed without buffering.")

    add("web",
        "Write a FastAPI endpoint that proxies WebSocket traffic.",
        "```python\nimport asyncio, websockets\nfrom fastapi import FastAPI, WebSocket, WebSocketDisconnect\n\napp = FastAPI()\n\n@app.websocket('/proxy')\nasync def proxy(ws: WebSocket):\n    await ws.accept()\n    async with websockets.connect('wss://upstream.example.com') as up:\n        async def from_client():\n            try:\n                while True:\n                    await up.send(await ws.receive_text())\n            except WebSocketDisconnect:\n                await up.close()\n        async def from_upstream():\n            async for msg in up:\n                await ws.send_text(msg)\n        await asyncio.gather(from_client(), from_upstream())\n```\nProxy WebSockets only when you must; reverse-proxying at the edge (nginx, Envoy) is simpler and faster.")

    add("web",
        "How do I keep a FastAPI app's docs hidden in production?",
        "```python\nimport os\nfrom fastapi import FastAPI\n\nopen_docs = os.getenv('ENV') != 'prod'\napp = FastAPI(\n    docs_url='/docs' if open_docs else None,\n    redoc_url=None,\n    openapi_url='/openapi.json' if open_docs else None,\n)\n```\nLeak prevention: hidden docs reduce attack surface but don't replace auth -- protect the API itself.")

    add("web",
        "Write a Flask error handler that logs the exception and returns 500 JSON.",
        "```python\nimport logging, traceback\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\nlog = logging.getLogger(__name__)\n\n@app.errorhandler(Exception)\ndef handle_unexpected(exc):\n    log.exception('unhandled error')\n    return jsonify(error='internal_error'), 500\n```\n`log.exception` captures the traceback automatically. Don't leak the message to the client.")

    add("web",
        "How do I do request-scoped DB transactions in FastAPI?",
        "```python\nfrom fastapi import Depends, FastAPI\nfrom sqlalchemy.orm import Session\n\ndef get_session():\n    db = SessionLocal()\n    try:\n        yield db\n        db.commit()\n    except Exception:\n        db.rollback()\n        raise\n    finally:\n        db.close()\n```\nCommit on success, roll back on any exception, always close. The dependency yields the session into the handler.")

    add("web",
        "Write a FastAPI endpoint that returns Pillow-generated PNG bytes.",
        "```python\nimport io\nfrom fastapi import FastAPI\nfrom fastapi.responses import Response\nfrom PIL import Image, ImageDraw\n\napp = FastAPI()\n\n@app.get('/avatar/{name}')\ndef avatar(name: str):\n    img = Image.new('RGB', (128, 128), color=(64, 128, 200))\n    ImageDraw.Draw(img).text((10, 60), name[:8], fill='white')\n    buf = io.BytesIO()\n    img.save(buf, format='PNG')\n    return Response(buf.getvalue(), media_type='image/png')\n```\nFor signed avatars cache by hash; image generation is CPU-bound and worth caching.")

    add("web",
        "How do I rate-limit by API key instead of IP?",
        "Use a custom key function with `slowapi`:\n```python\nfrom fastapi import Request\nfrom slowapi import Limiter\n\ndef api_key_id(request: Request) -> str:\n    return request.headers.get('X-API-Key') or 'anon'\n\nlimiter = Limiter(key_func=api_key_id, storage_uri='redis://localhost:6379')\n```\nFalling back to `'anon'` shares one bucket across unauthenticated callers -- usually what you want.")

    add("web",
        "Write a FastAPI app that uses background queue for emails.",
        "```python\nfrom fastapi import BackgroundTasks, FastAPI\n\napp = FastAPI()\n\ndef send_email(to: str, subject: str, body: str):\n    print(f'[email -> {to}] {subject}: {body}')\n\n@app.post('/signup')\ndef signup(email: str, tasks: BackgroundTasks) -> dict:\n    tasks.add_task(send_email, email, 'Welcome', 'Hello!')\n    return {'status': 'queued'}\n```\nFor reliability move to a real queue (Celery, RQ, Dramatiq) -- in-process tasks vanish when the worker restarts.")

    add("web",
        "How do I handle file path traversal safely in a FastAPI download endpoint?",
        "```python\nfrom pathlib import Path\nfrom fastapi import FastAPI, HTTPException\nfrom fastapi.responses import FileResponse\n\nDATA_DIR = Path('/srv/files').resolve()\napp = FastAPI()\n\n@app.get('/files/{name}')\ndef serve(name: str):\n    target = (DATA_DIR / name).resolve()\n    if not target.is_relative_to(DATA_DIR) or not target.is_file():\n        raise HTTPException(404)\n    return FileResponse(target)\n```\n`Path.resolve()` + `is_relative_to` blocks `../` escapes deterministically.")

    add("web",
        "Write a FastAPI app that uses async SQLAlchemy.",
        "```python\nfrom fastapi import Depends, FastAPI\nfrom sqlalchemy import select\nfrom sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine\n\nengine = create_async_engine('postgresql+asyncpg://...')\nSessionLocal = async_sessionmaker(engine, expire_on_commit=False)\n\nasync def get_db():\n    async with SessionLocal() as db:\n        yield db\n\napp = FastAPI()\n\n@app.get('/users')\nasync def users(db: AsyncSession = Depends(get_db)):\n    result = await db.execute(select(User))\n    return result.scalars().all()\n```\nAsync SQLAlchemy needs an async driver (`asyncpg`, `aiosqlite`).")

    add("web",
        "How do I implement simple feature flags in a FastAPI app?",
        "```python\nimport os\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\ndef flag_enabled(name: str) -> bool:\n    return os.getenv(f'FLAG_{name.upper()}', '').lower() in {'1', 'true', 'yes'}\n\n@app.get('/items')\ndef items() -> dict:\n    if flag_enabled('new_listing_format'):\n        return {'items': [], 'format': 'v2'}\n    return {'items': []}\n```\nFor anything more dynamic, reach for LaunchDarkly or Unleash. Env-var flags are fine for shipping safe defaults.")

    add("web",
        "Write a Flask endpoint that parses an uploaded CSV and returns row count.",
        "```python\nimport csv, io\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.post('/csv')\ndef parse_csv():\n    file = request.files['file']\n    text = io.TextIOWrapper(file.stream, encoding='utf-8')\n    reader = csv.reader(text)\n    next(reader, None)  # skip header\n    rows = sum(1 for _ in reader)\n    return jsonify(rows=rows)\n```\nWrap the binary file stream in `TextIOWrapper`; otherwise `csv.reader` chokes on bytes.")

    add("web",
        "How do I set up Sentry for a FastAPI app?",
        "```python\nimport sentry_sdk\nfrom sentry_sdk.integrations.fastapi import FastApiIntegration\nfrom fastapi import FastAPI\n\nsentry_sdk.init(\n    dsn='https://example@sentry.io/1',\n    traces_sample_rate=0.1,\n    integrations=[FastApiIntegration()],\n)\napp = FastAPI()\n```\nKeep `traces_sample_rate` low in production; full sampling can blow your quota fast.")

    add("web",
        "Write a FastAPI endpoint that posts JSON to a webhook with retries.",
        "```python\nimport httpx\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\n\n@app.post('/relay')\nasync def relay(payload: dict) -> dict:\n    transport = httpx.AsyncHTTPTransport(retries=3)\n    async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:\n        try:\n            r = await client.post('https://hook.example.com', json=payload)\n            r.raise_for_status()\n        except httpx.HTTPError as exc:\n            raise HTTPException(502, str(exc))\n    return {'forwarded': True}\n```\n`AsyncHTTPTransport(retries=...)` retries connection errors only; for status retries use `tenacity`.")

    add("web",
        "How do I implement HMAC webhook signature verification?",
        "```python\nimport hmac, hashlib\nfrom fastapi import FastAPI, Header, HTTPException, Request\n\nSECRET = b'change-me'\napp = FastAPI()\n\n@app.post('/webhook')\nasync def hook(request: Request, x_signature: str = Header(...)):\n    body = await request.body()\n    expected = hmac.new(SECRET, body, hashlib.sha256).hexdigest()\n    if not hmac.compare_digest(expected, x_signature):\n        raise HTTPException(401, 'bad signature')\n    return {'ok': True}\n```\n`hmac.compare_digest` is mandatory -- regular `==` leaks the signature via timing.")

    add("web",
        "Write a FastAPI route that returns a Pydantic model with computed fields.",
        "```python\nfrom pydantic import BaseModel, computed_field\nfrom fastapi import FastAPI\n\nclass Order(BaseModel):\n    unit_price: float\n    qty: int\n    @computed_field\n    @property\n    def total(self) -> float:\n        return round(self.unit_price * self.qty, 2)\n\napp = FastAPI()\n\n@app.get('/orders/{oid}', response_model=Order)\ndef get(oid: int):\n    return Order(unit_price=9.99, qty=3)\n```\nComputed fields appear in serialised output and the OpenAPI schema.")

    add("web",
        "How do I implement CSRF protection in Flask?",
        "Use `flask-wtf`:\n```python\nfrom flask import Flask\nfrom flask_wtf.csrf import CSRFProtect\n\napp = Flask(__name__)\napp.config['SECRET_KEY'] = 'change-me'\ncsrf = CSRFProtect(app)\n```\nAdd `{{ csrf_token() }}` to forms; for AJAX, send the token as `X-CSRFToken`. APIs using bearer tokens don't need CSRF.")

    add("web",
        "Write a FastAPI endpoint that paginates with cursors.",
        "```python\nimport base64, json\nfrom fastapi import FastAPI\n\napp = FastAPI()\nDATA = list(range(1000))\n\ndef encode(cursor: dict) -> str:\n    return base64.urlsafe_b64encode(json.dumps(cursor).encode()).decode()\n\ndef decode(cursor: str) -> dict:\n    return json.loads(base64.urlsafe_b64decode(cursor))\n\n@app.get('/items')\ndef items(cursor: str | None = None, limit: int = 20):\n    start = decode(cursor)['offset'] if cursor else 0\n    page = DATA[start:start + limit]\n    next_cursor = encode({'offset': start + limit}) if start + limit < len(DATA) else None\n    return {'items': page, 'next': next_cursor}\n```\nCursors avoid the consistency issues of offset-based pagination on changing data.")

    add("web",
        "How do I add per-route caching headers in FastAPI?",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import JSONResponse\n\napp = FastAPI()\n\n@app.get('/static-list')\ndef static_list():\n    return JSONResponse(\n        content={'items': [1, 2, 3]},\n        headers={'Cache-Control': 'public, max-age=300, stale-while-revalidate=60'},\n    )\n```\n`stale-while-revalidate` lets the CDN serve stale data while it refreshes -- great for low-latency apps.")

    add("web",
        "Write a FastAPI app that uses dependency injection for a logger.",
        "```python\nimport logging\nfrom fastapi import Depends, FastAPI\n\ndef get_logger() -> logging.Logger:\n    return logging.getLogger('myapp')\n\napp = FastAPI()\n\n@app.get('/items')\ndef items(log: logging.Logger = Depends(get_logger)):\n    log.info('listing items')\n    return []\n```\nMakes the logger easy to override in tests with a capturing handler.")

    add("web",
        "How do I implement a 'try once' deduplication header?",
        "```python\nimport time\nfrom fastapi import FastAPI, Header, HTTPException\n\napp = FastAPI()\nseen: dict[str, float] = {}\nTTL = 600\n\n@app.post('/charge')\ndef charge(amount: float, idempotency_key: str = Header(...)) -> dict:\n    now = time.time()\n    seen.update({k: t for k, t in seen.items() if now - t < TTL})\n    if idempotency_key in seen:\n        raise HTTPException(409, 'duplicate')\n    seen[idempotency_key] = now\n    return {'charged': amount}\n```\nMove the dedup table to Redis for multi-instance services.")

    add("web",
        "Write a Flask route that downloads a file with a content-disposition header.",
        "```python\nfrom flask import Flask, send_file\n\napp = Flask(__name__)\n\n@app.get('/report.pdf')\ndef report():\n    return send_file('report.pdf', as_attachment=True, download_name='report-2026.pdf')\n```\n`send_file` handles range requests, mimetype detection, and `If-Modified-Since` automatically.")

    add("web",
        "How do I keep an httpx client connection-pooled across a FastAPI app?",
        "```python\nfrom contextlib import asynccontextmanager\nimport httpx\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    app.state.http = httpx.AsyncClient(timeout=5.0)\n    yield\n    await app.state.http.aclose()\n\napp = FastAPI(lifespan=lifespan)\n\n@app.get('/forward')\nasync def forward():\n    r = await app.state.http.get('https://example.com')\n    return {'status': r.status_code}\n```\nReusing the client preserves the connection pool and the per-host concurrency limits.")

    add("web",
        "Write a FastAPI endpoint that logs slow queries.",
        "```python\nimport time, logging\nfrom fastapi import FastAPI, Request\n\nlog = logging.getLogger(__name__)\napp = FastAPI()\n\n@app.middleware('http')\nasync def slow_log(request: Request, call_next):\n    start = time.perf_counter()\n    resp = await call_next(request)\n    dur = (time.perf_counter() - start) * 1000\n    if dur > 500:\n        log.warning('slow %s %s %.0fms', request.method, request.url.path, dur)\n    return resp\n```\nThreshold and log level should match your SLO budget.")

    add("web",
        "How do I do server-side rendering with Jinja2 in FastAPI?",
        "```python\nfrom fastapi import FastAPI, Request\nfrom fastapi.templating import Jinja2Templates\n\napp = FastAPI()\ntemplates = Jinja2Templates(directory='templates')\n\n@app.get('/')\ndef index(request: Request):\n    return templates.TemplateResponse('index.html', {'request': request, 'name': 'world'})\n```\nThe `request` key is required by the response object; the rest are normal template variables.")

    add("web",
        "Write a FastAPI app that serves a robots.txt and sitemap.xml.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.responses import PlainTextResponse, Response\n\napp = FastAPI()\n\n@app.get('/robots.txt', response_class=PlainTextResponse)\ndef robots():\n    return 'User-agent: *\\nDisallow: /admin\\n'\n\n@app.get('/sitemap.xml')\ndef sitemap():\n    body = '<?xml version=\"1.0\" encoding=\"UTF-8\"?><urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\"></urlset>'\n    return Response(body, media_type='application/xml')\n```\nFor large sitemaps, generate them offline and serve from object storage.")

    add("web",
        "How do I parse a query parameter as a list in FastAPI?",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\n\n@app.get('/search')\ndef search(tag: list[str] = Query(default_factory=list)):\n    return {'tags': tag}\n```\n`/search?tag=a&tag=b` becomes `['a', 'b']`. Use `default_factory=list`, not `default=[]`, to avoid the mutable-default trap.")

    add("web",
        "Write a Flask CLI command using click.",
        "```python\nimport click\nfrom flask import Flask\n\napp = Flask(__name__)\n\n@app.cli.command('seed')\n@click.option('--count', default=10)\ndef seed(count: int):\n    for i in range(count):\n        print(f'seed user {i}')\n```\nRun with `flask seed --count 100`. Flask's CLI is just `click` under the hood.")

    add("web",
        "How do I gracefully drain in-flight requests on shutdown?",
        "Run uvicorn with a shutdown timeout, and let the orchestrator (Kubernetes, ECS) send SIGTERM before SIGKILL:\n```bash\nuvicorn main:app --timeout-graceful-shutdown 30\n```\nIn FastAPI, the lifespan shutdown phase fires on SIGTERM; finish your background tasks there before the timeout expires.")

    add("web",
        "Write a FastAPI route that uses ETags for client caching.",
        "```python\nimport hashlib, json\nfrom fastapi import FastAPI, Header, Response\n\napp = FastAPI()\nDATA = {'items': [1, 2, 3]}\n\n@app.get('/data')\ndef data(if_none_match: str | None = Header(None)):\n    body = json.dumps(DATA).encode()\n    etag = hashlib.md5(body).hexdigest()\n    if if_none_match == etag:\n        return Response(status_code=304, headers={'ETag': etag})\n    return Response(content=body, media_type='application/json', headers={'ETag': etag})\n```\nFor strong ETags use a content hash; for weak ETags include the version field.")

    add("web",
        "How do I configure Flask for testing?",
        "```python\nimport pytest\nfrom myapp import create_app\n\n@pytest.fixture\ndef app():\n    app = create_app({'TESTING': True, 'DATABASE_URL': 'sqlite:///:memory:'})\n    yield app\n\n@pytest.fixture\ndef client(app):\n    return app.test_client()\n```\nCombine with `app.test_request_context()` when you need to call code that depends on the Flask request stack.")

    add("web",
        "Write a FastAPI app that serves websocket chat rooms.",
        "```python\nfrom fastapi import FastAPI, WebSocket, WebSocketDisconnect\n\napp = FastAPI()\nrooms: dict[str, set[WebSocket]] = {}\n\n@app.websocket('/chat/{room}')\nasync def chat(ws: WebSocket, room: str):\n    await ws.accept()\n    rooms.setdefault(room, set()).add(ws)\n    try:\n        while True:\n            msg = await ws.receive_text()\n            for peer in list(rooms[room]):\n                if peer is not ws:\n                    await peer.send_text(msg)\n    except WebSocketDisconnect:\n        rooms[room].discard(ws)\n```\nFor multi-instance deployment use Redis pub/sub to fan out across processes.")

    add("web",
        "How do I do API versioning with FastAPI?",
        "Two common patterns:\n```python\n# 1) URL prefix\nfrom fastapi import APIRouter\nv1 = APIRouter(prefix='/api/v1')\nv2 = APIRouter(prefix='/api/v2')\n\n# 2) Header-based\n# Inspect 'Accept-Version' or 'X-API-Version' in a dependency.\n```\nURL-based is easier to debug, cache, and document. Headers are tidier but harder for ad-hoc curl users.")

    add("web",
        "Write a FastAPI app that exposes /healthz and /readyz.",
        "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n_db_ready = True\n\n@app.get('/healthz')\ndef healthz() -> dict:\n    return {'status': 'ok'}\n\n@app.get('/readyz')\ndef readyz():\n    if not _db_ready:\n        return Response(status_code=503)\n    return {'status': 'ready'}\n```\n`/healthz` returns 200 as long as the process is alive; `/readyz` checks dependencies.")

    add("web",
        "How do I send a multipart/form-data POST in Python?",
        "```python\nimport requests\n\nfiles = {'photo': ('cat.jpg', open('cat.jpg', 'rb'), 'image/jpeg')}\ndata = {'caption': 'cute'}\nresp = requests.post('https://api.example.com/upload', files=files, data=data, timeout=30)\nresp.raise_for_status()\n```\n`requests` builds the multipart body for you. Close the file (or use a `with` block) to avoid descriptor leaks on retries.")

    add("web",
        "Write a FastAPI exception handler that masks internal errors in production.",
        "```python\nimport os, logging\nfrom fastapi import FastAPI, Request\nfrom fastapi.responses import JSONResponse\n\nIS_PROD = os.getenv('ENV') == 'prod'\nlog = logging.getLogger(__name__)\napp = FastAPI()\n\n@app.exception_handler(Exception)\nasync def fallback(request: Request, exc: Exception):\n    log.exception('unhandled')\n    detail = 'internal error' if IS_PROD else f'{type(exc).__name__}: {exc}'\n    return JSONResponse({'error': detail}, status_code=500)\n```\nDev gets stack-trace info; prod gets a generic message.")

    add("web",
        "How do I do client-side load balancing across upstream replicas with httpx?",
        "Use a connection pool per upstream and round-robin across them:\n```python\nimport itertools, httpx\n\nupstreams = [httpx.AsyncClient(base_url=u, timeout=5.0)\n             for u in ['http://a:8000', 'http://b:8000']]\nrr = itertools.cycle(upstreams)\n\nasync def fetch(path: str):\n    client = next(rr)\n    r = await client.get(path)\n    r.raise_for_status()\n    return r.json()\n```\nFor anything serious use a service mesh or a dedicated client like `aiocluster`.")

    add("web",
        "Write a FastAPI route that returns paginated MongoDB documents.",
        "```python\nfrom fastapi import FastAPI, Query\nfrom motor.motor_asyncio import AsyncIOMotorClient\n\nclient = AsyncIOMotorClient('mongodb://localhost')\ncoll = client.db.items\napp = FastAPI()\n\n@app.get('/items')\nasync def items(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):\n    cursor = coll.find().skip((page - 1) * size).limit(size)\n    return [doc async for doc in cursor]\n```\nUse `_id` cursors instead of skip/limit once collections grow -- skip is O(n) on the server.")

    add("web",
        "Write a FastAPI app that gracefully closes a connection pool on shutdown.",
        "```python\nfrom contextlib import asynccontextmanager\nimport asyncpg\nfrom fastapi import FastAPI\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    app.state.pool = await asyncpg.create_pool('postgresql://localhost/app', max_size=10)\n    yield\n    await app.state.pool.close()\n\napp = FastAPI(lifespan=lifespan)\n\n@app.get('/users/{uid}')\nasync def user(uid: int):\n    async with app.state.pool.acquire() as conn:\n        return dict(await conn.fetchrow('SELECT id, email FROM users WHERE id=$1', uid))\n```\nClosing the pool drains in-flight connections cleanly; otherwise Postgres logs scary 'unexpected EOF' lines.")

    add("web",
        "How do I mount a Flask app inside a FastAPI app?",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.wsgi import WSGIMiddleware\nfrom flask_app import app as flask_app\n\napp = FastAPI()\napp.mount('/legacy', WSGIMiddleware(flask_app))\n```\nUseful during migration; the Flask app keeps running while you port routes one by one.")

    # ----- data science (25) -----
    add("data",
        "How do I read a CSV into a pandas DataFrame with explicit dtypes?",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv(\n    'sales.csv',\n    dtype={'product_id': 'int32', 'price': 'float32', 'category': 'category'},\n    parse_dates=['order_date'],\n)\n```\nSetting dtypes at load time saves both memory and a follow-up `astype` pass; categorical columns are especially cheap.")

    add("data",
        "Write code to compute a 7-day rolling mean in pandas.",
        "```python\nimport pandas as pd\n\ndaily = pd.read_csv('daily.csv', parse_dates=['date']).set_index('date').sort_index()\ndaily['ma7'] = daily['value'].rolling(window=7, min_periods=1).mean()\n```\n`min_periods=1` keeps the early days populated; otherwise the first six values are NaN.")

    add("data",
        "How do I group a DataFrame by month and aggregate?",
        "```python\nimport pandas as pd\n\ndf = pd.read_csv('sales.csv', parse_dates=['date'])\nmonthly = (\n    df.groupby(pd.Grouper(key='date', freq='MS'))\n      .agg(revenue=('amount', 'sum'), orders=('id', 'count'))\n      .reset_index()\n)\n```\n`freq='MS'` aligns to month-start; named aggregations keep the resulting columns readable.")

    add("data",
        "Write a numpy snippet that computes pairwise Euclidean distances.",
        "```python\nimport numpy as np\n\ndef pairwise_distances(a: np.ndarray) -> np.ndarray:\n    diff = a[:, None, :] - a[None, :, :]\n    return np.sqrt((diff ** 2).sum(axis=-1))\n```\nFor large `a` use `scipy.spatial.distance.cdist` -- same answer but less memory because it doesn't materialize the full diff tensor.")

    add("data",
        "How do I plot a histogram with matplotlib?",
        "```python\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ndata = np.random.normal(size=1000)\nfig, ax = plt.subplots()\nax.hist(data, bins=30, edgecolor='black')\nax.set_xlabel('Value')\nax.set_ylabel('Count')\nax.set_title('Distribution')\nfig.tight_layout()\nfig.savefig('hist.png', dpi=150)\n```\nAlways create explicit `fig, ax`; the global `plt.hist(...)` API trips you up in notebooks.")

    add("data",
        "Write a function that one-hot encodes a categorical column.",
        "```python\nimport pandas as pd\n\ndef encode(df: pd.DataFrame, col: str) -> pd.DataFrame:\n    return pd.get_dummies(df, columns=[col], drop_first=False)\n```\n`drop_first=True` for linear models to avoid the dummy-variable trap; for tree models keep all dummies.")

    add("data",
        "How do I merge two DataFrames on a column with overlapping names?",
        "```python\nimport pandas as pd\n\nresult = pd.merge(\n    a, b, on='user_id', suffixes=('_a', '_b'), how='left', validate='one_to_one'\n)\n```\n`validate='one_to_one'` raises if the assumption is wrong -- catches bad joins in development.")

    add("data",
        "Write code that reads a Parquet file partitioned by date.",
        "```python\nimport pandas as pd\n\ndf = pd.read_parquet('s3://bucket/events/', filters=[('date', '>=', '2026-01-01')])\n```\nThe `filters=` predicate is pushed down to the scanner -- only matching partitions are read off disk.")

    add("data",
        "How do I detect outliers in a numpy array using IQR?",
        "```python\nimport numpy as np\n\ndef iqr_mask(a: np.ndarray, k: float = 1.5) -> np.ndarray:\n    q1, q3 = np.percentile(a, [25, 75])\n    iqr = q3 - q1\n    return (a < q1 - k * iqr) | (a > q3 + k * iqr)\n```\nReturns a boolean mask; flip with `~` to keep inliers.")

    add("data",
        "Write a pandas pipeline that fills missing values then standardises numeric columns.",
        "```python\nimport pandas as pd\n\nnumeric = df.select_dtypes(include='number').columns\ndf[numeric] = df[numeric].fillna(df[numeric].median())\ndf[numeric] = (df[numeric] - df[numeric].mean()) / df[numeric].std(ddof=0)\n```\nFor a reusable pipeline wrap this in `sklearn.preprocessing.StandardScaler` so the same transformation can be applied to test data.")

    add("data",
        "How do I plot two series on a shared time axis in matplotlib?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nax.plot(df.index, df['revenue'], label='revenue', color='C0')\nax2 = ax.twinx()\nax2.plot(df.index, df['ad_spend'], label='ad spend', color='C1')\nax.set_ylabel('Revenue')\nax2.set_ylabel('Ad spend')\nfig.legend(loc='upper left')\nfig.autofmt_xdate()\n```\n`twinx()` keeps the time axis shared while letting each series have its own y-scale.")

    add("data",
        "Write a numpy snippet that computes a cumulative product.",
        "```python\nimport numpy as np\n\nx = np.array([1.0, 1.05, 0.97, 1.02])\ncumprod = np.cumprod(x)\n```\nFor returns, this gives the equity curve. To start at 1.0 use `np.concatenate(([1.0], cumprod))`.")

    add("data",
        "How do I aggregate two columns differently in a pandas groupby?",
        "```python\nimport pandas as pd\n\nresult = df.groupby('region').agg(\n    total_sales=('amount', 'sum'),\n    orders=('id', 'count'),\n    last_order=('date', 'max'),\n)\n```\nNamed aggregations make the output columns self-documenting.")

    add("data",
        "Write code that splits a DataFrame into train and test sets.",
        "```python\nfrom sklearn.model_selection import train_test_split\n\nX = df.drop(columns=['target'])\ny = df['target']\nX_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)\n```\n`stratify=y` keeps class proportions consistent; pin `random_state` for reproducibility.")

    add("data",
        "How do I compute the correlation between numeric columns?",
        "```python\nimport pandas as pd\n\ncorr = df.select_dtypes(include='number').corr(method='pearson')\n```\nFor non-linear relationships use Spearman or Kendall; correlation does not imply causation.")

    add("data",
        "Write a numpy snippet that creates a 2D Gaussian kernel.",
        "```python\nimport numpy as np\n\ndef gaussian_kernel(size: int, sigma: float) -> np.ndarray:\n    ax = np.arange(-size // 2 + 1, size // 2 + 1)\n    xx, yy = np.meshgrid(ax, ax)\n    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))\n    return k / k.sum()\n```\nNormalising by the sum keeps brightness preserved when convolving an image.")

    add("data",
        "How do I write a DataFrame to multiple Parquet files partitioned by a column?",
        "```python\nimport pandas as pd\n\ndf.to_parquet('out/', partition_cols=['year', 'month'])\n```\nPartitioning lets downstream readers prune what they load. Avoid partitioning on high-cardinality columns -- you'll create thousands of tiny files.")

    add("data",
        "Write a pandas snippet that computes the percentage change month over month.",
        "```python\nimport pandas as pd\n\nmonthly = df.set_index('date').resample('MS')['revenue'].sum()\nmonthly_pct = monthly.pct_change() * 100\n```\n`pct_change()` returns NaN for the first row -- expected, there's nothing to compare against.")

    add("data",
        "How do I save a matplotlib figure as a high-DPI PNG?",
        "```python\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots(figsize=(8, 4))\nax.plot([1, 2, 3], [1, 4, 9])\nfig.tight_layout()\nfig.savefig('plot.png', dpi=300, bbox_inches='tight')\nplt.close(fig)\n```\n`bbox_inches='tight'` trims whitespace; `plt.close(fig)` matters if you generate many figures in a loop (avoids memory growth).")

    add("data",
        "Write a snippet that computes z-scores for each column of a DataFrame.",
        "```python\nimport pandas as pd\n\nnumeric = df.select_dtypes(include='number')\nz = (numeric - numeric.mean()) / numeric.std(ddof=0)\n```\n`ddof=0` matches the population formula (`scipy.stats.zscore`'s default); use `ddof=1` for a sample estimator.")

    add("data",
        "How do I convert a numpy array to a pandas DataFrame with column names?",
        "```python\nimport numpy as np\nimport pandas as pd\n\narr = np.random.rand(3, 4)\ndf = pd.DataFrame(arr, columns=['a', 'b', 'c', 'd'])\n```\nThe column count must match `arr.shape[1]`; otherwise pandas raises a `ValueError`.")

    add("data",
        "Write a pandas snippet that pivots a long DataFrame to wide.",
        "```python\nimport pandas as pd\n\nwide = df.pivot_table(\n    index='date', columns='product', values='units', aggfunc='sum', fill_value=0,\n)\n```\n`pivot_table` (vs `pivot`) tolerates duplicate `(index, columns)` pairs by aggregating them.")

    add("data",
        "How do I sample rows from a large DataFrame reproducibly?",
        "```python\nimport pandas as pd\n\nsample = df.sample(n=1000, random_state=42)\n```\nFor a fraction of rows pass `frac=0.1`; for stratified samples, group by the stratification column first.")

    add("data",
        "Write a snippet that computes a confusion matrix with sklearn.",
        "```python\nfrom sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\nimport matplotlib.pyplot as plt\n\ncm = confusion_matrix(y_true, y_pred, labels=labels)\nfig, ax = plt.subplots()\nConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)\nfig.tight_layout()\n```\nPass `labels=` explicitly so the matrix order is deterministic across runs.")

    add("data",
        "How do I plot a ROC curve with sklearn and matplotlib?",
        "```python\nfrom sklearn.metrics import roc_curve, roc_auc_score\nimport matplotlib.pyplot as plt\n\nfpr, tpr, _ = roc_curve(y_true, y_score)\nauc = roc_auc_score(y_true, y_score)\nfig, ax = plt.subplots()\nax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')\nax.plot([0, 1], [0, 1], 'k--', alpha=0.5)\nax.set_xlabel('False positive rate')\nax.set_ylabel('True positive rate')\nax.legend()\nfig.tight_layout()\n```\nWhen classes are imbalanced, prefer precision/recall curves -- ROC can look misleadingly good.")
