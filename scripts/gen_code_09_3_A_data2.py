# SPDX-License-Identifier: MIT
"""Web / API entries (125 total). 50 of these have follow-ups."""


def register_all(add):
    # ---- 3-msg (75) ----
    add("web",
        "Write a Flask route that returns the current UTC time as JSON.",
        "```python\nfrom datetime import datetime, timezone\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.route('/time')\ndef now():\n    return jsonify(now=datetime.now(timezone.utc).isoformat())\n```\n`jsonify` sets `Content-Type: application/json` and serializes the dict for you.")

    add("web",
        "Write a FastAPI endpoint that accepts a JSON body {name: str} and returns a greeting.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI()\n\nclass Hello(BaseModel):\n    name: str\n\n@app.post('/greet')\ndef greet(body: Hello):\n    return {'message': f'Hello, {body.name}!'}\n```\nPydantic validates and parses the body before your handler runs.")

    add("web",
        "Write a function that fetches a JSON URL and returns the parsed dict using requests.",
        "```python\nimport requests\n\ndef fetch_json(url: str, timeout: float = 10.0) -> dict:\n    resp = requests.get(url, timeout=timeout)\n    resp.raise_for_status()\n    return resp.json()\n```\nAlways pass `timeout` — without it the request can hang indefinitely.")

    add("web",
        "Write a function that POSTs JSON and returns the response body.",
        "```python\nimport requests\n\ndef post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:\n    resp = requests.post(url, json=payload, timeout=timeout)\n    resp.raise_for_status()\n    return resp.json()\n```\n`json=` sets the body and the `Content-Type: application/json` header in one shot.")

    add("web",
        "Write a function that downloads a file from a URL to a local path.",
        "```python\nimport requests\nfrom pathlib import Path\n\ndef download(url: str, dest: str | Path, chunk: int = 1 << 14) -> None:\n    with requests.get(url, stream=True, timeout=30) as r:\n        r.raise_for_status()\n        with open(dest, 'wb') as f:\n            for piece in r.iter_content(chunk):\n                f.write(piece)\n```\n`stream=True` plus `iter_content` keeps memory usage flat regardless of file size.")

    add("web",
        "Write a Flask route that accepts a file upload and returns its size.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.post('/upload')\ndef upload():\n    f = request.files.get('file')\n    if not f:\n        return jsonify(error='no file'), 400\n    f.stream.seek(0, 2)\n    size = f.stream.tell()\n    return jsonify(name=f.filename, bytes=size)\n```\n`stream.seek(0, 2)` jumps to the end; `tell()` then gives the byte length without reading the contents.")

    add("web",
        "Write a FastAPI dependency that extracts a Bearer token from the Authorization header.",
        "```python\nfrom fastapi import Header, HTTPException\n\nasync def bearer_token(authorization: str = Header(...)) -> str:\n    if not authorization.startswith('Bearer '):\n        raise HTTPException(401, 'invalid auth scheme')\n    return authorization[len('Bearer '):]\n```\nUse it via `token: str = Depends(bearer_token)` on your endpoints.")

    add("web",
        "Write a function that builds a URL with query parameters from a base URL and a dict.",
        "```python\nfrom urllib.parse import urlencode, urlparse, urlunparse\n\ndef with_query(base: str, params: dict) -> str:\n    parsed = urlparse(base)\n    existing = parsed.query\n    new = urlencode(params, doseq=True)\n    merged = '&'.join(p for p in (existing, new) if p)\n    return urlunparse(parsed._replace(query=merged))\n```\n`doseq=True` lets list values become `?k=a&k=b` instead of `?k=['a','%20'b']`.")

    add("web",
        "Write a function that retries a GET request with exponential backoff on 5xx responses.",
        "```python\nimport time\nimport requests\n\ndef get_with_retry(url: str, attempts: int = 3) -> requests.Response:\n    last = None\n    for i in range(attempts):\n        last = requests.get(url, timeout=10)\n        if last.status_code < 500:\n            return last\n        time.sleep(0.5 * 2 ** i)\n    return last  # type: ignore[return-value]\n```\n5xx is retryable, 4xx generally is not. Real systems should also respect `Retry-After`.")

    add("web",
        "Write a function that parses a Link header (RFC 5988) into a dict of rel -> URL.",
        "```python\nimport re\n\ndef parse_link(header: str) -> dict[str, str]:\n    out: dict[str, str] = {}\n    for part in header.split(','):\n        m = re.match(r'\\s*<([^>]+)>;\\s*rel=\"([^\"]+)\"', part)\n        if m:\n            out[m.group(2)] = m.group(1)\n    return out\n```\nUseful for paginated APIs (GitHub, etc.). For full RFC 5988 you'd want a proper parser.")

    add("web",
        "Write a function that paginates a JSON API following 'next' links.",
        "```python\nimport requests\nfrom typing import Iterator\n\ndef paginate(url: str | None) -> Iterator[dict]:\n    while url:\n        r = requests.get(url, timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        yield from body.get('items', [])\n        url = body.get('next')\n```\nGenerator avoids buffering all pages. Stops when the API stops returning a `next`.")

    add("web",
        "Write a Flask middleware (using before_request) that logs each incoming request.",
        "```python\nimport logging\nfrom flask import Flask, request\n\napp = Flask(__name__)\nlog = logging.getLogger(__name__)\n\n@app.before_request\ndef log_request():\n    log.info('%s %s', request.method, request.path)\n```\nLog the method and path; remote address often comes from a header behind a proxy (`X-Forwarded-For`).")

    add("web",
        "Write a function that posts a multipart/form-data file upload using requests.",
        "```python\nimport requests\nfrom pathlib import Path\n\ndef upload(url: str, path: str | Path, field: str = 'file') -> requests.Response:\n    with open(path, 'rb') as f:\n        return requests.post(url, files={field: (Path(path).name, f)}, timeout=30)\n```\nThe `(filename, fileobj)` tuple lets requests send the original filename in the multipart payload.")

    add("web",
        "Write a function that validates a URL is HTTP(S) and well-formed.",
        "```python\nfrom urllib.parse import urlparse\n\ndef valid_http_url(url: str) -> bool:\n    p = urlparse(url)\n    return p.scheme in ('http', 'https') and bool(p.netloc)\n```\n`urlparse` is forgiving (it won't raise on garbage), so we explicitly check both pieces.")

    add("web",
        "Write a Flask route that returns a streaming CSV download.",
        "```python\nimport csv\nimport io\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\ndef rows():\n    yield ['id', 'name']\n    for i in range(1000):\n        yield [i, f'user{i}']\n\n@app.get('/users.csv')\ndef users_csv():\n    def stream():\n        buf = io.StringIO()\n        writer = csv.writer(buf)\n        for row in rows():\n            writer.writerow(row)\n            yield buf.getvalue()\n            buf.seek(0); buf.truncate(0)\n    return Response(stream(), mimetype='text/csv')\n```\nStreaming with a generator keeps memory flat; the buffer is rewound after each line.")

    add("web",
        "Write a FastAPI endpoint that streams server-sent events (SSE).",
        "```python\nimport asyncio\nfrom fastapi import FastAPI\nfrom fastapi.responses import StreamingResponse\n\napp = FastAPI()\n\nasync def event_gen():\n    for i in range(5):\n        await asyncio.sleep(1)\n        yield f'data: tick {i}\\n\\n'\n\n@app.get('/events')\nasync def events():\n    return StreamingResponse(event_gen(), media_type='text/event-stream')\n```\nThe `\\n\\n` separator and `text/event-stream` mimetype are what makes browsers treat the response as SSE.")

    add("web",
        "Write a function that signs a payload with HMAC-SHA256 for webhook verification.",
        "```python\nimport hmac\nimport hashlib\n\ndef sign(payload: bytes, secret: bytes) -> str:\n    return hmac.new(secret, payload, hashlib.sha256).hexdigest()\n\ndef verify(payload: bytes, secret: bytes, signature: str) -> bool:\n    expected = sign(payload, secret)\n    return hmac.compare_digest(expected, signature)\n```\nUse `hmac.compare_digest` (constant-time) — never `==` — to avoid timing attacks.")

    add("web",
        "Write a function that base64-encodes a file's contents for inline transmission.",
        "```python\nimport base64\nfrom pathlib import Path\n\ndef file_to_b64(path: str | Path) -> str:\n    return base64.b64encode(Path(path).read_bytes()).decode('ascii')\n```\nReading the whole file is fine for small attachments; for big files, stream and encode in chunks.")

    add("web",
        "Write a Flask route that responds with 404 if a user isn't found.",
        "```python\nfrom flask import Flask, abort, jsonify\n\napp = Flask(__name__)\n\nUSERS = {'1': {'name': 'Alice'}}\n\n@app.get('/users/<uid>')\ndef get_user(uid: str):\n    user = USERS.get(uid)\n    if user is None:\n        abort(404)\n    return jsonify(user)\n```\n`abort(404)` triggers Flask's standard 404 page; for a JSON-only API, register an error handler that returns JSON instead.")

    add("web",
        "Write a FastAPI endpoint that returns 404 if an item isn't found.",
        "```python\nfrom fastapi import FastAPI, HTTPException\n\napp = FastAPI()\nITEMS = {1: {'name': 'widget'}}\n\n@app.get('/items/{item_id}')\ndef get_item(item_id: int):\n    if item_id not in ITEMS:\n        raise HTTPException(status_code=404, detail='item not found')\n    return ITEMS[item_id]\n```\nFastAPI auto-renders `HTTPException` as JSON `{detail: ...}`.")

    add("web",
        "Write a function that issues parallel HTTP GETs using concurrent.futures.",
        "```python\nimport requests\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef fetch_many(urls: list[str], workers: int = 8) -> list[requests.Response]:\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        return list(ex.map(lambda u: requests.get(u, timeout=10), urls))\n```\nThreads work well here because requests releases the GIL while waiting on I/O.")

    add("web",
        "Write a function that issues parallel HTTP GETs using asyncio + httpx.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_many(urls: list[str]) -> list[str]:\n    async with httpx.AsyncClient(timeout=10) as client:\n        results = await asyncio.gather(*(client.get(u) for u in urls))\n        return [r.text for r in results]\n```\nReusing one `AsyncClient` is critical: connection pooling cuts TLS handshake overhead drastically.")

    add("web",
        "Write a Flask route that requires an API key in the X-API-Key header.",
        "```python\nimport os\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nAPI_KEY = os.environ['API_KEY']\n\n@app.before_request\ndef check_key():\n    if request.headers.get('X-API-Key') != API_KEY:\n        return jsonify(error='unauthorized'), 401\n```\nUse `before_request` so the check applies to every route. Reject early to avoid running handlers under bad auth.")

    add("web",
        "Write a function that uploads bytes to S3 using boto3.",
        "```python\nimport boto3\n\ndef put_object(bucket: str, key: str, data: bytes, content_type: str = 'application/octet-stream') -> None:\n    s3 = boto3.client('s3')\n    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)\n```\nThe boto3 client picks up credentials from env vars, ~/.aws/credentials, or instance metadata.")

    add("web",
        "Write a function that creates a presigned S3 URL valid for 1 hour.",
        "```python\nimport boto3\n\ndef presign(bucket: str, key: str) -> str:\n    s3 = boto3.client('s3')\n    return s3.generate_presigned_url(\n        'get_object',\n        Params={'Bucket': bucket, 'Key': key},\n        ExpiresIn=3600,\n    )\n```\nPresigned URLs let clients download directly from S3 without your server proxying the bytes.")

    add("web",
        "Write a function that decodes a JWT without verifying the signature (debug only).",
        "```python\nimport base64\nimport json\n\ndef peek_jwt(token: str) -> dict:\n    payload = token.split('.')[1]\n    padding = '=' * (-len(payload) % 4)\n    decoded = base64.urlsafe_b64decode(payload + padding)\n    return json.loads(decoded)\n```\nDebugging only — never trust an unverified token. Use the `jwt` library and a key for real verification.")

    add("web",
        "Write a function that POSTs JSON to a Slack webhook.",
        "```python\nimport requests\n\ndef notify_slack(webhook_url: str, text: str) -> None:\n    r = requests.post(webhook_url, json={'text': text}, timeout=5)\n    r.raise_for_status()\n```\nSlack returns 200 for accepted messages. A non-200 means malformed payload or revoked webhook.")

    add("web",
        "Write a function that constructs a URL with optional query parameters using urllib.",
        "```python\nfrom urllib.parse import urlencode\n\ndef build_url(base: str, **params) -> str:\n    qs = urlencode({k: v for k, v in params.items() if v is not None})\n    return f'{base}?{qs}' if qs else base\n```\nDropping `None` values avoids `?key=None` in the URL.")

    add("web",
        "Write a Flask route that returns gzipped JSON when the client accepts it.",
        "```python\nimport gzip\nimport json\nfrom flask import Flask, Response, request\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    body = json.dumps({'items': list(range(1000))}).encode()\n    if 'gzip' in request.headers.get('Accept-Encoding', ''):\n        body = gzip.compress(body)\n        return Response(body, mimetype='application/json',\n                        headers={'Content-Encoding': 'gzip'})\n    return Response(body, mimetype='application/json')\n```\nMost reverse proxies (nginx, Caddy) handle this for you; do it in-app only when you can't add one.")

    add("web",
        "Write a FastAPI WebSocket endpoint that echoes received messages.",
        "```python\nfrom fastapi import FastAPI, WebSocket, WebSocketDisconnect\n\napp = FastAPI()\n\n@app.websocket('/echo')\nasync def echo(ws: WebSocket):\n    await ws.accept()\n    try:\n        while True:\n            msg = await ws.receive_text()\n            await ws.send_text(msg)\n    except WebSocketDisconnect:\n        pass\n```\nCatching `WebSocketDisconnect` lets you exit cleanly when the client closes.")

    add("web",
        "Write a function that fetches a URL with a custom User-Agent header.",
        "```python\nimport requests\n\ndef fetch(url: str) -> str:\n    headers = {'User-Agent': 'MyBot/1.0 (https://example.com)'}\n    r = requests.get(url, headers=headers, timeout=10)\n    r.raise_for_status()\n    return r.text\n```\nMany sites block the default `python-requests/...` UA. A descriptive UA is also good etiquette.")

    add("web",
        "Write a function that posts form-encoded data (not JSON).",
        "```python\nimport requests\n\ndef post_form(url: str, data: dict) -> requests.Response:\n    return requests.post(url, data=data, timeout=10)\n```\nPassing `data=` (instead of `json=`) sends `application/x-www-form-urlencoded`.")

    add("web",
        "Write a function that follows HTTP redirects manually, capping at 5 hops.",
        "```python\nimport requests\n\ndef follow(url: str, max_hops: int = 5) -> requests.Response:\n    for _ in range(max_hops):\n        r = requests.get(url, allow_redirects=False, timeout=10)\n        if 300 <= r.status_code < 400 and 'Location' in r.headers:\n            url = r.headers['Location']\n            continue\n        return r\n    raise RuntimeError('too many redirects')\n```\nUseful when you want to inspect each redirect (e.g., to detect open redirects).")

    add("web",
        "Write a function that parses a 'Set-Cookie' header into a dict.",
        "```python\nfrom http.cookies import SimpleCookie\n\ndef parse_set_cookie(header: str) -> dict:\n    cookie = SimpleCookie()\n    cookie.load(header)\n    return {k: m.value for k, m in cookie.items()}\n```\nThe stdlib's `SimpleCookie` handles attributes like `Path`, `Domain`, etc. correctly.")

    add("web",
        "Write a Flask error handler for uncaught exceptions that returns JSON.",
        "```python\nimport logging\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\nlog = logging.getLogger(__name__)\n\n@app.errorhandler(Exception)\ndef on_error(e):\n    log.exception('unhandled')\n    return jsonify(error=str(e)), 500\n```\n`log.exception` captures the traceback. In production, surface a generic message and log internally.")

    add("web",
        "Write a function that tests if a port is open on a host.",
        "```python\nimport socket\n\ndef port_open(host: str, port: int, timeout: float = 2.0) -> bool:\n    try:\n        with socket.create_connection((host, port), timeout=timeout):\n            return True\n    except (OSError, socket.timeout):\n        return False\n```\nA successful connect means something is accepting on that port — not necessarily a healthy service.")

    add("web",
        "Write a FastAPI endpoint that returns a paginated list with skip/limit query params.",
        "```python\nfrom fastapi import FastAPI, Query\n\napp = FastAPI()\nITEMS = list(range(1000))\n\n@app.get('/items')\ndef list_items(skip: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)):\n    return {'items': ITEMS[skip:skip + limit], 'total': len(ITEMS)}\n```\n`Query(..., ge=, le=)` enforces bounds and FastAPI returns a 422 with helpful messages on violation.")

    add("web",
        "Write a function that downloads a URL with a tqdm progress bar.",
        "```python\nimport requests\nfrom tqdm import tqdm\n\ndef download(url: str, dest: str) -> None:\n    with requests.get(url, stream=True, timeout=30) as r:\n        r.raise_for_status()\n        total = int(r.headers.get('Content-Length', 0)) or None\n        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:\n            for chunk in r.iter_content(8192):\n                f.write(chunk)\n                bar.update(len(chunk))\n```\nIf the server doesn't send Content-Length, tqdm just shows transferred-bytes without a percentage.")

    add("web",
        "Write a Flask route that sets a session cookie.",
        "```python\nfrom flask import Flask, session\n\napp = Flask(__name__)\napp.secret_key = 'change-me-in-prod'\n\n@app.get('/login')\ndef login():\n    session['user'] = 'alice'\n    return 'logged in'\n```\nFlask's session is a signed cookie. `secret_key` is what makes the signature unforgeable.")

    add("web",
        "Write a function that builds a basic GraphQL query for a given type.",
        "```python\ndef build_query(type_name: str, fields: list[str]) -> str:\n    return f'query {{ {type_name} {{ {\" \".join(fields)} }} }}'\n```\nFor non-trivial queries, use a proper GraphQL client like `gql` to handle variables, fragments, etc.")

    add("web",
        "Write a function that validates an email address with a basic regex.",
        "```python\nimport re\n\n_EMAIL_RE = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$')\n\ndef looks_like_email(s: str) -> bool:\n    return bool(_EMAIL_RE.match(s))\n```\nRegex can never fully validate an email per RFC 5322. Use this as a syntax sniff and verify by sending a confirmation email.")

    add("web",
        "Write a Flask route that returns a CORS-friendly response.",
        "```python\nfrom flask import Flask, jsonify\nfrom flask_cors import CORS\n\napp = Flask(__name__)\nCORS(app, origins=['https://example.com'])\n\n@app.get('/api/data')\ndef data():\n    return jsonify(value=42)\n```\n`flask_cors` handles preflight OPTIONS requests too. Pin allowed origins instead of `'*'` in production.")

    add("web",
        "Write a function that constructs a SHA-256 ETag for a response body.",
        "```python\nimport hashlib\n\ndef etag(body: bytes) -> str:\n    return f'\"{hashlib.sha256(body).hexdigest()}\"'\n```\nETags must be quoted per RFC 7232. Strong ETags imply byte-identical content; weak ETags use a `W/` prefix.")

    add("web",
        "Write a FastAPI app with CORS configured for a specific origin.",
        "```python\nfrom fastapi import FastAPI\nfrom fastapi.middleware.cors import CORSMiddleware\n\napp = FastAPI()\napp.add_middleware(\n    CORSMiddleware,\n    allow_origins=['https://example.com'],\n    allow_methods=['*'],\n    allow_headers=['*'],\n)\n```\nList specific origins rather than `'*'` when you need cookies or auth headers — browsers reject `*` with credentials.")

    add("web",
        "Write a function that parses an RFC 7231 HTTP date string into a datetime.",
        "```python\nfrom email.utils import parsedate_to_datetime\nfrom datetime import datetime\n\ndef parse_http_date(s: str) -> datetime:\n    return parsedate_to_datetime(s)\n```\nThe stdlib's `email.utils` already handles HTTP date formats. Don't reinvent.")

    add("web",
        "Write a function that converts a datetime to an HTTP date string.",
        "```python\nfrom email.utils import format_datetime\nfrom datetime import datetime, timezone\n\ndef to_http_date(dt: datetime) -> str:\n    if dt.tzinfo is None:\n        dt = dt.replace(tzinfo=timezone.utc)\n    return format_datetime(dt, usegmt=True)\n```\n`usegmt=True` produces the RFC 7231-canonical 'GMT' suffix for UTC timestamps.")

    add("web",
        "Write a function that POSTs JSON with an Authorization: Bearer token.",
        "```python\nimport requests\n\ndef authed_post(url: str, token: str, payload: dict) -> dict:\n    headers = {'Authorization': f'Bearer {token}'}\n    r = requests.post(url, json=payload, headers=headers, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nKeep the token out of logs; redact `Authorization` from any logged headers.")

    add("web",
        "Write a function that parses a robots.txt and returns allow/disallow rules for a user-agent.",
        "```python\nfrom urllib.robotparser import RobotFileParser\n\ndef can_fetch(robots_url: str, user_agent: str, target_url: str) -> bool:\n    rp = RobotFileParser()\n    rp.set_url(robots_url)\n    rp.read()\n    return rp.can_fetch(user_agent, target_url)\n```\nThe stdlib's parser handles wildcards and crawl-delay correctly.")

    add("web",
        "Write a Flask route that handles a GET with optional query parameters.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.get('/search')\ndef search():\n    q = request.args.get('q', '')\n    limit = int(request.args.get('limit', 10))\n    return jsonify(query=q, limit=limit, results=[])\n```\nUse `request.args.get(key, default)` to avoid `KeyError` on missing params. For complex schemas, prefer pydantic.")

    add("web",
        "Write a function that fetches a URL behind HTTP Basic auth.",
        "```python\nimport requests\nfrom requests.auth import HTTPBasicAuth\n\ndef fetch_basic(url: str, user: str, password: str) -> requests.Response:\n    return requests.get(url, auth=HTTPBasicAuth(user, password), timeout=10)\n```\n`requests.get(..., auth=(user, password))` works as a shortcut, but the explicit form documents intent.")

    add("web",
        "Write a function that connects to a Postgres DB and runs a parameterized query.",
        "```python\nimport psycopg2\n\ndef get_user(dsn: str, user_id: int) -> dict | None:\n    with psycopg2.connect(dsn) as conn, conn.cursor() as cur:\n        cur.execute('SELECT id, name FROM users WHERE id = %s', (user_id,))\n        row = cur.fetchone()\n        return {'id': row[0], 'name': row[1]} if row else None\n```\n%s is psycopg2's parameter placeholder — never use string formatting (SQL injection risk).")

    add("web",
        "Write a function that reads from Redis using redis-py.",
        "```python\nimport redis\n\ndef get_value(host: str, key: str) -> str | None:\n    r = redis.Redis(host=host, decode_responses=True)\n    return r.get(key)\n```\n`decode_responses=True` returns `str` instead of `bytes` — usually what you want.")

    add("web",
        "Write a function that publishes a message to RabbitMQ using pika.",
        "```python\nimport pika\n\ndef publish(amqp_url: str, queue: str, body: bytes) -> None:\n    params = pika.URLParameters(amqp_url)\n    with pika.BlockingConnection(params) as conn:\n        ch = conn.channel()\n        ch.queue_declare(queue=queue, durable=True)\n        ch.basic_publish(exchange='', routing_key=queue, body=body,\n                          properties=pika.BasicProperties(delivery_mode=2))\n```\n`delivery_mode=2` makes the message persistent so it survives a broker restart.")

    add("web",
        "Write a function that sends an email via SMTP with TLS.",
        "```python\nimport smtplib\nfrom email.message import EmailMessage\n\ndef send_email(host: str, port: int, user: str, password: str,\n               sender: str, to: str, subject: str, body: str) -> None:\n    msg = EmailMessage()\n    msg['From'] = sender\n    msg['To'] = to\n    msg['Subject'] = subject\n    msg.set_content(body)\n    with smtplib.SMTP(host, port) as s:\n        s.starttls()\n        s.login(user, password)\n        s.send_message(msg)\n```\n`EmailMessage` (newer than `MIMEText`) handles encoding and headers correctly.")

    add("web",
        "Write a function that posts a payload to a webhook with HMAC signature.",
        "```python\nimport hmac\nimport hashlib\nimport json\nimport requests\n\ndef post_signed(url: str, secret: bytes, payload: dict) -> requests.Response:\n    body = json.dumps(payload, separators=(',', ':')).encode()\n    sig = hmac.new(secret, body, hashlib.sha256).hexdigest()\n    headers = {'X-Signature': f'sha256={sig}', 'Content-Type': 'application/json'}\n    return requests.post(url, data=body, headers=headers, timeout=10)\n```\nUsing `data=body` (not `json=`) keeps the bytes the receiver verifies identical to what we signed.")

    add("web",
        "Write a function that uses requests.Session for connection pooling across multiple calls.",
        "```python\nimport requests\n\ndef fetch_all(urls: list[str]) -> list[str]:\n    with requests.Session() as session:\n        return [session.get(u, timeout=10).text for u in urls]\n```\nA `Session` reuses TCP/TLS connections across requests — much faster for many calls to the same host.")

    add("web",
        "Write a Flask route that returns 304 Not Modified when If-None-Match matches.",
        "```python\nfrom flask import Flask, request, Response, jsonify\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    payload = jsonify(value=42)\n    payload.set_etag('v1')\n    return payload.make_conditional(request)\n```\n`make_conditional` automatically returns 304 when the request's `If-None-Match` matches the response's ETag.")

    add("web",
        "Write a function that opens an HTTP/2 connection using httpx.",
        "```python\nimport httpx\n\ndef http2_get(url: str) -> str:\n    with httpx.Client(http2=True, timeout=10) as client:\n        return client.get(url).text\n```\n`http2=True` requires the `h2` package. Multiplexed requests on a single connection are great for many parallel small calls.")

    add("web",
        "Write a function that converts a list of users to an XML response body.",
        "```python\nfrom xml.etree.ElementTree import Element, SubElement, tostring\n\ndef users_to_xml(users: list[dict]) -> bytes:\n    root = Element('users')\n    for u in users:\n        el = SubElement(root, 'user')\n        for k, v in u.items():\n            child = SubElement(el, k)\n            child.text = str(v)\n    return tostring(root, encoding='utf-8', xml_declaration=True)\n```\nThe stdlib `xml.etree.ElementTree` is enough for simple XML. Use `defusedxml` when parsing untrusted input.")

    add("web",
        "Write a FastAPI endpoint that uses background tasks to send a confirmation email.",
        "```python\nfrom fastapi import BackgroundTasks, FastAPI\n\napp = FastAPI()\n\ndef send_email(to: str) -> None:\n    pass  # actual send logic\n\n@app.post('/signup')\ndef signup(email: str, bg: BackgroundTasks):\n    bg.add_task(send_email, email)\n    return {'queued': True}\n```\nBackgroundTasks run after the response is returned, keeping latency low.")

    add("web",
        "Write a function that validates a JSON body against a JSON Schema.",
        "```python\nfrom jsonschema import validate, ValidationError\n\ndef is_valid(data, schema) -> tuple[bool, str]:\n    try:\n        validate(data, schema)\n        return True, ''\n    except ValidationError as e:\n        return False, e.message\n```\n`jsonschema` follows the standard. For higher throughput, compile a `Validator` once and reuse it.")

    add("web",
        "Write a function that constructs a curl command equivalent of a requests call (for debugging).",
        "```python\nimport shlex\n\ndef to_curl(method: str, url: str, headers: dict | None = None, body: str | None = None) -> str:\n    parts = ['curl', '-X', method, shlex.quote(url)]\n    for k, v in (headers or {}).items():\n        parts += ['-H', shlex.quote(f'{k}: {v}')]\n    if body is not None:\n        parts += ['--data', shlex.quote(body)]\n    return ' '.join(parts)\n```\n`shlex.quote` makes the output safe to paste into a shell.")

    add("web",
        "Write a function that returns a list of hyperlinks from an HTML page.",
        "```python\nfrom bs4 import BeautifulSoup\nimport requests\n\ndef get_links(url: str) -> list[str]:\n    html = requests.get(url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    return [a['href'] for a in soup.find_all('a', href=True)]\n```\n`BeautifulSoup` with the stdlib parser keeps the dependency footprint small. `lxml` is faster if you have it.")

    add("web",
        "Write a Flask app with a health-check endpoint at /healthz.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/healthz')\ndef health():\n    return jsonify(status='ok')\n```\nKubernetes and most load balancers expect a fast, side-effect-free 200 from a health endpoint.")

    add("web",
        "Write a function that issues an authenticated GraphQL query using requests.",
        "```python\nimport requests\n\ndef graphql(url: str, token: str, query: str, variables: dict | None = None) -> dict:\n    headers = {'Authorization': f'Bearer {token}'}\n    r = requests.post(url, json={'query': query, 'variables': variables or {}},\n                      headers=headers, timeout=15)\n    r.raise_for_status()\n    body = r.json()\n    if 'errors' in body:\n        raise RuntimeError(body['errors'])\n    return body['data']\n```\nGraphQL APIs return 200 even for errors, so we have to inspect the body.")

    add("web",
        "Write a function that schedules a task to run periodically using APScheduler.",
        "```python\nfrom apscheduler.schedulers.background import BackgroundScheduler\n\ndef start_scheduler():\n    sched = BackgroundScheduler()\n    sched.add_job(lambda: print('tick'), 'interval', seconds=10)\n    sched.start()\n    return sched\n```\nFor more durability, swap the in-memory job store for a Redis/SQLAlchemy-backed one.")

    add("web",
        "Write a function that converts a dict to a query string suitable for a URL.",
        "```python\nfrom urllib.parse import urlencode\n\ndef to_qs(params: dict) -> str:\n    return urlencode(params, doseq=True)\n```\n`doseq=True` expands list values into repeated keys, which most servers expect.")

    add("web",
        "Write a function that issues an HTTP DELETE request.",
        "```python\nimport requests\n\ndef delete(url: str, token: str | None = None) -> int:\n    headers = {'Authorization': f'Bearer {token}'} if token else {}\n    r = requests.delete(url, headers=headers, timeout=10)\n    return r.status_code\n```\nReturning the status code lets the caller decide what 'success' means (some APIs return 204, some 200, some 202).")

    add("web",
        "Write a Flask route that streams a large file to the client without loading it all in memory.",
        "```python\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/big.bin')\ndef big():\n    def gen():\n        with open('/path/to/big.bin', 'rb') as f:\n            while True:\n                chunk = f.read(64 * 1024)\n                if not chunk:\n                    break\n                yield chunk\n    return Response(gen(), mimetype='application/octet-stream')\n```\nStreaming via a generator keeps memory flat regardless of file size.")

    add("web",
        "Write a function that escapes user-supplied text for safe inclusion in HTML.",
        "```python\nimport html\n\ndef escape(s: str) -> str:\n    return html.escape(s, quote=True)\n```\nThe stdlib's `html.escape` handles `< > & \" '`. For attribute contexts, the `quote=True` is essential.")

    add("web",
        "Write a FastAPI endpoint that returns 422 on a Pydantic validation error.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, Field\n\napp = FastAPI()\n\nclass Item(BaseModel):\n    name: str = Field(min_length=1, max_length=100)\n    qty: int = Field(gt=0)\n\n@app.post('/items')\ndef create(item: Item):\n    return item\n```\nFastAPI converts Pydantic ValidationErrors to 422 responses automatically — no manual handling needed.")

    add("web",
        "Write a function that loads an OpenAPI spec from disk and returns the list of operation IDs.",
        "```python\nimport yaml\nfrom pathlib import Path\n\ndef operation_ids(path: str) -> list[str]:\n    spec = yaml.safe_load(Path(path).read_text())\n    out = []\n    for _path, methods in spec.get('paths', {}).items():\n        for _method, op in methods.items():\n            if isinstance(op, dict) and 'operationId' in op:\n                out.append(op['operationId'])\n    return out\n```\nUseful for generating client stubs or docs from a spec.")

    add("web",
        "Write a function that proxies a request body to another URL.",
        "```python\nimport requests\nfrom flask import Flask, request, Response\n\napp = Flask(__name__)\n\n@app.post('/proxy')\ndef proxy():\n    upstream = requests.post(\n        'https://upstream.example/api',\n        data=request.get_data(),\n        headers={k: v for k, v in request.headers if k.lower() != 'host'},\n        timeout=30,\n    )\n    return Response(upstream.content, status=upstream.status_code,\n                    headers=dict(upstream.headers))\n```\nStrip the `Host` header — your upstream needs its own. Mind hop-by-hop headers (`Connection`, etc.) for production proxies.")

    add("web",
        "Write a function that fetches Twitter/X-style cursor-paginated results.",
        "```python\nimport requests\n\ndef paginate_cursor(url: str, headers: dict | None = None):\n    cursor = None\n    while True:\n        params = {'cursor': cursor} if cursor else {}\n        r = requests.get(url, params=params, headers=headers, timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        yield from body.get('data', [])\n        cursor = body.get('next_cursor')\n        if not cursor:\n            break\n```\nGenerators make pagination feel like one big iterable to consumers.")

    add("web",
        "Write a function that constructs a signed S3 v4 PUT URL using boto3.",
        "```python\nimport boto3\n\ndef put_url(bucket: str, key: str, expires: int = 3600, content_type: str = 'application/octet-stream') -> str:\n    s3 = boto3.client('s3')\n    return s3.generate_presigned_url(\n        'put_object',\n        Params={'Bucket': bucket, 'Key': key, 'ContentType': content_type},\n        ExpiresIn=expires,\n        HttpMethod='PUT',\n    )\n```\nClients PUT directly to S3 with this URL — useful for browser uploads bypassing your server.")

    # ---- 5-msg with follow-ups (50) ----
    add("web",
        "Write a Flask route that accepts JSON and stores it in a list.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nstore: list[dict] = []\n\n@app.post('/items')\ndef create():\n    data = request.get_json()\n    store.append(data)\n    return jsonify(data), 201\n```\n`request.get_json()` parses the body. Returning 201 signals 'created'.",
        "Make it reject requests where 'name' is missing.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nstore: list[dict] = []\n\n@app.post('/items')\ndef create():\n    data = request.get_json(silent=True) or {}\n    if 'name' not in data:\n        return jsonify(error=\"'name' is required\"), 400\n    store.append(data)\n    return jsonify(data), 201\n```\n`silent=True` returns `None` instead of raising on bad JSON, so we can produce a clean 400.")

    add("web",
        "Write a function that fetches a URL and returns the response.",
        "```python\nimport requests\n\ndef fetch(url: str) -> str:\n    return requests.get(url).text\n```\nMinimal happy-path version.",
        "Add proper error handling and a timeout.",
        "```python\nimport requests\n\ndef fetch(url: str, timeout: float = 10.0) -> str:\n    try:\n        r = requests.get(url, timeout=timeout)\n        r.raise_for_status()\n        return r.text\n    except requests.RequestException as e:\n        raise RuntimeError(f'fetch failed: {e}') from e\n```\nA timeout is non-negotiable for production. `RequestException` is the umbrella class for connection / HTTP / timeout errors.")

    add("web",
        "Write a FastAPI endpoint that creates a user.",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, EmailStr\n\napp = FastAPI()\n\nclass UserIn(BaseModel):\n    name: str\n    email: EmailStr\n\n@app.post('/users')\ndef create_user(user: UserIn):\n    return {'id': 1, **user.dict()}\n```\n`EmailStr` validates email syntax. `user.dict()` converts the model to a dict.",
        "How would I separate input and output models so the response includes the id but the request doesn't accept it?",
        "```python\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel, EmailStr\n\napp = FastAPI()\n\nclass UserIn(BaseModel):\n    name: str\n    email: EmailStr\n\nclass UserOut(UserIn):\n    id: int\n\n@app.post('/users', response_model=UserOut)\ndef create_user(user: UserIn) -> UserOut:\n    return UserOut(id=1, **user.dict())\n```\n`response_model` makes FastAPI strip any extra fields the function might return — a defense-in-depth against accidentally leaking internal fields.")

    add("web",
        "Write a function that returns the IP address of a hostname.",
        "```python\nimport socket\n\ndef resolve(host: str) -> str:\n    return socket.gethostbyname(host)\n```\nReturns the first IPv4 address.",
        "I need IPv6 too, and all addresses, not just one.",
        "```python\nimport socket\n\ndef resolve(host: str) -> list[str]:\n    infos = socket.getaddrinfo(host, None)\n    return sorted({info[4][0] for info in infos})\n```\n`getaddrinfo` returns both v4 and v6, deduplicated and sorted for stability.")

    add("web",
        "Write a Flask app with two routes: GET /items and GET /items/<id>.",
        "```python\nfrom flask import Flask, jsonify, abort\n\napp = Flask(__name__)\nITEMS = {1: {'id': 1, 'name': 'a'}, 2: {'id': 2, 'name': 'b'}}\n\n@app.get('/items')\ndef list_items():\n    return jsonify(list(ITEMS.values()))\n\n@app.get('/items/<int:id>')\ndef get_item(id: int):\n    item = ITEMS.get(id)\n    if not item:\n        abort(404)\n    return jsonify(item)\n```\nFlask's `<int:id>` converter validates and casts the path param.",
        "Add a POST /items endpoint that creates an item.",
        "```python\nfrom flask import Flask, jsonify, abort, request\n\napp = Flask(__name__)\nITEMS = {1: {'id': 1, 'name': 'a'}}\n\n@app.post('/items')\ndef create_item():\n    body = request.get_json(silent=True) or {}\n    if 'name' not in body:\n        return jsonify(error=\"'name' required\"), 400\n    new_id = max(ITEMS, default=0) + 1\n    ITEMS[new_id] = {'id': new_id, 'name': body['name']}\n    return jsonify(ITEMS[new_id]), 201\n```\nUsing `max(ITEMS, default=0) + 1` is fine for an in-memory toy; for real systems use a DB sequence.")

    add("web",
        "Write a function that calls an API with a GET and returns the parsed JSON.",
        "```python\nimport requests\n\ndef api_get(url: str) -> dict:\n    return requests.get(url).json()\n```\nMinimal version without error handling.",
        "Add caching so identical calls within 60 seconds return without hitting the network.",
        "```python\nimport time\nimport requests\nfrom functools import lru_cache\n\n_CACHE: dict[str, tuple[float, dict]] = {}\n\ndef api_get(url: str, ttl: float = 60.0) -> dict:\n    now = time.time()\n    cached = _CACHE.get(url)\n    if cached and now - cached[0] < ttl:\n        return cached[1]\n    body = requests.get(url, timeout=10).json()\n    _CACHE[url] = (now, body)\n    return body\n```\n`lru_cache` doesn't expire entries, so we hand-roll a TTL cache. For multi-process work, use Redis.")

    add("web",
        "Write a function that downloads JSON from a URL and writes it to a file.",
        "```python\nimport json\nimport requests\n\ndef download_json(url: str, dest: str) -> None:\n    body = requests.get(url, timeout=10).json()\n    with open(dest, 'w', encoding='utf-8') as f:\n        json.dump(body, f, indent=2)\n```\n`indent=2` produces human-readable output.",
        "Make it append a timestamp to the filename so reruns don't overwrite.",
        "```python\nimport json\nimport requests\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n\ndef download_json(url: str, dest: str) -> Path:\n    body = requests.get(url, timeout=10).json()\n    p = Path(dest)\n    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n    final = p.with_name(f'{p.stem}-{stamp}{p.suffix}')\n    final.write_text(json.dumps(body, indent=2), encoding='utf-8')\n    return final\n```\nReturning the actual path lets the caller log or report where the file landed.")

    add("web",
        "Write a function that scrapes the title tag from an HTML page.",
        "```python\nimport requests\nfrom bs4 import BeautifulSoup\n\ndef get_title(url: str) -> str:\n    html = requests.get(url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    return soup.title.string if soup.title else ''\n```\nGrab whatever `<title>` contains. Some sites set it via JavaScript — those need a headless browser.",
        "What if I want the og:title meta tag instead, falling back to <title>?",
        "```python\nimport requests\nfrom bs4 import BeautifulSoup\n\ndef get_title(url: str) -> str:\n    html = requests.get(url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    og = soup.find('meta', property='og:title')\n    if og and og.get('content'):\n        return og['content']\n    return soup.title.string if soup.title else ''\n```\n`og:title` is what social link previews use, so it's often higher quality than the HTML `<title>`.")

    add("web",
        "Write a function that uploads a file to a URL via PUT.",
        "```python\nimport requests\n\ndef put_file(url: str, path: str) -> int:\n    with open(path, 'rb') as f:\n        return requests.put(url, data=f, timeout=60).status_code\n```\nPassing the file object as `data=` streams it without loading into memory.",
        "Add a Content-MD5 header so the server can verify integrity.",
        "```python\nimport base64\nimport hashlib\nimport requests\nfrom pathlib import Path\n\ndef put_file(url: str, path: str) -> int:\n    data = Path(path).read_bytes()\n    md5 = base64.b64encode(hashlib.md5(data).digest()).decode('ascii')\n    return requests.put(url, data=data, headers={'Content-MD5': md5}, timeout=60).status_code\n```\nMD5 must be base64-encoded per RFC 1864. We have to load the bytes to hash them.")

    add("web",
        "Write a function that polls a URL until it returns 200 or times out.",
        "```python\nimport time\nimport requests\n\ndef wait_for_200(url: str, timeout: float = 30.0, interval: float = 1.0) -> bool:\n    deadline = time.time() + timeout\n    while time.time() < deadline:\n        try:\n            if requests.get(url, timeout=2).status_code == 200:\n                return True\n        except requests.RequestException:\n            pass\n        time.sleep(interval)\n    return False\n```\nUseful for waiting on a service to come up in tests.",
        "Make the interval grow with each attempt (exponential backoff).",
        "```python\nimport time\nimport requests\n\ndef wait_for_200(url: str, timeout: float = 30.0, base: float = 0.5, cap: float = 5.0) -> bool:\n    deadline = time.time() + timeout\n    interval = base\n    while time.time() < deadline:\n        try:\n            if requests.get(url, timeout=2).status_code == 200:\n                return True\n        except requests.RequestException:\n            pass\n        time.sleep(interval)\n        interval = min(cap, interval * 2)\n    return False\n```\nCapping the interval prevents very long sleeps near the deadline.")

    add("web",
        "Write a function that loads a YAML config file.",
        "```python\nimport yaml\nfrom pathlib import Path\n\ndef load_config(path: str) -> dict:\n    return yaml.safe_load(Path(path).read_text())\n```\nUse `safe_load` (not `load`) — `load` can execute arbitrary Python.",
        "Allow the path to be optional, falling back to a default location.",
        "```python\nimport os\nimport yaml\nfrom pathlib import Path\n\nDEFAULT_PATHS = [\n    Path.cwd() / 'config.yaml',\n    Path.home() / '.config' / 'myapp' / 'config.yaml',\n]\n\ndef load_config(path: str | None = None) -> dict:\n    if path:\n        return yaml.safe_load(Path(path).read_text())\n    for candidate in DEFAULT_PATHS:\n        if candidate.is_file():\n            return yaml.safe_load(candidate.read_text())\n    raise FileNotFoundError('no config found')\n```\nFalling back to standard locations follows the XDG-ish pattern most CLIs use.")

    add("web",
        "Write a Flask route that returns paginated results from a list.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nDATA = list(range(1000))\n\n@app.get('/data')\ndef data():\n    page = int(request.args.get('page', 1))\n    size = int(request.args.get('size', 10))\n    start = (page - 1) * size\n    return jsonify(items=DATA[start:start + size], page=page, size=size)\n```\nSimple offset-pagination is fine for small data sets.",
        "Switch to cursor-based pagination using the last item as the cursor.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nDATA = list(range(1000))\n\n@app.get('/data')\ndef data():\n    after = request.args.get('after')\n    size = int(request.args.get('size', 10))\n    start = 0 if after is None else int(after) + 1\n    items = DATA[start:start + size]\n    next_cursor = items[-1] if len(items) == size else None\n    return jsonify(items=items, next=next_cursor)\n```\nCursor pagination is stable under inserts/deletes, unlike `page=N` which can skip or repeat items.")

    add("web",
        "Write an async function that fetches multiple URLs concurrently.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_all(urls: list[str]) -> list[str]:\n    async with httpx.AsyncClient(timeout=10) as client:\n        responses = await asyncio.gather(*(client.get(u) for u in urls))\n        return [r.text for r in responses]\n```\nOne client, many concurrent requests.",
        "Cap the concurrency at 10 even if I pass 1000 URLs.",
        "```python\nimport asyncio\nimport httpx\n\nasync def fetch_all(urls: list[str], max_concurrency: int = 10) -> list[str]:\n    sem = asyncio.Semaphore(max_concurrency)\n    async with httpx.AsyncClient(timeout=10) as client:\n        async def fetch(u):\n            async with sem:\n                return (await client.get(u)).text\n        return await asyncio.gather(*(fetch(u) for u in urls))\n```\n`asyncio.Semaphore` is the standard way to bound concurrency in async code.")

    add("web",
        "Write a function that converts a dict into URL query parameters.",
        "```python\nfrom urllib.parse import urlencode\n\ndef qs(params: dict) -> str:\n    return urlencode(params)\n```\nThe stdlib does the right thing.",
        "What if some values are lists?",
        "```python\nfrom urllib.parse import urlencode\n\ndef qs(params: dict) -> str:\n    return urlencode(params, doseq=True)\n```\n`doseq=True` expands list values into repeated keys (e.g., `tag=a&tag=b`).")

    add("web",
        "Write a Flask app that serves static files from a 'public' directory.",
        "```python\nfrom flask import Flask, send_from_directory\n\napp = Flask(__name__)\n\n@app.get('/static/<path:filename>')\ndef static_files(filename: str):\n    return send_from_directory('public', filename)\n```\n`send_from_directory` guards against path traversal attacks.",
        "Add cache headers so browsers cache for an hour.",
        "```python\nfrom flask import Flask, send_from_directory\n\napp = Flask(__name__)\n\n@app.get('/static/<path:filename>')\ndef static_files(filename: str):\n    resp = send_from_directory('public', filename)\n    resp.headers['Cache-Control'] = 'public, max-age=3600'\n    return resp\n```\nFor versioned asset URLs you can use `max-age=31536000, immutable` for aggressive caching.")

    add("web",
        "Write a function that authenticates against an OAuth client_credentials flow and returns a token.",
        "```python\nimport requests\n\ndef get_token(token_url: str, client_id: str, client_secret: str) -> str:\n    r = requests.post(token_url, data={\n        'grant_type': 'client_credentials',\n        'client_id': client_id,\n        'client_secret': client_secret,\n    }, timeout=10)\n    r.raise_for_status()\n    return r.json()['access_token']\n```\nclient_credentials is for server-to-server auth (no user involved).",
        "Add caching so we don't request a new token on every call.",
        "```python\nimport time\nimport requests\nfrom dataclasses import dataclass\n\n@dataclass\nclass _Cached:\n    token: str\n    expires_at: float\n\n_cache: dict[str, _Cached] = {}\n\ndef get_token(token_url: str, client_id: str, client_secret: str) -> str:\n    cached = _cache.get(client_id)\n    if cached and cached.expires_at > time.time() + 30:\n        return cached.token\n    r = requests.post(token_url, data={\n        'grant_type': 'client_credentials',\n        'client_id': client_id,\n        'client_secret': client_secret,\n    }, timeout=10)\n    r.raise_for_status()\n    body = r.json()\n    _cache[client_id] = _Cached(body['access_token'],\n                                time.time() + body.get('expires_in', 3600))\n    return body['access_token']\n```\nThe 30-second buffer means we refresh slightly early to avoid using a token that expires mid-request.")

    add("web",
        "Write a function that POSTs JSON and ignores the response.",
        "```python\nimport requests\n\ndef post_json(url: str, payload: dict) -> None:\n    requests.post(url, json=payload, timeout=10)\n```\nFire-and-forget. Note: this still blocks on the response — use a queue for true fire-and-forget.",
        "Make it raise on a non-2xx response.",
        "```python\nimport requests\n\ndef post_json(url: str, payload: dict) -> None:\n    r = requests.post(url, json=payload, timeout=10)\n    r.raise_for_status()\n```\n`raise_for_status` is a one-liner that turns 4xx/5xx into an exception.")

    add("web",
        "Write an async FastAPI endpoint that runs a long DB query.",
        "```python\nfrom fastapi import FastAPI\nimport asyncpg\n\napp = FastAPI()\n\n@app.get('/users/{uid}')\nasync def get_user(uid: int):\n    conn = await asyncpg.connect('postgresql://localhost/mydb')\n    try:\n        row = await conn.fetchrow('SELECT id, name FROM users WHERE id = $1', uid)\n        return dict(row) if row else None\n    finally:\n        await conn.close()\n```\nasyncpg is faster than psycopg2 for async workloads.",
        "Move the connection to a pool so we don't open one per request.",
        "```python\nfrom fastapi import FastAPI\nimport asyncpg\n\napp = FastAPI()\n\n@app.on_event('startup')\nasync def startup():\n    app.state.db = await asyncpg.create_pool('postgresql://localhost/mydb')\n\n@app.on_event('shutdown')\nasync def shutdown():\n    await app.state.db.close()\n\n@app.get('/users/{uid}')\nasync def get_user(uid: int):\n    async with app.state.db.acquire() as conn:\n        row = await conn.fetchrow('SELECT id, name FROM users WHERE id = $1', uid)\n        return dict(row) if row else None\n```\nA pool removes the per-request connection overhead and amortizes the TLS/auth cost.")

    add("web",
        "Write a function that parses an Accept header and returns the preferred media type.",
        "```python\ndef preferred_type(header: str, supported: list[str]) -> str | None:\n    parsed = []\n    for part in header.split(','):\n        item = part.strip()\n        q = 1.0\n        if ';' in item:\n            mt, *params = item.split(';')\n            for p in params:\n                k, _, v = p.strip().partition('=')\n                if k == 'q':\n                    try:\n                        q = float(v)\n                    except ValueError:\n                        pass\n            item = mt.strip()\n        parsed.append((item, q))\n    parsed.sort(key=lambda x: x[1], reverse=True)\n    for mt, _q in parsed:\n        if mt in supported:\n            return mt\n    return None\n```\nQ-value parsing per RFC 7231. Doesn't handle wildcards (`*/*`) — add that for full compliance.",
        "Add wildcard support like `text/*` and `*/*`.",
        "```python\nimport fnmatch\n\ndef preferred_type(header: str, supported: list[str]) -> str | None:\n    parsed = []\n    for part in header.split(','):\n        item = part.strip()\n        q = 1.0\n        if ';' in item:\n            mt, *params = item.split(';')\n            for p in params:\n                k, _, v = p.strip().partition('=')\n                if k == 'q':\n                    try:\n                        q = float(v)\n                    except ValueError:\n                        pass\n            item = mt.strip()\n        parsed.append((item, q))\n    parsed.sort(key=lambda x: x[1], reverse=True)\n    for pattern, _q in parsed:\n        for s in supported:\n            if fnmatch.fnmatch(s, pattern):\n                return s\n    return None\n```\n`fnmatch` matches `text/*` against `text/html`. The Accept header doesn't use full glob semantics, but `*` is the only special character we need.")

    add("web",
        "Write a function that fetches stock-quote-like data from a public API.",
        "```python\nimport requests\n\ndef get_quote(symbol: str) -> dict:\n    url = f'https://api.example.com/v1/quote/{symbol}'\n    r = requests.get(url, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nReplace `api.example.com` with the actual provider — Alpha Vantage, IEX, etc.",
        "Add a retry on rate-limit responses (429) with respect to Retry-After.",
        "```python\nimport time\nimport requests\n\ndef get_quote(symbol: str, attempts: int = 3) -> dict:\n    url = f'https://api.example.com/v1/quote/{symbol}'\n    for _ in range(attempts):\n        r = requests.get(url, timeout=10)\n        if r.status_code == 429:\n            wait = float(r.headers.get('Retry-After', 1))\n            time.sleep(wait)\n            continue\n        r.raise_for_status()\n        return r.json()\n    raise RuntimeError('rate-limited too many times')\n```\nHonoring `Retry-After` is the polite way to back off and what most APIs ask of clients.")

    add("web",
        "Write a Flask route that requires JSON content-type.",
        "```python\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.post('/items')\ndef create():\n    if not request.is_json:\n        return jsonify(error='expected application/json'), 415\n    return jsonify(received=request.get_json()), 201\n```\n415 Unsupported Media Type is the right code when the body type is wrong.",
        "Make it a reusable decorator.",
        "```python\nfrom functools import wraps\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\ndef require_json(fn):\n    @wraps(fn)\n    def wrapper(*args, **kwargs):\n        if not request.is_json:\n            return jsonify(error='expected application/json'), 415\n        return fn(*args, **kwargs)\n    return wrapper\n\n@app.post('/items')\n@require_json\ndef create():\n    return jsonify(received=request.get_json()), 201\n```\nDecorators are the right tool when the same precondition shows up across many routes.")

    add("web",
        "Write a function that streams a response body and counts bytes.",
        "```python\nimport requests\n\ndef stream_count(url: str) -> int:\n    total = 0\n    with requests.get(url, stream=True, timeout=30) as r:\n        for chunk in r.iter_content(8192):\n            total += len(chunk)\n    return total\n```\n`stream=True` plus `iter_content` keeps memory bounded.",
        "Add a callback called every 1 MB.",
        "```python\nimport requests\nfrom typing import Callable\n\ndef stream_count(url: str, on_progress: Callable[[int], None] | None = None) -> int:\n    total = 0\n    last_report = 0\n    step = 1 << 20  # 1 MB\n    with requests.get(url, stream=True, timeout=30) as r:\n        for chunk in r.iter_content(8192):\n            total += len(chunk)\n            if on_progress and total - last_report >= step:\n                on_progress(total)\n                last_report = total\n    return total\n```\nThrottling via `last_report` keeps the callback rate predictable regardless of chunk size.")

    add("web",
        "Write a function that issues an HTTP HEAD request to check if a URL exists.",
        "```python\nimport requests\n\ndef url_exists(url: str) -> bool:\n    try:\n        r = requests.head(url, allow_redirects=True, timeout=5)\n        return r.status_code < 400\n    except requests.RequestException:\n        return False\n```\nHEAD avoids downloading the body. `allow_redirects=True` follows 3xx chains.",
        "Some servers don't support HEAD properly — fall back to a small GET range.",
        "```python\nimport requests\n\ndef url_exists(url: str) -> bool:\n    try:\n        r = requests.head(url, allow_redirects=True, timeout=5)\n        if r.status_code == 405 or r.status_code == 501:\n            r = requests.get(url, headers={'Range': 'bytes=0-0'}, allow_redirects=True, timeout=5)\n        return r.status_code < 400\n    except requests.RequestException:\n        return False\n```\nThe `Range: bytes=0-0` request fetches a single byte if the server supports ranges, otherwise the full body — but most CDNs honor it.")

    add("web",
        "Write a function that makes an XML-RPC call.",
        "```python\nfrom xmlrpc.client import ServerProxy\n\ndef call_xmlrpc(url: str, method: str, *args):\n    proxy = ServerProxy(url, allow_none=True)\n    return getattr(proxy, method)(*args)\n```\n`allow_none=True` lets `None` round-trip as XML-RPC `<nil>`.",
        "How is JSON-RPC different and how would I do that?",
        "JSON-RPC is a simpler text protocol over HTTP that uses JSON envelopes (`{jsonrpc, method, params, id}`) instead of XML. There's no single stdlib client, but it's a one-liner with `requests`:\n```python\nimport requests\n\ndef call_jsonrpc(url: str, method: str, *args):\n    body = {'jsonrpc': '2.0', 'method': method, 'params': list(args), 'id': 1}\n    r = requests.post(url, json=body, timeout=10)\n    r.raise_for_status()\n    body = r.json()\n    if 'error' in body:\n        raise RuntimeError(body['error'])\n    return body['result']\n```\nXML-RPC is older, more verbose, and rarely seen in modern services; JSON-RPC is a drop-in replacement most of the time.")

    add("web",
        "Write a function that fetches a JSON file from GitHub raw.",
        "```python\nimport requests\n\ndef github_raw_json(user: str, repo: str, path: str, ref: str = 'main') -> dict:\n    url = f'https://raw.githubusercontent.com/{user}/{repo}/{ref}/{path}'\n    r = requests.get(url, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nRaw URLs return file contents directly — no API quota needed.",
        "How do I authenticate when the repo is private?",
        "```python\nimport os\nimport requests\n\ndef github_raw_json(user: str, repo: str, path: str, ref: str = 'main') -> dict:\n    token = os.environ['GITHUB_TOKEN']\n    headers = {\n        'Authorization': f'Bearer {token}',\n        'Accept': 'application/vnd.github.raw',\n    }\n    api = f'https://api.github.com/repos/{user}/{repo}/contents/{path}?ref={ref}'\n    r = requests.get(api, headers=headers, timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nFor private content go through the contents API with a token; the `application/vnd.github.raw` media type returns the file body directly.")

    add("web",
        "Write a function that pings a Slack channel via the Web API (chat.postMessage).",
        "```python\nimport os\nimport requests\n\ndef slack_post(channel: str, text: str) -> dict:\n    token = os.environ['SLACK_TOKEN']\n    r = requests.post(\n        'https://slack.com/api/chat.postMessage',\n        headers={'Authorization': f'Bearer {token}'},\n        json={'channel': channel, 'text': text},\n        timeout=10,\n    )\n    r.raise_for_status()\n    body = r.json()\n    if not body.get('ok'):\n        raise RuntimeError(body.get('error', 'unknown'))\n    return body\n```\nSlack returns 200 even on logical errors — always check the `ok` field.",
        "How do I send a Block Kit message instead of plain text?",
        "```python\nimport os\nimport requests\n\ndef slack_post_blocks(channel: str, blocks: list[dict], fallback: str) -> dict:\n    token = os.environ['SLACK_TOKEN']\n    r = requests.post(\n        'https://slack.com/api/chat.postMessage',\n        headers={'Authorization': f'Bearer {token}'},\n        json={'channel': channel, 'text': fallback, 'blocks': blocks},\n        timeout=10,\n    )\n    r.raise_for_status()\n    body = r.json()\n    if not body.get('ok'):\n        raise RuntimeError(body.get('error', 'unknown'))\n    return body\n```\nThe `text` field is still required as a fallback for clients (notifications, screen readers) that can't render blocks.")

    add("web",
        "Write a function that times out an HTTP request after 5 seconds.",
        "```python\nimport requests\n\ndef fetch(url: str) -> str:\n    return requests.get(url, timeout=5).text\n```\nA single timeout covers both connect and read.",
        "Use separate connect and read timeouts.",
        "```python\nimport requests\n\ndef fetch(url: str) -> str:\n    return requests.get(url, timeout=(2, 10)).text\n```\nThe tuple is `(connect, read)`. Useful when you can tolerate a slow body but not a slow handshake.")

    add("web",
        "Write a function that converts a date string from one format to another.",
        "```python\nfrom datetime import datetime\n\ndef reformat_date(s: str, src: str, dst: str) -> str:\n    return datetime.strptime(s, src).strftime(dst)\n```\nClassic two-step: parse then format.",
        "Make it ISO 8601 specifically and timezone-aware.",
        "```python\nfrom datetime import datetime, timezone\n\ndef to_iso(s: str, src: str, tz: timezone = timezone.utc) -> str:\n    dt = datetime.strptime(s, src).replace(tzinfo=tz)\n    return dt.isoformat()\n```\nReplacing `tzinfo` works only when the source string has no timezone info; otherwise parse with `%z` or use `dateutil`.")

    add("web",
        "Write a Flask route that returns Prometheus-compatible metrics.",
        "```python\nfrom flask import Flask, Response\nfrom prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST\n\napp = Flask(__name__)\nrequests_total = Counter('app_requests_total', 'Total requests')\n\n@app.get('/metrics')\ndef metrics():\n    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)\n\n@app.before_request\ndef count():\n    requests_total.inc()\n```\n`prometheus_client` ships the right metric serializer; just expose `generate_latest()`.",
        "Add a histogram that tracks request duration.",
        "```python\nimport time\nfrom flask import Flask, Response, g\nfrom prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST\n\napp = Flask(__name__)\nrequests_total = Counter('app_requests_total', 'Total requests')\nlatency = Histogram('app_request_duration_seconds', 'Request duration')\n\n@app.get('/metrics')\ndef metrics():\n    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)\n\n@app.before_request\ndef start():\n    requests_total.inc()\n    g.start = time.time()\n\n@app.after_request\ndef stop(resp):\n    latency.observe(time.time() - g.start)\n    return resp\n```\n`g` is the per-request context; perfect for stashing the start time.")

    add("web",
        "Write a function that parses a User-Agent string into browser/OS.",
        "```python\nfrom user_agents import parse\n\ndef parse_ua(ua: str) -> dict:\n    p = parse(ua)\n    return {\n        'browser': p.browser.family,\n        'browser_version': p.browser.version_string,\n        'os': p.os.family,\n        'device': p.device.family,\n    }\n```\nUA strings are too inconsistent to parse by regex reliably; the `user_agents` library bundles the curated rule set.",
        "I can't add a dependency — give me a regex-based fallback.",
        "```python\nimport re\n\ndef parse_ua(ua: str) -> dict:\n    browser = 'Other'\n    for name, pat in (('Edge', r'Edg/'), ('Chrome', r'Chrome/'), ('Safari', r'Safari/'), ('Firefox', r'Firefox/')):\n        if re.search(pat, ua):\n            browser = name\n            break\n    if 'Windows' in ua:\n        os_name = 'Windows'\n    elif 'Mac OS X' in ua or 'Macintosh' in ua:\n        os_name = 'macOS'\n    elif 'Android' in ua:\n        os_name = 'Android'\n    elif 'Linux' in ua:\n        os_name = 'Linux'\n    elif 'iPhone' in ua or 'iPad' in ua:\n        os_name = 'iOS'\n    else:\n        os_name = 'Unknown'\n    return {'browser': browser, 'os': os_name}\n```\nOrder matters — Edge UAs contain 'Chrome' too, so test for Edge first. This is rough but adequate for analytics buckets.")

    add("web",
        "Write a function that turns a list of objects into a CSV-streamed response in Flask.",
        "```python\nimport csv\nimport io\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/users.csv')\ndef users():\n    users = [{'id': i, 'name': f'user{i}'} for i in range(1000)]\n    def stream():\n        buf = io.StringIO()\n        w = csv.DictWriter(buf, fieldnames=['id', 'name'])\n        w.writeheader()\n        yield buf.getvalue(); buf.seek(0); buf.truncate(0)\n        for u in users:\n            w.writerow(u)\n            yield buf.getvalue(); buf.seek(0); buf.truncate(0)\n    return Response(stream(), mimetype='text/csv')\n```\nStreaming yields rows as they're built, so memory stays flat even for huge result sets.",
        "Add Content-Disposition so the browser saves it as a download.",
        "```python\nimport csv\nimport io\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/users.csv')\ndef users():\n    users = [{'id': i, 'name': f'user{i}'} for i in range(1000)]\n    def stream():\n        buf = io.StringIO()\n        w = csv.DictWriter(buf, fieldnames=['id', 'name'])\n        w.writeheader()\n        yield buf.getvalue(); buf.seek(0); buf.truncate(0)\n        for u in users:\n            w.writerow(u)\n            yield buf.getvalue(); buf.seek(0); buf.truncate(0)\n    return Response(\n        stream(), mimetype='text/csv',\n        headers={'Content-Disposition': 'attachment; filename=\"users.csv\"'},\n    )\n```\nThe `attachment` disposition tells the browser to save rather than render. Inline-quote the filename in case it has spaces.")

    add("web",
        "Write a function that submits a multipart form with files and other fields.",
        "```python\nimport requests\nfrom pathlib import Path\n\ndef submit(url: str, file_path: str, fields: dict) -> requests.Response:\n    with open(file_path, 'rb') as f:\n        return requests.post(\n            url,\n            files={'file': (Path(file_path).name, f)},\n            data=fields,\n            timeout=30,\n        )\n```\nMixing `files=` and `data=` produces a multipart body with both file parts and text fields.",
        "Allow multiple files under the same field name.",
        "```python\nimport requests\nfrom pathlib import Path\n\ndef submit(url: str, file_paths: list[str], fields: dict) -> requests.Response:\n    files = []\n    handles = []\n    try:\n        for p in file_paths:\n            f = open(p, 'rb')\n            handles.append(f)\n            files.append(('file', (Path(p).name, f)))\n        return requests.post(url, files=files, data=fields, timeout=30)\n    finally:\n        for f in handles:\n            f.close()\n```\nA list of `(field_name, ...)` tuples lets multiple parts share the same field name. We track handles explicitly so we can close them all in the `finally`.")

    add("web",
        "Write a function that does HTTP basic authentication and returns the response JSON.",
        "```python\nimport requests\n\ndef get_authed(url: str, user: str, password: str) -> dict:\n    r = requests.get(url, auth=(user, password), timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nThe tuple form is shorthand for `HTTPBasicAuth(user, password)`.",
        "How do I switch to digest auth?",
        "```python\nimport requests\nfrom requests.auth import HTTPDigestAuth\n\ndef get_authed(url: str, user: str, password: str) -> dict:\n    r = requests.get(url, auth=HTTPDigestAuth(user, password), timeout=10)\n    r.raise_for_status()\n    return r.json()\n```\nDigest auth is a challenge-response handshake — `requests` handles the second round-trip transparently.")

    add("web",
        "Write a function that downloads a list of files concurrently.",
        "```python\nimport requests\nfrom concurrent.futures import ThreadPoolExecutor\nfrom pathlib import Path\n\ndef download(url: str, dest: str) -> None:\n    with requests.get(url, stream=True, timeout=30) as r:\n        r.raise_for_status()\n        with open(dest, 'wb') as f:\n            for chunk in r.iter_content(8192):\n                f.write(chunk)\n\ndef download_all(urls: list[str], out_dir: str, workers: int = 8) -> None:\n    Path(out_dir).mkdir(parents=True, exist_ok=True)\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        futures = [ex.submit(download, u, str(Path(out_dir) / Path(u).name)) for u in urls]\n        for fu in futures:\n            fu.result()\n```\nThreads are great for I/O-bound work like HTTP. `fu.result()` re-raises exceptions in the main thread.",
        "Skip files that already exist.",
        "```python\nimport requests\nfrom concurrent.futures import ThreadPoolExecutor\nfrom pathlib import Path\n\ndef download(url: str, dest: Path) -> None:\n    if dest.exists():\n        return\n    with requests.get(url, stream=True, timeout=30) as r:\n        r.raise_for_status()\n        with open(dest, 'wb') as f:\n            for chunk in r.iter_content(8192):\n                f.write(chunk)\n\ndef download_all(urls: list[str], out_dir: str, workers: int = 8) -> None:\n    out = Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        list(ex.map(lambda u: download(u, out / Path(u).name), urls))\n```\nFor truly idempotent re-runs, also check the file size matches `Content-Length` to detect a partial previous download.")

    add("web",
        "Write a function that returns the HTTP status of a list of URLs.",
        "```python\nimport requests\n\ndef status_codes(urls: list[str]) -> dict[str, int]:\n    out = {}\n    for u in urls:\n        try:\n            out[u] = requests.head(u, allow_redirects=True, timeout=5).status_code\n        except requests.RequestException:\n            out[u] = 0\n    return out\n```\nUsing 0 for failures keeps the dict shape stable — caller can decide what 0 means.",
        "Use a thread pool to make it ~10x faster.",
        "```python\nimport requests\nfrom concurrent.futures import ThreadPoolExecutor\n\ndef _check(u: str) -> tuple[str, int]:\n    try:\n        return u, requests.head(u, allow_redirects=True, timeout=5).status_code\n    except requests.RequestException:\n        return u, 0\n\ndef status_codes(urls: list[str], workers: int = 16) -> dict[str, int]:\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        return dict(ex.map(_check, urls))\n```\nWorkers >> CPU count is fine here because the threads spend almost all their time waiting on the network.")

    add("web",
        "Write a function that constructs an absolute URL from a base URL and a relative path.",
        "```python\nfrom urllib.parse import urljoin\n\ndef absolute(base: str, rel: str) -> str:\n    return urljoin(base, rel)\n```\n`urljoin` handles all the edge cases (relative vs absolute, trailing slashes, etc.).",
        "What if I want to handle the case where rel is already absolute?",
        "`urljoin` already handles it correctly: if `rel` is an absolute URL, it's returned as-is. For example:\n```python\n>>> urljoin('https://a.com/x/', 'https://b.com/y')\n'https://b.com/y'\n```\nSo the original one-liner is sufficient. The function only resolves `rel` against `base` when `rel` is itself relative.")

    add("web",
        "Write a function that retrieves all pages from a paginated API and returns a flat list.",
        "```python\nimport requests\n\ndef get_all(url: str) -> list:\n    out = []\n    while url:\n        r = requests.get(url, timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        out.extend(body['items'])\n        url = body.get('next')\n    return out\n```\nFollow `next` until it disappears.",
        "How would I make this safe against infinite loops if the API is buggy and never stops returning a 'next'?",
        "```python\nimport requests\n\ndef get_all(url: str, max_pages: int = 1000) -> list:\n    out = []\n    seen = set()\n    for _ in range(max_pages):\n        if not url or url in seen:\n            break\n        seen.add(url)\n        r = requests.get(url, timeout=10)\n        r.raise_for_status()\n        body = r.json()\n        out.extend(body['items'])\n        url = body.get('next')\n    return out\n```\nCombine a hard cap with a loop-detection set: cap protects against quietly slow APIs, loop detection catches buggy `next` URLs that re-point at an earlier page.")

    add("web",
        "Write a function that exposes a Flask metrics endpoint listing route hit counts.",
        "```python\nfrom collections import Counter\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nhits: Counter = Counter()\n\n@app.before_request\ndef tally():\n    hits[request.endpoint] += 1\n\n@app.get('/_metrics')\ndef metrics():\n    return jsonify(dict(hits))\n```\nIn-memory counts are fine for small services. For production, use prometheus_client.",
        "Reset counts via a POST.",
        "```python\nfrom collections import Counter\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\nhits: Counter = Counter()\n\n@app.before_request\ndef tally():\n    hits[request.endpoint] += 1\n\n@app.get('/_metrics')\ndef metrics():\n    return jsonify(dict(hits))\n\n@app.post('/_metrics/reset')\ndef reset():\n    hits.clear()\n    return jsonify(reset=True)\n```\nProtect the reset endpoint with auth in any non-toy deployment.")

    add("web",
        "Write a function that converts a Flask response to a CORS-enabled response inline.",
        "```python\nfrom flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/data')\ndef data():\n    resp = jsonify(value=42)\n    resp.headers['Access-Control-Allow-Origin'] = '*'\n    return resp\n```\nQuick and dirty. For real apps use `flask-cors`.",
        "Echo the request's Origin if it's in an allowlist.",
        "```python\nfrom flask import Flask, jsonify, request\n\napp = Flask(__name__)\nALLOWED = {'https://example.com', 'https://app.example.com'}\n\n@app.get('/data')\ndef data():\n    resp = jsonify(value=42)\n    origin = request.headers.get('Origin', '')\n    if origin in ALLOWED:\n        resp.headers['Access-Control-Allow-Origin'] = origin\n        resp.headers['Vary'] = 'Origin'\n    return resp\n```\nThe `Vary: Origin` header tells caches that the response depends on the request's Origin.")

    add("web",
        "Write a function that tracks rate limits using a token bucket.",
        "```python\nimport time\n\nclass TokenBucket:\n    def __init__(self, capacity: float, refill_per_sec: float):\n        self.capacity = capacity\n        self.refill_per_sec = refill_per_sec\n        self.tokens = capacity\n        self.last = time.time()\n\n    def take(self, n: float = 1.0) -> bool:\n        now = time.time()\n        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_sec)\n        self.last = now\n        if self.tokens >= n:\n            self.tokens -= n\n            return False if False else True  # simplified\n        return False\n```\nBursts up to `capacity`, sustained rate of `refill_per_sec`. Single-process only.",
        "Make a decorator that uses it.",
        "```python\nimport time\nfrom functools import wraps\n\nclass TokenBucket:\n    def __init__(self, capacity: float, refill_per_sec: float):\n        self.capacity = capacity\n        self.refill_per_sec = refill_per_sec\n        self.tokens = capacity\n        self.last = time.time()\n\n    def take(self, n: float = 1.0) -> bool:\n        now = time.time()\n        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_sec)\n        self.last = now\n        if self.tokens >= n:\n            self.tokens -= n\n            return True\n        return False\n\ndef rate_limit(bucket: TokenBucket):\n    def deco(fn):\n        @wraps(fn)\n        def wrapper(*args, **kwargs):\n            if not bucket.take():\n                raise RuntimeError('rate limit exceeded')\n            return fn(*args, **kwargs)\n        return wrapper\n    return deco\n```\nThe decorator factory pattern lets each callsite share or have its own bucket as needed.")

    add("web",
        "Write a function that parses a HAR file and prints requests/responses.",
        "```python\nimport json\nfrom pathlib import Path\n\ndef summarize_har(path: str) -> list[dict]:\n    har = json.loads(Path(path).read_text())\n    out = []\n    for entry in har['log']['entries']:\n        out.append({\n            'method': entry['request']['method'],\n            'url': entry['request']['url'],\n            'status': entry['response']['status'],\n            'ms': entry['time'],\n        })\n    return out\n```\nHAR is just JSON. Useful for replaying traffic captured from a browser DevTools session.",
        "Filter to only failed requests (status >= 400).",
        "```python\nimport json\nfrom pathlib import Path\n\ndef failed_from_har(path: str) -> list[dict]:\n    har = json.loads(Path(path).read_text())\n    return [\n        {\n            'method': e['request']['method'],\n            'url': e['request']['url'],\n            'status': e['response']['status'],\n            'ms': e['time'],\n        }\n        for e in har['log']['entries']\n        if e['response']['status'] >= 400\n    ]\n```\nUseful for triaging which assets a page can't load.")

    add("web",
        "Write a function that fetches the favicon for a URL.",
        "```python\nimport requests\nfrom urllib.parse import urlparse\n\ndef favicon_bytes(url: str) -> bytes:\n    p = urlparse(url)\n    fav_url = f'{p.scheme}://{p.netloc}/favicon.ico'\n    r = requests.get(fav_url, timeout=10)\n    r.raise_for_status()\n    return r.content\n```\nThe `/favicon.ico` location is the historical default; many sites still rely on it.",
        "Some sites declare favicons in <link> tags — pick the largest.",
        "```python\nimport requests\nfrom bs4 import BeautifulSoup\nfrom urllib.parse import urljoin, urlparse\n\ndef favicon_url(page_url: str) -> str:\n    html = requests.get(page_url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    candidates = []\n    for link in soup.find_all('link', rel=True):\n        rels = [r.lower() for r in link['rel']]\n        if any('icon' in r for r in rels) and link.get('href'):\n            sizes = link.get('sizes', '')\n            try:\n                area = int(sizes.split('x')[0]) if 'x' in sizes else 0\n            except ValueError:\n                area = 0\n            candidates.append((area, link['href']))\n    if candidates:\n        candidates.sort(reverse=True)\n        return urljoin(page_url, candidates[0][1])\n    p = urlparse(page_url)\n    return f'{p.scheme}://{p.netloc}/favicon.ico'\n```\nWe parse `<link rel=\"icon\">` and `apple-touch-icon` tags, prefer the one with the largest declared size, and fall back to `/favicon.ico`.")

    add("web",
        "Write a Flask route that produces a JSON-streamed array using yields.",
        "```python\nimport json\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/numbers')\ndef numbers():\n    def stream():\n        yield '['\n        first = True\n        for i in range(1000):\n            if not first:\n                yield ','\n            yield json.dumps(i)\n            first = False\n        yield ']'\n    return Response(stream(), mimetype='application/json')\n```\nManual JSON streaming because the stdlib's encoder doesn't stream natively.",
        "Switch to JSON Lines so each line is independently parseable.",
        "```python\nimport json\nfrom flask import Flask, Response\n\napp = Flask(__name__)\n\n@app.get('/numbers')\ndef numbers():\n    def stream():\n        for i in range(1000):\n            yield json.dumps(i) + '\\n'\n    return Response(stream(), mimetype='application/x-ndjson')\n```\nNDJSON (newline-delimited JSON) is far simpler to stream and consume — many clients handle it line-by-line.")

    add("web",
        "Write a function that scrapes all images from a webpage.",
        "```python\nimport requests\nfrom bs4 import BeautifulSoup\nfrom urllib.parse import urljoin\n\ndef images_on(url: str) -> list[str]:\n    html = requests.get(url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    return [urljoin(url, img['src']) for img in soup.find_all('img', src=True)]\n```\n`urljoin` handles relative `src` values (e.g. `/images/foo.png`).",
        "Now actually download them.",
        "```python\nimport requests\nfrom bs4 import BeautifulSoup\nfrom concurrent.futures import ThreadPoolExecutor\nfrom pathlib import Path\nfrom urllib.parse import urljoin, urlparse\n\ndef download_images(page_url: str, out_dir: str, workers: int = 8) -> None:\n    html = requests.get(page_url, timeout=10).text\n    soup = BeautifulSoup(html, 'html.parser')\n    urls = [urljoin(page_url, img['src']) for img in soup.find_all('img', src=True)]\n    out = Path(out_dir)\n    out.mkdir(parents=True, exist_ok=True)\n\n    def grab(u: str) -> None:\n        name = Path(urlparse(u).path).name or 'image'\n        r = requests.get(u, timeout=15)\n        r.raise_for_status()\n        (out / name).write_bytes(r.content)\n\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        list(ex.map(grab, urls))\n```\nDeriving the filename from the URL path is sufficient for most cases; collision handling can be added when needed.")

    add("web",
        "Write a function that performs a DNS lookup for MX records.",
        "```python\nimport dns.resolver\n\ndef mx_records(domain: str) -> list[tuple[int, str]]:\n    answers = dns.resolver.resolve(domain, 'MX')\n    return sorted((rec.preference, str(rec.exchange).rstrip('.')) for rec in answers)\n```\n`dnspython` is the canonical Python DNS lib. Lower preference = higher priority.",
        "How would I do this with only the standard library?",
        "The stdlib's `socket` module doesn't expose MX records — it's restricted to A/AAAA. You'd need to send a raw DNS query yourself or shell out to `dig`/`nslookup`. The pragmatic answer is: take the `dnspython` dependency. If absolutely necessary:\n```python\nimport subprocess\n\ndef mx_records(domain: str) -> list[str]:\n    out = subprocess.run(['dig', '+short', 'MX', domain], capture_output=True, text=True, check=True)\n    return [line.strip() for line in out.stdout.splitlines() if line.strip()]\n```\nBut `dnspython` is the right answer for any non-trivial DNS work.")

    add("web",
        "Write a function that counts the size of an HTTP response body without downloading it.",
        "```python\nimport requests\n\ndef content_length(url: str) -> int | None:\n    r = requests.head(url, allow_redirects=True, timeout=10)\n    cl = r.headers.get('Content-Length')\n    return int(cl) if cl else None\n```\nReturns `None` if the server doesn't provide `Content-Length` (e.g. chunked transfer).",
        "What if HEAD doesn't return Content-Length but GET does?",
        "```python\nimport requests\n\ndef content_length(url: str) -> int | None:\n    r = requests.head(url, allow_redirects=True, timeout=10)\n    cl = r.headers.get('Content-Length')\n    if cl:\n        return int(cl)\n    with requests.get(url, stream=True, timeout=10) as g:\n        cl = g.headers.get('Content-Length')\n        return int(cl) if cl else None\n```\n`stream=True` keeps the body unread so we don't pay the bandwidth cost just to read headers.")

    add("web",
        "Write a function that tests whether a given email's MX records resolve.",
        "```python\nimport dns.resolver\n\ndef email_domain_resolves(email: str) -> bool:\n    domain = email.split('@', 1)[1]\n    try:\n        dns.resolver.resolve(domain, 'MX')\n        return True\n    except dns.resolver.NoAnswer:\n        return False\n    except dns.resolver.NXDOMAIN:\n        return False\n```\nCatching the specific resolver exceptions is more honest than a bare `except Exception`.",
        "Add a fallback: if there's no MX record, accept an A record instead.",
        "```python\nimport dns.resolver\n\ndef email_domain_resolves(email: str) -> bool:\n    domain = email.split('@', 1)[1]\n    for rrtype in ('MX', 'A', 'AAAA'):\n        try:\n            dns.resolver.resolve(domain, rrtype)\n            return True\n        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):\n            continue\n    return False\n```\nThis matches the SMTP behavior: many MTAs fall back to A/AAAA when MX is missing.")
