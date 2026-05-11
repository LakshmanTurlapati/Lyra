"""Extra tool builders for TC-H2 wave 2 (cloud/devops/iot/media/ecommerce)."""
from __future__ import annotations

import json


REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1"]
GCP_REGIONS = ["us-central1", "europe-west1", "asia-east1"]
AZURE_REGIONS = ["eastus", "westus2", "northeurope"]
CLUSTERS = ["prod-east", "prod-west", "staging", "dev-eu", "qa-us", "edge-1"]
NAMESPACES = ["default", "platform", "billing", "auth", "search", "ingest", "checkout", "media"]
APPS = ["api-prod", "worker", "scheduler", "ingest-svc", "media-encoder", "checkout-api", "auth-svc"]
DEVICES = ["sensor-A12", "sensor-B07", "gateway-EU3", "thermo-91", "valve-22", "cam-roof-7"]
PLAYLISTS = ["chill-friday", "deep-focus", "morning-coffee", "rainy-night", "workout-mix"]
SKUS = ["SKU-1042", "SKU-2298", "SKU-7711", "SKU-3360", "SKU-5125"]


def _id(rng, prefix, n=10):
    chars = "0123456789abcdef"
    return prefix + "".join(rng.choice(chars) for _ in range(n))


def extra_tools():
    T = {}

    # --- Cloudflare ---
    def t_cf_purge_cache(rng):
        zone = rng.choice(["lyra.dev", "shop.example.com", "media.example.io", "iot-bridge.net"])
        urls = [f"https://{zone}/{p}" for p in rng.sample(["index.html", "static/app.js", "img/logo.png", "robots.txt"], 2)]
        user = rng.choice([
            f"Purge Cloudflare cache for {urls[0]} and {urls[1]}.",
            f"Drop CF cache on these {zone} URLs: {urls[0]}, {urls[1]}.",
            f"Invalidate {zone} cache for those two assets.",
        ])
        args = {"zone": zone, "urls": urls}
        result = json.dumps({"success": True, "purged": len(urls)})
        summary = f"Purged {len(urls)} URLs from {zone}."
        return user, args, result, summary
    T["cloudflare_purge_cache"] = ("Purge Cloudflare CDN cache for given URLs.", t_cf_purge_cache)

    def t_cf_create_dns_record(rng):
        zone = rng.choice(["lyra.dev", "shop.example.com"])
        sub = rng.choice(["api", "cdn", "stage", "dash", "ws"])
        ip = f"{rng.randint(1,254)}.{rng.randint(0,254)}.{rng.randint(0,254)}.{rng.randint(1,254)}"
        user = rng.choice([
            f"Add an A record {sub}.{zone} -> {ip}.",
            f"Create DNS A {sub}.{zone} pointing to {ip}.",
            f"Wire up {sub}.{zone} as A record to {ip} on Cloudflare.",
        ])
        args = {"zone": zone, "type": "A", "name": f"{sub}.{zone}", "content": ip}
        rid = _id(rng, "", 16)
        result = json.dumps({"id": rid, "name": f"{sub}.{zone}", "type": "A", "content": ip})
        summary = f"Record {sub}.{zone} -> {ip} created."
        return user, args, result, summary
    T["cloudflare_create_dns_record"] = ("Create a Cloudflare DNS record.", t_cf_create_dns_record)

    # --- DigitalOcean ---
    def t_do_create_droplet(rng):
        name = f"droplet-{rng.choice(['web','api','db','worker'])}-{rng.randint(1,99)}"
        size = rng.choice(["s-1vcpu-1gb", "s-2vcpu-2gb", "s-2vcpu-4gb", "c-4"])
        region = rng.choice(["nyc3", "sfo3", "ams3", "sgp1", "fra1"])
        user = rng.choice([
            f"Spin up a DO droplet '{name}' size {size} in {region}.",
            f"Create a {size} droplet named {name} in {region}.",
            f"Provision DigitalOcean {name} ({size}, {region}).",
        ])
        args = {"name": name, "size": size, "region": region, "image": "ubuntu-24-04-x64"}
        did = rng.randint(10**8, 10**9 - 1)
        result = json.dumps({"droplet_id": did, "status": "new", "name": name})
        summary = f"Droplet {name} ({did}) provisioning."
        return user, args, result, summary
    T["digitalocean_create_droplet"] = ("Create a DigitalOcean droplet.", t_do_create_droplet)

    def t_do_destroy_droplet(rng):
        did = rng.randint(10**8, 10**9 - 1)
        user = rng.choice([
            f"Destroy DO droplet {did}.",
            f"Tear down droplet {did} on DigitalOcean.",
            f"Delete DO droplet id={did}.",
        ])
        args = {"droplet_id": did}
        result = json.dumps({"droplet_id": did, "status": "destroying"})
        summary = f"Droplet {did} destroying."
        return user, args, result, summary
    T["digitalocean_destroy_droplet"] = ("Destroy a DigitalOcean droplet.", t_do_destroy_droplet)

    # --- Fastly ---
    def t_fastly_purge_url(rng):
        url = rng.choice([
            "https://cdn.example.com/static/app.css",
            "https://media.example.io/v/clip-22.mp4",
            "https://shop.example.com/p/wool-beanie",
        ])
        user = rng.choice([
            f"Purge {url} from Fastly.",
            f"Drop Fastly cache on {url}.",
            f"Invalidate {url} via Fastly.",
        ])
        args = {"url": url}
        result = json.dumps({"status": "ok", "id": _id(rng, "purge_", 12)})
        summary = f"Fastly purged {url}."
        return user, args, result, summary
    T["fastly_purge_url"] = ("Purge a single URL from Fastly CDN.", t_fastly_purge_url)

    # --- Vercel ---
    def t_vercel_deploy(rng):
        proj = rng.choice(["lyra-dash", "marketing-site", "checkout-ui", "iot-portal"])
        branch = rng.choice(["main", "preview", "staging", "feature/redesign"])
        user = rng.choice([
            f"Trigger a Vercel deploy of {proj} from {branch}.",
            f"Ship {proj} ({branch}) to Vercel.",
            f"Kick off Vercel build for {proj} on {branch}.",
        ])
        args = {"project": proj, "branch": branch}
        dep = _id(rng, "dpl_", 16)
        result = json.dumps({"deployment_id": dep, "url": f"https://{proj}-{dep[-6:]}.vercel.app", "state": "BUILDING"})
        summary = f"Deployment {dep} for {proj} building."
        return user, args, result, summary
    T["vercel_deploy"] = ("Trigger a Vercel deployment.", t_vercel_deploy)

    def t_vercel_list_deployments(rng):
        proj = rng.choice(["lyra-dash", "marketing-site", "checkout-ui"])
        user = rng.choice([
            f"List recent Vercel deployments for {proj}.",
            f"What did we ship to Vercel on {proj} lately?",
            f"Pull the last few {proj} Vercel deploys.",
        ])
        args = {"project": proj, "limit": 10}
        n = rng.randint(3, 10)
        result = json.dumps({"project": proj, "deployments": n})
        summary = f"{n} recent deployments for {proj}."
        return user, args, result, summary
    T["vercel_list_deployments"] = ("List recent Vercel deployments.", t_vercel_list_deployments)

    # --- Netlify ---
    def t_netlify_create_site(rng):
        name = rng.choice(["blog-mk2", "lp-spring", "docs-portal", "events-2026"])
        user = rng.choice([
            f"Create a Netlify site called {name}.",
            f"Stand up Netlify site {name}.",
            f"Provision a new Netlify site: {name}.",
        ])
        args = {"name": name}
        sid = _id(rng, "", 12)
        result = json.dumps({"site_id": sid, "name": name, "url": f"https://{name}.netlify.app"})
        summary = f"Site {name} created."
        return user, args, result, summary
    T["netlify_create_site"] = ("Create a Netlify site.", t_netlify_create_site)

    # --- Sentry ---
    def t_sentry_resolve_issue(rng):
        iid = _id(rng, "", 10).upper()
        user = rng.choice([
            f"Mark Sentry issue {iid} resolved.",
            f"Close out Sentry {iid}.",
            f"Resolve issue {iid} in Sentry.",
        ])
        args = {"issue_id": iid}
        result = json.dumps({"issue": iid, "status": "resolved"})
        summary = f"Issue {iid} resolved."
        return user, args, result, summary
    T["sentry_resolve_issue"] = ("Resolve a Sentry issue.", t_sentry_resolve_issue)

    # --- HashiCorp Vault ---
    def t_vault_read_secret(rng):
        path = rng.choice(["secret/data/db/prod", "secret/data/api/stripe", "secret/data/iot/mqtt"])
        user = rng.choice([
            f"Read secret at {path}.",
            f"Pull the secret stored at {path}.",
            f"Fetch Vault secret {path}.",
        ])
        args = {"path": path}
        result = json.dumps({"path": path, "version": rng.randint(1, 12), "keys": ["username", "password"]})
        summary = f"Read {path}."
        return user, args, result, summary
    T["vault_read_secret"] = ("Read a HashiCorp Vault secret.", t_vault_read_secret)

    # --- IoT extras ---
    def t_iot_reboot_device(rng):
        d = rng.choice(DEVICES)
        user = rng.choice([
            f"Reboot device {d}.",
            f"Power-cycle {d} please.",
            f"Restart IoT device {d}.",
        ])
        args = {"device_id": d}
        result = json.dumps({"device": d, "command": "reboot", "queued": True})
        summary = f"Reboot queued for {d}."
        return user, args, result, summary
    T["iot_reboot_device"] = ("Reboot an IoT device.", t_iot_reboot_device)

    def t_iot_update_firmware(rng):
        d = rng.choice(DEVICES)
        ver = f"{rng.randint(1,3)}.{rng.randint(0,9)}.{rng.randint(0,20)}"
        user = rng.choice([
            f"Push firmware {ver} to {d}.",
            f"OTA-update {d} to v{ver}.",
            f"Roll firmware {ver} out to device {d}.",
        ])
        args = {"device_id": d, "version": ver}
        result = json.dumps({"device": d, "target_version": ver, "status": "pending"})
        summary = f"OTA to {ver} pending on {d}."
        return user, args, result, summary
    T["iot_update_firmware"] = ("Schedule an IoT firmware update.", t_iot_update_firmware)

    def t_iot_get_telemetry(rng):
        d = rng.choice(DEVICES)
        metric = rng.choice(["temperature", "humidity", "battery", "rssi"])
        user = rng.choice([
            f"What's the latest {metric} from {d}?",
            f"Pull most recent {metric} reading on {d}.",
            f"Read {metric} for device {d}.",
        ])
        args = {"device_id": d, "metric": metric}
        val = round(rng.uniform(0, 100), 2)
        result = json.dumps({"device": d, "metric": metric, "value": val, "ts": "2026-05-08T12:00:00Z"})
        summary = f"{d} {metric} = {val}."
        return user, args, result, summary
    T["iot_get_telemetry"] = ("Get latest telemetry reading from an IoT device.", t_iot_get_telemetry)

    # --- Media (Spotify-like, YouTube-like) ---
    def t_media_search_track(rng):
        q = rng.choice(["midnight drive", "kestrel", "lo-fi piano", "neon tide remix"])
        user = rng.choice([
            f"Search tracks for '{q}'.",
            f"Find songs matching '{q}'.",
            f"Look up '{q}' on the catalog.",
        ])
        args = {"query": q, "limit": 10}
        n = rng.randint(2, 10)
        result = json.dumps({"query": q, "hits": n})
        summary = f"{n} tracks matched '{q}'."
        return user, args, result, summary
    T["media_search_track"] = ("Search the music catalog for tracks.", t_media_search_track)

    def t_media_add_to_playlist(rng):
        pl = rng.choice(PLAYLISTS)
        track = _id(rng, "trk_", 16)
        user = rng.choice([
            f"Add track {track} to playlist {pl}.",
            f"Drop {track} into the {pl} playlist.",
            f"Append {track} to {pl}.",
        ])
        args = {"playlist": pl, "track_id": track}
        result = json.dumps({"playlist": pl, "added": track, "size": rng.randint(5, 80)})
        summary = f"Added {track} to {pl}."
        return user, args, result, summary
    T["media_add_to_playlist"] = ("Add a track to a playlist.", t_media_add_to_playlist)

    def t_media_get_video_stats(rng):
        vid = _id(rng, "vid_", 11)
        user = rng.choice([
            f"Get stats for video {vid}.",
            f"Pull view/like counts on {vid}.",
            f"How is video {vid} performing?",
        ])
        args = {"video_id": vid}
        views = rng.randint(100, 5_000_000)
        likes = rng.randint(10, views // 5 or 1)
        result = json.dumps({"video": vid, "views": views, "likes": likes})
        summary = f"{vid}: {views} views, {likes} likes."
        return user, args, result, summary
    T["media_get_video_stats"] = ("Get statistics for a video.", t_media_get_video_stats)

    def t_media_transcode(rng):
        src = f"s3://media-uploads/raw/{_id(rng, '', 12)}.mov"
        preset = rng.choice(["1080p-h264", "720p-h264", "480p-vp9", "audio-aac"])
        user = rng.choice([
            f"Transcode {src} with preset {preset}.",
            f"Run {preset} transcode on {src}.",
            f"Encode {src} into {preset}.",
        ])
        args = {"source": src, "preset": preset}
        jid = _id(rng, "job_", 14)
        result = json.dumps({"job_id": jid, "status": "queued", "preset": preset})
        summary = f"Transcode job {jid} queued."
        return user, args, result, summary
    T["media_transcode_job"] = ("Submit a media transcode job.", t_media_transcode)

    # --- E-commerce extras ---
    def t_shop_create_discount(rng):
        code = rng.choice(["SPRING10", "VIP20", "BUNDLE15", "FLASH25"])
        pct = rng.choice([10, 15, 20, 25])
        user = rng.choice([
            f"Create discount code {code} for {pct}% off.",
            f"Set up a {pct}% discount with code {code}.",
            f"Spin up coupon {code} ({pct}% off).",
        ])
        args = {"code": code, "percent_off": pct}
        result = json.dumps({"code": code, "percent_off": pct, "active": True})
        summary = f"Coupon {code} ({pct}%) live."
        return user, args, result, summary
    T["shop_create_discount"] = ("Create a discount code in the shop.", t_shop_create_discount)

    def t_shop_get_inventory(rng):
        sku = rng.choice(SKUS)
        user = rng.choice([
            f"How many {sku} units are in stock?",
            f"Check inventory for {sku}.",
            f"What's the on-hand count for {sku}?",
        ])
        args = {"sku": sku}
        qty = rng.randint(0, 480)
        result = json.dumps({"sku": sku, "on_hand": qty, "reserved": rng.randint(0, qty // 4 or 1)})
        summary = f"{sku} on hand: {qty}."
        return user, args, result, summary
    T["shop_get_inventory"] = ("Get inventory level for a SKU.", t_shop_get_inventory)

    def t_shop_update_price(rng):
        sku = rng.choice(SKUS)
        price = round(rng.uniform(5.99, 199.99), 2)
        user = rng.choice([
            f"Set price of {sku} to ${price}.",
            f"Update {sku} pricing to ${price}.",
            f"Change {sku} list price -> ${price}.",
        ])
        args = {"sku": sku, "price_usd": price}
        result = json.dumps({"sku": sku, "price_usd": price, "updated": True})
        summary = f"{sku} now ${price}."
        return user, args, result, summary
    T["shop_update_price"] = ("Update the price of a SKU.", t_shop_update_price)

    def t_shop_fulfill_order(rng):
        oid = _id(rng, "ord_", 12)
        carrier = rng.choice(["ups", "fedex", "usps", "dhl"])
        tracking = _id(rng, "", 14).upper()
        user = rng.choice([
            f"Mark order {oid} fulfilled via {carrier} ({tracking}).",
            f"Ship out {oid} on {carrier}, tracking {tracking}.",
            f"Fulfill {oid} — carrier {carrier}, tracking {tracking}.",
        ])
        args = {"order_id": oid, "carrier": carrier, "tracking": tracking}
        result = json.dumps({"order": oid, "status": "fulfilled", "carrier": carrier, "tracking": tracking})
        summary = f"Order {oid} fulfilled via {carrier}."
        return user, args, result, summary
    T["shop_fulfill_order"] = ("Fulfill a shop order.", t_shop_fulfill_order)

    return T
