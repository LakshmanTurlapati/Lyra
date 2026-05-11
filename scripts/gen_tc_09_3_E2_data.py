"""Per-tool prompt + arg + result builders for TC-E2 (wave 2 — fresh tools).

Domain focus: vector DBs, graph DBs, time-series, search engines,
CDC/streaming connectors, feature stores, data catalogs, data quality,
MDM/dedup, advanced migrations, query plans, partitions, materialized
views, Iceberg/Delta lake operations.
"""
import random


COLLECTIONS = ["docs_v3", "products_idx", "support_tickets", "kb_articles", "code_chunks", "image_embeds", "policies_v2"]
NAMESPACES = ["prod", "staging", "tenant_a", "tenant_b", "shared"]
GRAPHS = ["social_graph", "fraud_ring", "supply_chain", "kg_main", "ontology_v4"]
NODE_LABELS = ["User", "Account", "Device", "Merchant", "Product", "Order", "Concept", "Address", "IP"]
REL_TYPES = ["KNOWS", "BOUGHT", "USES", "LOGGED_IN_FROM", "SHIPPED_TO", "MENTIONS", "OWNS"]
TS_METRICS = ["cpu_usage_pct", "mem_used_bytes", "http_requests_total", "p99_latency_ms", "queue_depth", "disk_io_bytes", "active_users"]
ES_INDICES = ["logs-app-2026.05", "logs-nginx-2026.05", "audit-2026.05", "products-v7", "users-search-v2", "orders-search-2026"]
TOPICS_CDC = ["pg.public.orders", "pg.public.users", "mysql.shop.products", "mongo.app.events"]
FEATURES = ["user_total_spend_30d", "session_clicks_1h", "items_in_cart", "days_since_signup", "ltv_est", "fraud_score_v3"]
ENTITIES = ["user_id", "merchant_id", "device_id", "session_id", "order_id"]
DATASETS_CAT = ["analytics.fct_orders", "marts.dim_user", "raw.events", "billing.invoices", "growth.signups"]
ICEBERG_TABLES = ["lakehouse.orders", "lakehouse.events", "lakehouse.users", "lakehouse.payments", "datalake.clickstream"]
DELTA_PATHS = ["s3://lyra-lake/silver/orders", "s3://lyra-lake/bronze/events", "abfss://lake@acct/gold/users"]
PROM_QUERIES = [
    "rate(http_requests_total[5m])",
    "histogram_quantile(0.99, sum(rate(req_duration_seconds_bucket[5m])) by (le))",
    "avg(node_cpu_seconds_total)",
    "sum(kube_pod_status_phase{phase=\"Pending\"})",
]
INFLUX_BUCKETS = ["telemetry", "iot_sensors", "k8s_metrics", "edge_devices"]
KCONNECT_CONNECTORS = ["debezium-pg-orders", "s3-sink-events", "snowflake-sink-marts", "jdbc-source-billing"]
SEARCH_QUERIES = ["wireless headphones", "refund policy", "rate limiting", "kubernetes pod pending", "annual report 2025"]
PG_VECTOR_TABLES = ["doc_embeddings", "product_vectors", "user_profile_emb", "support_emb"]
PARTITIONS = ["2026-04", "2026-05", "p_2026q1", "p_2026q2", "us-east-1", "eu-west-1"]


def rint(a, b): return random.randint(a, b)
def rfloat(a, b, p=4): return round(random.uniform(a, b), p)
def pick(seq): return random.choice(seq)


def gen_for_tool(name):
    """Returns (user_prompt, args_dict, tool_result_json_str, summary_fragment)."""

    if name == "pinecone_upsert":
        ns = pick(NAMESPACES); idx = pick(COLLECTIONS); n = rint(1, 200)
        prompt = pick([
            f"Upsert {n} vectors into Pinecone index `{idx}` under namespace `{ns}`.",
            f"Push {n} embeddings into the `{idx}` Pinecone index, namespace `{ns}` please.",
            f"Send a batch of {n} vectors to Pinecone — index `{idx}`, ns `{ns}`.",
        ])
        return (prompt, {"index": idx, "namespace": ns, "vectors": [{"id": f"v{rint(1,9999)}", "values_dim": 1536}], "count": n},
                '{"upserted_count":' + str(n) + ',"index":"' + idx + '"}',
                f"{n} vectors upserted to `{idx}`")
    if name == "pinecone_query":
        ns = pick(NAMESPACES); idx = pick(COLLECTIONS); k = rint(3, 25)
        prompt = pick([
            f"Query Pinecone `{idx}` (ns={ns}) for the top {k} nearest matches to my query embedding.",
            f"Pull the {k} closest neighbors from Pinecone index `{idx}` in namespace `{ns}`.",
            f"Run a top-{k} similarity search against `{idx}`/{ns} on Pinecone.",
        ])
        return (prompt, {"index": idx, "namespace": ns, "top_k": k, "vector_dim": 1536},
                '{"matches":[{"id":"doc_42","score":' + str(rfloat(0.7, 0.99)) + '},{"id":"doc_91","score":' + str(rfloat(0.6, 0.95)) + '}]}',
                f"top {k} matches returned from `{idx}`")
    if name == "pinecone_delete":
        ns = pick(NAMESPACES); idx = pick(COLLECTIONS)
        prompt = pick([
            f"Delete all vectors in Pinecone `{idx}` namespace `{ns}`.",
            f"Wipe the `{ns}` namespace inside the `{idx}` Pinecone index.",
            f"Clear Pinecone — drop everything under `{idx}`/{ns}.",
        ])
        return (prompt, {"index": idx, "namespace": ns, "delete_all": True},
                '{"deleted":true}', f"namespace `{ns}` cleared in `{idx}`")
    if name == "weaviate_search":
        cls = pick(["Document", "Article", "Product", "Ticket"]); k = rint(3, 15); q = pick(SEARCH_QUERIES)
        prompt = pick([
            f"Search Weaviate class `{cls}` for `{q}`, limit {k}.",
            f"Run a Weaviate near-text query against `{cls}` for `{q}` (top {k}).",
            f"In Weaviate, find {k} `{cls}` items semantically close to `{q}`.",
        ])
        return (prompt, {"class_name": cls, "near_text": q, "limit": k},
                '{"data":{"Get":{"' + cls + '":[{"_additional":{"distance":' + str(rfloat(0.1, 0.5)) + '},"title":"sample"}]}}}',
                f"hybrid search returned matches in `{cls}`")
    if name == "weaviate_create_class":
        cls = pick(["KbArticle", "Embedding", "ChatMessage", "Snippet"]) + str(rint(1, 9))
        prompt = pick([
            f"Create a Weaviate class `{cls}` with text2vec-openai vectorizer.",
            f"Stand up a new Weaviate class named `{cls}` using the openai vectorizer.",
            f"Define `{cls}` in Weaviate — title and body properties, openai vectorizer.",
        ])
        return (prompt, {"class_name": cls, "vectorizer": "text2vec-openai", "properties": ["title", "body"]},
                '{"created":true,"class":"' + cls + '"}', f"class `{cls}` defined")
    if name == "qdrant_upsert_points":
        coll = pick(COLLECTIONS); n = rint(10, 500)
        prompt = pick([
            f"Push {n} points into the Qdrant collection `{coll}`.",
            f"Bulk-upsert {n} points to `{coll}` on Qdrant.",
            f"Send {n} new vectors to the Qdrant `{coll}` collection.",
        ])
        return (prompt, {"collection": coll, "points": [{"id": rint(1, 10**6), "vector_dim": 768}], "count": n},
                '{"operation_id":' + str(rint(1, 9999)) + ',"status":"completed"}',
                f"{n} points upserted to `{coll}`")
    if name == "qdrant_search":
        coll = pick(COLLECTIONS); k = rint(5, 30)
        prompt = pick([
            f"Run a Qdrant similarity search on `{coll}` returning {k} hits.",
            f"Find {k} nearest points in `{coll}` on Qdrant.",
            f"Hit Qdrant — top {k} from collection `{coll}`.",
        ])
        return (prompt, {"collection": coll, "limit": k, "vector_dim": 768},
                '{"result":[{"id":91,"score":' + str(rfloat(0.7, 0.99)) + '},{"id":17,"score":' + str(rfloat(0.6, 0.9)) + '}]}',
                f"{k}-nn search executed against `{coll}`")
    if name == "qdrant_create_collection":
        coll = "lyra_" + pick(["docs", "code", "imgs", "tickets"]) + "_" + str(rint(1, 99))
        prompt = pick([
            f"Create a Qdrant collection `{coll}` with cosine distance, dim 1024.",
            f"Provision the `{coll}` collection on Qdrant — 1024-dim, cosine.",
            f"Set up Qdrant collection `{coll}` (cosine, 1024-dim vectors).",
        ])
        return (prompt, {"collection": coll, "vector_size": 1024, "distance": "Cosine"},
                '{"result":true,"status":"ok"}', f"collection `{coll}` ready")
    if name == "pgvector_similarity":
        tbl = pick(PG_VECTOR_TABLES); k = rint(3, 20)
        prompt = pick([
            f"Find the {k} closest rows in `{tbl}` to my embedding using pgvector.",
            f"Run a pgvector ANN lookup on `{tbl}`, return {k} neighbors.",
            f"Use the cosine operator on `{tbl}` to fetch the top {k} matches.",
        ])
        return (prompt, {"table": tbl, "limit": k, "operator": "<=>"},
                '{"rows":[{"id":7,"distance":' + str(rfloat(0.05, 0.4)) + '},{"id":22,"distance":' + str(rfloat(0.1, 0.5)) + '}]}',
                f"pgvector returned {k} neighbors from `{tbl}`")
    if name == "pgvector_create_index":
        tbl = pick(PG_VECTOR_TABLES); lists = pick([100, 200, 500, 1000])
        prompt = pick([
            f"Create an IVFFlat index on `{tbl}` (embedding) with {lists} lists.",
            f"Add an ANN index to `{tbl}.embedding` — IVFFlat, {lists} lists.",
            f"Build the pgvector IVFFlat index for `{tbl}` ({lists} lists).",
        ])
        return (prompt, {"table": tbl, "column": "embedding", "index_type": "ivfflat", "lists": lists},
                '{"created":true,"index":"ix_' + tbl + '_emb"}',
                f"IVFFlat index on `{tbl}` built")

    if name == "neo4j_run_cypher":
        cy = f"MATCH (u:{pick(NODE_LABELS)})-[:{pick(REL_TYPES)}]->(x) WHERE u.id = $id RETURN x LIMIT 25"
        prompt = pick([
            f"Run this Cypher: `{cy}`",
            f"Execute on Neo4j: `{cy}`",
            f"Pop this Cypher into Neo4j and return the results: `{cy}`",
        ])
        return (prompt, {"cypher": cy, "params": {"id": rint(1, 99999)}},
                '{"records":[{"x":{"id":42,"label":"' + pick(NODE_LABELS) + '"}}],"summary":{"result_available_after_ms":' + str(rint(2, 200)) + '}}',
                "Cypher results returned")
    if name == "neo4j_create_node":
        lbl = pick(NODE_LABELS); nid = rint(100, 9999)
        prompt = pick([
            f"Create a `{lbl}` node with id={nid} in Neo4j.",
            f"Add a new `{lbl}` to Neo4j (id {nid}, created today).",
            f"In Neo4j, MERGE a `{lbl}` node carrying id={nid}.",
        ])
        return (prompt, {"label": lbl, "properties": {"id": nid, "created_at": "2026-05-08"}},
                '{"node_id":' + str(rint(10**6, 10**7)) + ',"label":"' + lbl + '"}',
                f"`{lbl}` node created")
    if name == "neo4j_create_relationship":
        a = pick(NODE_LABELS); b = pick(NODE_LABELS); r = pick(REL_TYPES)
        fid = rint(1, 9999); tid = rint(1, 9999)
        prompt = pick([
            f"Connect ({a} id={fid})-[:{r}]->({b} id={tid}) in Neo4j.",
            f"Add a `{r}` edge from `{a}` {fid} to `{b}` {tid}.",
            f"Wire up Neo4j: ({a})-[:{r}]->({b}) using ids {fid} and {tid}.",
        ])
        return (prompt, {"from_label": a, "to_label": b, "rel_type": r, "from_id": fid, "to_id": tid},
                '{"created":true,"rel_type":"' + r + '"}',
                f"`{r}` edge created")
    if name == "neptune_gremlin":
        gq = "g.V().has('User','id'," + str(rint(1, 9999)) + ").out('KNOWS').limit(10)"
        prompt = pick([
            f"Run this Gremlin traversal on Neptune: `{gq}`",
            f"Execute on AWS Neptune: `{gq}`",
            f"Traverse Neptune with: `{gq}`",
        ])
        return (prompt, {"query": gq},
                '{"vertices":[{"id":"u_42","label":"User"}],"count":1}',
                "Gremlin traversal executed")
    if name == "graph_shortest_path":
        g = pick(GRAPHS); src = rint(1, 9999); tgt = rint(1, 9999)
        prompt = pick([
            f"Find the shortest path between users {src} and {tgt} in `{g}`.",
            f"How are nodes {src} and {tgt} connected in `{g}`? Use shortest-path.",
            f"Run shortest-path on `{g}` from {src} to {tgt}, max depth 6.",
        ])
        return (prompt, {"graph": g, "source_id": src, "target_id": tgt, "max_depth": 6},
                '{"path_length":' + str(rint(2, 6)) + ',"nodes":["a","b","c"]}',
                f"shortest path found in `{g}`")

    if name == "influx_write_points":
        bucket = pick(INFLUX_BUCKETS); n = rint(10, 5000); m = pick(TS_METRICS)
        prompt = pick([
            f"Write {n} `{m}` points to InfluxDB bucket `{bucket}`.",
            f"Send {n} new measurements (`{m}`) into Influx bucket `{bucket}`.",
            f"Push {n} points (measurement `{m}`) to InfluxDB → `{bucket}`.",
        ])
        return (prompt, {"bucket": bucket, "measurement": m, "point_count": n},
                '{"written":' + str(n) + ',"bucket":"' + bucket + '"}',
                f"{n} points written to `{bucket}`")
    if name == "influx_flux_query":
        bucket = pick(INFLUX_BUCKETS); m = pick(TS_METRICS)
        flux = f'from(bucket: "{bucket}") |> range(start: -1h) |> filter(fn: (r) => r._measurement == "{m}") |> mean()'
        prompt = pick([
            f"Run this Flux query: `{flux}`",
            f"Execute on InfluxDB: `{flux}`",
            f"Get the 1h mean of `{m}` from bucket `{bucket}` via Flux.",
        ])
        return (prompt, {"flux": flux},
                '{"tables":[{"records":[{"_value":' + str(rfloat(0, 100)) + ',"_measurement":"' + m + '"}]}]}',
                "Flux mean returned")
    if name == "timescale_create_hypertable":
        tbl = pick(["sensor_readings", "events_ts", "metrics_raw", "audit_ts"])
        prompt = pick([
            f"Convert `{tbl}` to a TimescaleDB hypertable on `time`.",
            f"Promote `{tbl}` into a Timescale hypertable, chunk by 1 day.",
            f"Make `{tbl}` a hypertable in TimescaleDB partitioned on `time`.",
        ])
        return (prompt, {"table": tbl, "time_column": "time", "chunk_interval": "1 day"},
                '{"hypertable_id":' + str(rint(1, 999)) + ',"created":true}',
                f"`{tbl}` is now a hypertable")
    if name == "timescale_compress_chunks":
        tbl = pick(["sensor_readings", "events_ts", "metrics_raw"])
        prompt = pick([
            f"Compress chunks older than 7 days on `{tbl}`.",
            f"Run TimescaleDB compression on `{tbl}` for chunks > 7d old.",
            f"Compact aged chunks of `{tbl}` (anything past 7 days).",
        ])
        return (prompt, {"hypertable": tbl, "older_than": "7 days"},
                '{"compressed_chunks":' + str(rint(1, 50)) + '}',
                f"chunks on `{tbl}` compressed")
    if name == "prometheus_query":
        q = pick(PROM_QUERIES)
        prompt = pick([
            f"Run a Prometheus instant query: `{q}`",
            f"What does this PromQL return right now? `{q}`",
            f"Hit Prometheus with: `{q}`",
        ])
        return (prompt, {"query": q},
                '{"status":"success","data":{"resultType":"vector","result":[{"value":[1715126400,"' + str(rfloat(0, 1000)) + '"]}]}}',
                "PromQL value returned")
    if name == "prometheus_range_query":
        q = pick(PROM_QUERIES); step = pick(["15s", "1m", "5m"])
        prompt = pick([
            f"Run a Prometheus range query for the last hour at step {step}: `{q}`",
            f"Pull a 1h time-series for `{q}` from Prometheus, step {step}.",
            f"Range-query Prometheus: `{q}` between 00:00 and 01:00 step={step}.",
        ])
        return (prompt, {"query": q, "start": "2026-05-08T00:00:00Z", "end": "2026-05-08T01:00:00Z", "step": step},
                '{"status":"success","data":{"resultType":"matrix","result":[{"values":[[1715126400,"1.5"],[1715126415,"1.7"]]}]}}',
                "range query series returned")

    if name == "elastic_create_index":
        idx = pick(["logs-app", "events", "products-v8", "users-search"]) + "-" + str(rint(1, 99))
        prompt = pick([
            f"Create the Elasticsearch index `{idx}` with 3 shards and 1 replica.",
            f"Stand up a new ES index named `{idx}` (3/1 sharding).",
            f"Provision Elasticsearch index `{idx}` — 3 shards, 1 replica.",
        ])
        return (prompt, {"index": idx, "shards": 3, "replicas": 1},
                '{"acknowledged":true,"index":"' + idx + '"}',
                f"index `{idx}` created")
    if name == "elastic_bulk_index":
        idx = pick(ES_INDICES); n = rint(50, 5000)
        prompt = pick([
            f"Bulk-index {n} docs into `{idx}`.",
            f"Push {n} documents to Elasticsearch index `{idx}` via _bulk.",
            f"Send a bulk batch of {n} records into ES `{idx}`.",
        ])
        return (prompt, {"index": idx, "doc_count": n},
                '{"took":' + str(rint(20, 800)) + ',"errors":false,"items_count":' + str(n) + '}',
                f"{n} docs indexed into `{idx}`")
    if name == "elastic_reindex":
        src = pick(ES_INDICES); dst = src.replace("2026.05", "2026.06") if "2026.05" in src else src + "-v2"
        prompt = pick([
            f"Reindex from `{src}` into `{dst}`.",
            f"Copy `{src}` over to `{dst}` via the reindex API.",
            f"Migrate ES data: `{src}` → `{dst}`.",
        ])
        return (prompt, {"source": src, "dest": dst},
                '{"task":"' + str(rint(10**5, 10**7)) + ':2","total":' + str(rint(1000, 9_000_000)) + '}',
                f"reindex from `{src}` queued")
    if name == "opensearch_query":
        idx = pick(ES_INDICES)
        prompt = pick([
            f"Query OpenSearch index `{idx}` for the most recent error logs.",
            f"In OpenSearch, surface the latest ERROR-level entries from `{idx}`.",
            f"Search `{idx}` on OpenSearch — level=ERROR, sort newest first.",
        ])
        return (prompt, {"index": idx, "query": {"match": {"level": "ERROR"}}, "size": 20, "sort": [{"@timestamp": "desc"}]},
                '{"hits":{"total":' + str(rint(0, 99999)) + ',"hits":[{"_id":"a1","_source":{"level":"ERROR"}}]}}',
                f"top hits returned from `{idx}`")
    if name == "meilisearch_search":
        idx = pick(["movies", "books", "products", "people"]); q = pick(SEARCH_QUERIES)
        prompt = pick([
            f"Search Meilisearch `{idx}` for `{q}`.",
            f"Run a Meilisearch lookup against `{idx}` with query `{q}`.",
            f"Pull the top 20 hits for `{q}` from the Meili `{idx}` index.",
        ])
        return (prompt, {"index": idx, "q": q, "limit": 20},
                '{"hits":[{"id":1,"title":"sample"}],"estimatedTotalHits":' + str(rint(0, 5000)) + '}',
                f"Meilisearch hits for `{q}` returned")
    if name == "typesense_search":
        coll = pick(["catalog", "kb", "events_search"]); q = pick(SEARCH_QUERIES)
        prompt = pick([
            f"Run a Typesense search in `{coll}` for `{q}`, prefix match on title.",
            f"Search the `{coll}` Typesense collection — `{q}`, query_by title.",
            f"Hit Typesense `{coll}` with `{q}` (search title, 25 per page).",
        ])
        return (prompt, {"collection": coll, "q": q, "query_by": "title", "per_page": 25},
                '{"found":' + str(rint(0, 9999)) + ',"hits":[{"document":{"id":"42"}}]}',
                f"Typesense returned hits for `{q}`")

    if name == "debezium_create_connector":
        conn = "dbz-pg-" + pick(["orders", "users", "events", "billing"]) + "-" + str(rint(1, 99))
        prompt = pick([
            f"Stand up a Debezium Postgres connector named `{conn}` for the orders table.",
            f"Register a new Debezium source `{conn}` against `public.orders`.",
            f"Spin up Debezium connector `{conn}` to stream Postgres orders into Kafka.",
        ])
        return (prompt, {"name": conn, "config": {"connector.class": "io.debezium.connector.postgresql.PostgresConnector", "database.hostname": "pg.prod", "table.include.list": "public.orders"}},
                '{"name":"' + conn + '","tasks":[{"state":"RUNNING"}]}',
                f"`{conn}` connector running")
    if name == "debezium_pause_connector":
        conn = pick(KCONNECT_CONNECTORS)
        prompt = pick([
            f"Pause the `{conn}` connector while we cut over the source.",
            f"Halt `{conn}` on Kafka Connect for now.",
            f"Suspend the `{conn}` Debezium task — we're doing a maintenance window.",
        ])
        return (prompt, {"connector": conn},
                '{"paused":true,"connector":"' + conn + '"}',
                f"`{conn}` paused")
    if name == "kafka_connect_status":
        conn = pick(KCONNECT_CONNECTORS)
        prompt = pick([
            f"What's the status of `{conn}`?",
            f"Check Kafka Connect — is `{conn}` running?",
            f"Pull the connector state for `{conn}`.",
        ])
        return (prompt, {"connector": conn},
                '{"name":"' + conn + '","connector":{"state":"RUNNING"},"tasks":[{"id":0,"state":"RUNNING"}]}',
                f"`{conn}` is healthy")
    if name == "kafka_topic_create":
        tp = pick(["cdc.orders", "cdc.users", "events.clickstream", "metrics.app"]) + "." + pick(["v1", "v2"])
        prompt = pick([
            f"Create Kafka topic `{tp}` with 12 partitions, RF=3.",
            f"Provision a new topic `{tp}` (12 partitions, replication factor 3).",
            f"Add `{tp}` to Kafka — 12 partitions, RF 3.",
        ])
        return (prompt, {"topic": tp, "partitions": 12, "replication_factor": 3},
                '{"created":true,"topic":"' + tp + '"}',
                f"topic `{tp}` created")
    if name == "kafka_topic_describe":
        tp = pick(TOPICS_CDC)
        prompt = pick([
            f"Describe the Kafka topic `{tp}`.",
            f"Show me the partition count and configs for `{tp}`.",
            f"What's the layout of `{tp}` — partitions, RF, retention?",
        ])
        return (prompt, {"topic": tp},
                '{"topic":"' + tp + '","partitions":' + str(rint(3, 24)) + ',"replication_factor":3,"configs":{"cleanup.policy":"compact"}}',
                f"topic `{tp}` described")

    if name == "feast_get_online_features":
        ent = pick(ENTITIES); fl = random.sample(FEATURES, 3); eid = rint(1, 9999)
        prompt = pick([
            f"Pull online features {fl} for entity `{ent}={eid}`.",
            f"Fetch Feast online features {fl} for {ent} {eid}.",
            f"Get the live `{ent}` feature vector ({fl}) for id={eid}.",
        ])
        return (prompt, {"entity": ent, "entity_id": eid, "features": fl},
                '{"results":{"' + fl[0] + '":' + str(rfloat(0, 1000)) + ',"' + fl[1] + '":' + str(rfloat(0, 1000)) + ',"' + fl[2] + '":' + str(rfloat(0, 1000)) + '}}',
                "online features served")
    if name == "feast_materialize":
        fv = pick(["user_stats_30d_fv", "session_fv", "fraud_fv", "ltv_fv"])
        prompt = pick([
            f"Materialize `{fv}` from offline to online store for the last 24h.",
            f"Run Feast materialize for `{fv}` covering yesterday's data.",
            f"Push `{fv}` to the online store — window 2026-05-07 to 2026-05-08.",
        ])
        return (prompt, {"feature_view": fv, "start": "2026-05-07T00:00:00Z", "end": "2026-05-08T00:00:00Z"},
                '{"materialized":true,"rows":' + str(rint(1000, 9_000_000)) + '}',
                f"`{fv}` materialized")
    if name == "feast_apply":
        prompt = pick([
            "Apply the latest Feast feature definitions in `feature_repo/`.",
            "Run `feast apply` against the `feature_repo/` directory.",
            "Sync the Feast registry from `feature_repo/`.",
        ])
        return (prompt, {"repo_path": "feature_repo/"},
                '{"registered_feature_views":' + str(rint(1, 30)) + ',"registered_entities":' + str(rint(1, 10)) + '}',
                "Feast registry updated")
    if name == "tecton_get_features":
        fs = pick(["fraud:user_score_v3", "growth:dau_features", "search:rank_features"]); uid = rint(1, 9999)
        prompt = pick([
            f"Fetch Tecton features from `{fs}` for user_id={uid}.",
            f"Pull the `{fs}` Tecton feature service for user {uid}.",
            f"Score user {uid} using Tecton service `{fs}`.",
        ])
        return (prompt, {"feature_service": fs, "join_keys": {"user_id": uid}},
                '{"features":{"score":' + str(rfloat(0, 1)) + ',"is_high_risk":false}}',
                f"Tecton features for `{fs}` returned")

    if name == "datahub_lookup_dataset":
        ds = pick(DATASETS_CAT)
        prompt = pick([
            f"Look up `{ds}` in DataHub and return its owners and tags.",
            f"What does DataHub say about `{ds}`? I need owners + tags.",
            f"Pull DataHub metadata for `{ds}`.",
        ])
        return (prompt, {"urn": f"urn:li:dataset:(urn:li:dataPlatform:postgres,{ds},PROD)"},
                '{"owners":["data-platform"],"tags":["pii","tier-1"],"description":"Daily orders fact"}',
                f"DataHub metadata for `{ds}` returned")
    if name == "datahub_set_owner":
        ds = pick(DATASETS_CAT); owner = pick(["analytics-eng", "growth-team", "billing-eng", "ml-platform"])
        prompt = pick([
            f"Set the owner of `{ds}` to `{owner}` in DataHub.",
            f"Reassign `{ds}` ownership to `{owner}` on DataHub.",
            f"DataHub: change `{ds}` owner → `{owner}`.",
        ])
        return (prompt, {"urn": f"urn:li:dataset:(urn:li:dataPlatform:postgres,{ds},PROD)", "owner": owner},
                '{"updated":true,"owner":"' + owner + '"}',
                f"`{ds}` ownership set to `{owner}`")
    if name == "amundsen_search_table":
        q = pick(["orders", "user", "revenue", "subscription", "event"])
        prompt = pick([
            f"Search Amundsen for tables matching `{q}`.",
            f"On Amundsen, find datasets whose names mention `{q}`.",
            f"Look up `{q}`-related tables in Amundsen.",
        ])
        return (prompt, {"query": q, "page_size": 10},
                '{"tables":[{"name":"' + pick(DATASETS_CAT).split(".")[-1] + '","schema":"' + pick(["analytics", "marts"]) + '"}],"total":' + str(rint(1, 50)) + '}',
                f"Amundsen results for `{q}` returned")

    if name == "ge_run_checkpoint":
        cp = pick(["orders_checkpoint", "users_checkpoint", "events_checkpoint", "billing_checkpoint"])
        prompt = pick([
            f"Run the Great Expectations checkpoint `{cp}`.",
            f"Kick off GE checkpoint `{cp}` against the latest batch.",
            f"Validate today's load by running `{cp}` in Great Expectations.",
        ])
        return (prompt, {"checkpoint": cp},
                '{"success":true,"validation_results":{"successful_expectations":' + str(rint(20, 80)) + ',"failed_expectations":0}}',
                f"`{cp}` validations passed")
    if name == "ge_validate_expectation":
        col = pick(["amount_cents", "user_id", "email", "created_at"]); ds = pick(DATASETS_CAT)
        prompt = pick([
            f"Validate that `{col}` has no nulls in `{ds}`.",
            f"Run a not-null check on `{col}` for asset `{ds}`.",
            f"GE expectation: ensure `{ds}.{col}` is fully populated.",
        ])
        return (prompt, {"expectation": "expect_column_values_to_not_be_null", "column": col, "asset": ds},
                '{"success":true,"unexpected_count":0,"element_count":' + str(rint(1000, 999999)) + '}',
                f"`{col}` not-null check passed")
    if name == "dbt_test_freshness":
        src = pick(["stripe.charges", "salesforce.accounts", "ga.events", "shopify.orders"])
        prompt = pick([
            f"Run dbt source freshness checks on `{src}`.",
            f"Is `{src}` fresh? Run `dbt source freshness` for it.",
            f"Check freshness of `{src}` via dbt.",
        ])
        return (prompt, {"source": src},
                '{"sources":[{"name":"' + src + '","status":"pass","max_loaded_at":"2026-05-08T03:14:00Z"}]}',
                f"`{src}` freshness check passed")
    if name == "soda_scan":
        ds = pick(DATASETS_CAT)
        prompt = pick([
            f"Run a Soda scan against `{ds}`.",
            f"Trigger Soda checks on `{ds}`.",
            f"Run the Soda data-quality scan defined for `{ds}`.",
        ])
        return (prompt, {"data_source": ds.split(".")[0], "scan_definition": ds.split(".")[-1] + "_scan.yml"},
                '{"checks_passed":' + str(rint(5, 40)) + ',"checks_failed":0,"checks_warned":' + str(rint(0, 3)) + '}',
                f"Soda scan on `{ds}` clean")

    if name == "mdm_match_records":
        ent = pick(["customer", "supplier", "patient", "company"])
        prompt = pick([
            f"Run an MDM match pass on the `{ent}` golden record set.",
            f"Kick off MDM matching for `{ent}` records, threshold 0.85.",
            f"Find duplicate `{ent}` records via Jaro–Winkler matching.",
        ])
        return (prompt, {"entity_type": ent, "threshold": 0.85, "algorithm": "jaro_winkler"},
                '{"pairs_examined":' + str(rint(10000, 1_000_000)) + ',"matches":' + str(rint(100, 9999)) + ',"merge_candidates":' + str(rint(50, 999)) + '}',
                f"`{ent}` MDM match complete")
    if name == "mdm_merge_records":
        a = "rec_" + str(rint(10000, 99999)); b = "rec_" + str(rint(10000, 99999))
        prompt = pick([
            f"Merge `{a}` and `{b}` as duplicates, keeping the older record as survivor.",
            f"In MDM, fold `{b}` into `{a}` (older wins).",
            f"Resolve duplicates: survivor `{a}`, retire `{b}`.",
        ])
        return (prompt, {"survivor_id": a, "merged_id": b, "rule": "older_wins"},
                '{"merged":true,"survivor":"' + a + '","retired":"' + b + '"}',
                "records merged")
    if name == "dedup_table":
        tbl = pick(DATASETS_CAT); col = pick(["email", "phone", "external_id"])
        prompt = pick([
            f"Deduplicate `{tbl}` on `{col}`, keeping the most recent row.",
            f"Drop dupes from `{tbl}` using `{col}` as the key.",
            f"Run a dedup pass on `{tbl}` keyed by `{col}`, latest survives.",
        ])
        return (prompt, {"table": tbl, "key_columns": [col], "keep": "latest"},
                '{"duplicates_removed":' + str(rint(10, 99999)) + ',"final_rows":' + str(rint(10000, 9_000_000)) + '}',
                f"`{tbl}` deduped on `{col}`")

    if name == "alembic_upgrade":
        prompt = pick([
            "Bring the Alembic schema up to head.",
            "Run `alembic upgrade head`.",
            "Apply all pending Alembic revisions.",
        ])
        return (prompt, {"revision": "head"},
                '{"applied":["' + str(rint(1000, 9999)) + '_add_user_flags","' + str(rint(1000, 9999)) + '_index_orders_status"],"current":"head"}',
                "Alembic migrated to head")
    if name == "alembic_downgrade":
        prompt = pick([
            "Roll Alembic back one revision.",
            "Run `alembic downgrade -1`.",
            "Undo the most recent Alembic migration.",
        ])
        return (prompt, {"revision": "-1"},
                '{"reverted":["' + str(rint(1000, 9999)) + '_index_orders_status"],"current":"' + str(rint(1000, 9999)) + '_add_user_flags"}',
                "Alembic downgraded one step")
    if name == "flyway_info":
        sch = pick(["public", "analytics", "marts"])
        prompt = pick([
            f"Show Flyway migration status for the `{sch}` schema.",
            f"Run `flyway info` against `{sch}`.",
            f"What Flyway revisions have been applied to `{sch}`?",
        ])
        return (prompt, {"schema": sch},
                '{"migrations":[{"version":"V12","state":"Success"},{"version":"V13","state":"Pending"}]}',
                "Flyway status reported")
    if name == "schema_compare":
        a = pick(["staging", "prod"]); b = "prod" if a == "staging" else "staging"
        prompt = pick([
            f"Diff the schema between `{a}` and `{b}` and tell me what changed.",
            f"Compare schemas: `{a}` vs `{b}`.",
            f"What's drifted between `{a}` and `{b}`?",
        ])
        return (prompt, {"left": a, "right": b},
                '{"added_tables":["' + pick(["new_signups", "ml_predictions"]) + '"],"removed_tables":[],"changed_columns":[{"table":"orders","column":"status","change":"type"}]}',
                f"`{a}` vs `{b}` schema diff returned")

    if name == "explain_analyze":
        sql = "SELECT user_id, SUM(amount_cents) FROM analytics.orders WHERE created_at > NOW() - INTERVAL '7 days' GROUP BY 1"
        prompt = pick([
            f"Run EXPLAIN ANALYZE on: `{sql}`",
            f"Get the actual plan with timings for: `{sql}`",
            f"Profile this query end-to-end with EXPLAIN ANALYZE: `{sql}`",
        ])
        return (prompt, {"sql": sql, "format": "json"},
                '{"plan":{"Node Type":"HashAggregate","Total Cost":' + str(rfloat(100, 99999)) + ',"Actual Total Time":' + str(rfloat(10, 5000)) + '}}',
                "execution plan returned with timings")
    if name == "list_partitions":
        tbl = pick(DATASETS_CAT)
        prompt = pick([
            f"List partitions on `{tbl}`.",
            f"What partitions exist for `{tbl}`?",
            f"Show me every partition under `{tbl}`.",
        ])
        return (prompt, {"table": tbl},
                '{"partitions":["' + '","'.join(random.sample(PARTITIONS, 4)) + '"],"count":4}',
                f"partitions of `{tbl}` listed")
    if name == "drop_partition":
        tbl = pick(DATASETS_CAT); part = pick(PARTITIONS)
        prompt = pick([
            f"Drop partition `{part}` from `{tbl}`.",
            f"Remove the `{part}` partition on `{tbl}`.",
            f"Detach and drop `{part}` off `{tbl}`.",
        ])
        return (prompt, {"table": tbl, "partition": part},
                '{"dropped":true,"partition":"' + part + '"}',
                f"partition `{part}` dropped from `{tbl}`")
    if name == "create_partition":
        tbl = pick(DATASETS_CAT); month = rint(6, 12); part = f"p_2026_{month:02d}"
        prompt = pick([
            f"Create the next monthly partition `{part}` on `{tbl}`.",
            f"Add a 2026-{month:02d} partition (`{part}`) to `{tbl}`.",
            f"Provision partition `{part}` on `{tbl}` covering month {month}.",
        ])
        return (prompt, {"table": tbl, "partition_name": part, "range_start": f"2026-{month:02d}-01", "range_end": f"2026-{(month%12)+1:02d}-01"},
                '{"created":true,"partition":"' + part + '"}',
                f"partition `{part}` provisioned on `{tbl}`")
    if name == "refresh_materialized_view":
        mv = "mv_" + pick(["daily_revenue", "active_users", "top_skus", "session_funnel"])
        prompt = pick([
            f"Refresh the materialized view `{mv}` concurrently.",
            f"Run REFRESH MATERIALIZED VIEW CONCURRENTLY `{mv}`.",
            f"Force a refresh of `{mv}` without locking readers.",
        ])
        return (prompt, {"view": mv, "concurrently": True},
                '{"refreshed":true,"view":"' + mv + '","duration_ms":' + str(rint(800, 60000)) + '}',
                f"`{mv}` refreshed")
    if name == "create_materialized_view":
        mv = "mv_" + pick(["churn_30d", "ltv_v2", "weekly_active_devices"])
        sql = "SELECT user_id, COUNT(*) FROM analytics.events GROUP BY 1"
        prompt = pick([
            f"Create materialized view `{mv}` defined as: `{sql}`",
            f"Build a new MV called `{mv}` from: `{sql}`",
            f"Persist this query as a materialized view `{mv}`: `{sql}`",
        ])
        return (prompt, {"name": mv, "sql": sql, "with_data": True},
                '{"created":true,"view":"' + mv + '"}',
                f"`{mv}` created")

    if name == "iceberg_create_table":
        tbl = pick(ICEBERG_TABLES) + "_v" + str(rint(2, 5))
        prompt = pick([
            f"Create the Iceberg table `{tbl}` partitioned by day(created_at).",
            f"Provision Iceberg table `{tbl}` with daily partitioning on created_at.",
            f"Set up `{tbl}` in the Iceberg catalog, format v2, day partition.",
        ])
        return (prompt, {"table": tbl, "schema": ["id:long", "created_at:timestamp"], "partition_spec": "day(created_at)"},
                '{"created":true,"table":"' + tbl + '","format_version":2}',
                f"Iceberg table `{tbl}` created")
    if name == "iceberg_snapshot_history":
        tbl = pick(ICEBERG_TABLES)
        prompt = pick([
            f"Show the snapshot history for `{tbl}`.",
            f"List Iceberg snapshots on `{tbl}`.",
            f"Pull the commit log of `{tbl}` from the Iceberg catalog.",
        ])
        return (prompt, {"table": tbl},
                '{"snapshots":[{"snapshot_id":' + str(rint(10**15, 10**16)) + ',"committed_at":"2026-05-08T01:00:00Z","operation":"append"}]}',
                f"`{tbl}` snapshot history returned")
    if name == "iceberg_expire_snapshots":
        tbl = pick(ICEBERG_TABLES)
        prompt = pick([
            f"Expire snapshots older than 7 days on `{tbl}`.",
            f"Run snapshot expiration on Iceberg `{tbl}` (7d retention).",
            f"Clean up old Iceberg snapshots beyond a week on `{tbl}`.",
        ])
        return (prompt, {"table": tbl, "older_than": "7 days"},
                '{"deleted_data_files":' + str(rint(5, 500)) + ',"deleted_manifest_files":' + str(rint(1, 50)) + '}',
                f"old snapshots cleaned on `{tbl}`")
    if name == "delta_optimize":
        path = pick(DELTA_PATHS)
        prompt = pick([
            f"Run OPTIMIZE on the Delta table at `{path}` with ZORDER on user_id.",
            f"Compact the Delta files at `{path}`, zorder by user_id.",
            f"Trigger Delta OPTIMIZE for `{path}` (ZORDER user_id).",
        ])
        return (prompt, {"path": path, "zorder_by": ["user_id"]},
                '{"files_added":' + str(rint(1, 50)) + ',"files_removed":' + str(rint(50, 500)) + '}',
                f"Delta `{path}` optimized")
    if name == "delta_vacuum":
        path = pick(DELTA_PATHS)
        prompt = pick([
            f"VACUUM the Delta table at `{path}` keeping 168h of history.",
            f"Run VACUUM on Delta `{path}` (retention 168 hours).",
            f"Garbage-collect old files on `{path}`, keep last 7 days.",
        ])
        return (prompt, {"path": path, "retention_hours": 168},
                '{"deleted_files":' + str(rint(10, 9999)) + '}',
                f"Delta `{path}` vacuumed")
    if name == "delta_time_travel":
        path = pick(DELTA_PATHS); v = rint(1, 999)
        prompt = pick([
            f"Read the Delta table `{path}` as of version {v}.",
            f"Time-travel into Delta `{path}` at version {v}.",
            f"Pull `{path}` at snapshot v{v} from Delta.",
        ])
        return (prompt, {"path": path, "version": v},
                '{"rows":' + str(rint(1000, 9_000_000)) + ',"version":' + str(v) + '}',
                f"Delta `{path}` snapshot @v{v} read")

    raise KeyError(name)


SYSTEM_VARIANTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a vector-database and search assistant. Always call tools when retrieving data.",
    "You are a data infrastructure assistant covering graph, time-series, and lakehouse systems.",
    "You are a data quality and catalog assistant. Use tools for inspections.",
    "You are a streaming and CDC assistant. Prefer tool calls over guessing.",
    "You are a feature store and ML data assistant.",
    "You are a lakehouse operations assistant for Iceberg and Delta.",
    "You are a database operations assistant. Inspect or change state via tools, never freehand.",
]
