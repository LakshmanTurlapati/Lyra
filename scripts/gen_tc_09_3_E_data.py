"""Per-tool prompt + arg + result builders for TC-E."""
import random

SCHEMAS = ["public", "analytics", "staging", "raw", "ods", "dwh", "marts", "core", "events", "billing", "ops", "audit"]
TABLES = ["orders", "users", "events_raw", "sessions", "page_views", "transactions", "invoices", "customers", "products", "shipments", "subscriptions", "logins", "signups", "clicks", "impressions", "leads", "accounts", "tickets", "messages", "carts", "line_items", "refunds", "audit_log", "feature_flags", "device_events", "ab_assignments"]
COLS = ["user_id", "created_at", "updated_at", "email", "status", "amount_cents", "country", "tenant_id", "session_id", "event_name", "device_type", "campaign_id", "order_id", "sku", "price", "quantity", "ip", "referrer"]
DBS = ["app_prod", "warehouse", "analytics_db", "billing_prod", "logs", "events_db"]
WAREHOUSES = ["WH_REPORTING", "WH_ELT", "WH_AD_HOC", "WH_FINANCE", "WH_DBT_PROD"]
DATASETS = ["analytics_prod", "raw_events", "marketing", "finance_dwh", "product_metrics"]
TOPICS = ["orders.created", "users.signup", "payments.processed", "inventory.updated", "clickstream", "audit.events"]
METRICS = ["dau", "mau", "revenue_usd", "checkout_conversion", "p95_latency_ms", "error_rate", "arpu", "ltv", "churn_rate", "signup_rate"]
PIPELINES = ["nightly_orders_load", "events_ingest", "stripe_sync", "salesforce_sync", "ga4_export", "dim_user_build"]
DASHBOARDS = ["dash_growth_v3", "dash_finance_q2", "dash_ops_health", "dash_exec_overview"]


def rint(a, b): return random.randint(a, b)
def rfloat(a, b, p=2): return round(random.uniform(a, b), p)


def pick(seq): return random.choice(seq)


def gen_for_tool(name):
    """Returns (user_prompt, args_dict, tool_result_json_str, summary_fragment)."""
    s = pick(SCHEMAS)
    t = pick(TABLES)
    if name == "sql_query":
        col = pick(COLS); n = rint(100, 9_000_000)
        sql = f"SELECT {col}, COUNT(*) AS n FROM {s}.{t} GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
        prompt = pick([f"Run: `{sql}`", f"Can you run this query? {sql}", f"Execute this SQL against the warehouse:\n{sql}"])
        return prompt, {"sql": sql}, f'{{"rows":[{{"{col}":"a","n":{n}}}],"row_count":1}}', f"top {col} by count returned"
    if name == "sql_explain":
        sql = f"SELECT * FROM {s}.{t} WHERE {pick(COLS)} = $1"
        return f"Show the plan for: {sql}", {"sql": sql}, '{"plan":"Index Scan using ix_a on '+t+' (cost=0.42..8.44 rows=1)"}', "index scan with low cost"
    if name == "sql_count_rows":
        n = rint(1000, 50_000_000)
        return f"How many rows are in {s}.{t}?", {"schema": s, "table": t}, f'{{"count":{n}}}', f"{n:,} rows in `{s}.{t}`"
    if name == "truncate_table":
        return pick([f"Truncate {s}.{t}.", f"Please empty out the {s}.{t} table.", f"Wipe all rows from {s}.{t}."]), {"schema": s, "table": t}, '{"truncated":true,"rows_removed":'+str(rint(1000,500000))+'}', f"`{s}.{t}` truncated"
    if name == "drop_table":
        return f"Drop {s}.{t} cascade.", {"schema": s, "table": t, "cascade": True}, '{"dropped":true}', f"`{s}.{t}` dropped"
    if name == "create_index":
        cols = random.sample(COLS, 2); idx = f"ix_{t}_{cols[0]}"
        return f"Create an index on {s}.{t}({', '.join(cols)}).", {"table": f"{s}.{t}", "columns": cols, "name": idx}, '{"created":true,"name":"'+idx+'"}', f"index `{idx}` created"
    if name == "drop_index":
        idx = f"ix_{t}_{pick(COLS)}"
        return f"Drop the index {idx}.", {"name": idx}, '{"dropped":true}', f"index `{idx}` dropped"
    if name == "list_tables":
        return f"List all tables in the {s} schema.", {"schema": s}, '{"tables":["'+'","'.join(random.sample(TABLES,5))+'"]}', f"5 tables in `{s}`"
    if name == "describe_table":
        return f"Describe {s}.{t}.", {"schema": s, "table": t}, '{"columns":[{"name":"id","type":"bigint"},{"name":"created_at","type":"timestamptz"}]}', f"schema for `{s}.{t}` returned"
    if name == "vacuum_analyze":
        return f"Run VACUUM ANALYZE on {s}.{t}.", {"table": f"{s}.{t}"}, '{"ok":true,"duration_ms":'+str(rint(120,9000))+'}', "vacuum complete"
    if name == "redis_get":
        k = f"user:{rint(1000,99999)}:session"
        return f"What's the value of `{k}`?", {"key": k}, '{"value":"'+("sess_"+str(rint(10**9,10**10)))+'"}', f"value for `{k}` returned"
    if name == "redis_set":
        k = f"flag:{pick(['ab_test','beta','rollout'])}:{rint(1,99)}"; v = pick(["on","off","true","false"])
        return f"Set `{k}` to `{v}` with TTL 3600.", {"key": k, "value": v, "ttl_s": 3600}, '{"ok":true}', f"`{k}` set"
    if name == "redis_keys":
        p = f"user:*:{pick(['cart','session','token'])}"
        return f"List redis keys matching `{p}`.", {"pattern": p}, '{"keys":["user:42:cart","user:91:cart"],"count":2}', "2 keys matched"
    if name == "redis_expire":
        k = f"cache:{pick(TABLES)}:{rint(1,9999)}"
        return f"Expire `{k}` in 600s.", {"key": k, "ttl_s": 600}, '{"ok":true,"ttl_s":600}', f"TTL set on `{k}`"
    if name == "redis_lpush":
        k = f"queue:{pick(['emails','jobs','webhooks'])}"; v = f"job_{rint(10000,99999)}"
        return f"Push {v} onto {k}.", {"key": k, "value": v}, '{"length":'+str(rint(1,500))+'}', f"pushed to `{k}`"
    if name == "redis_zadd":
        k = "leaderboard:weekly"; m = f"user_{rint(1,9999)}"; sc = rfloat(0,10000)
        return f"Add {m} to {k} with score {sc}.", {"key": k, "member": m, "score": sc}, '{"added":1}', f"{m} added"
    if name == "redis_del":
        k = f"cache:{pick(TABLES)}:{rint(1,9999)}"
        return f"Delete redis key {k}.", {"key": k}, '{"deleted":1}', f"`{k}` deleted"
    if name == "mongo_find":
        coll = pick(TABLES); f = {"status": pick(["active","pending","cancelled"])}
        return f"Find {coll} where status is {f['status']}, limit 10.", {"collection": coll, "filter": f, "limit": 10}, '{"docs":[{"_id":"a","status":"'+f["status"]+'"}],"count":1}', f"matched docs in `{coll}`"
    if name == "mongo_insert":
        coll = pick(TABLES); doc = {"name": "test", "value": rint(1,100)}
        return f"Insert a doc into {coll}: {doc}.", {"collection": coll, "document": doc}, '{"inserted_id":"6634abf1c2"}', "inserted"
    if name == "mongo_aggregate":
        coll = pick(TABLES)
        pipe = [{"$match": {"status": "active"}}, {"$group": {"_id": "$country", "n": {"$sum": 1}}}]
        return f"Group {coll} by country where active.", {"collection": coll, "pipeline": pipe}, '{"results":[{"_id":"US","n":'+str(rint(100,9999))+'},{"_id":"GB","n":'+str(rint(50,3000))+'}]}', "aggregation returned"
    if name == "mongo_update":
        coll = pick(TABLES)
        return f"In {coll}, set status='archived' for items older than 90 days.", {"collection": coll, "filter": {"created_at": {"$lt": "2026-02-07"}}, "update": {"$set": {"status":"archived"}}}, '{"matched":'+str(rint(10,5000))+',"modified":'+str(rint(10,5000))+'}', "documents archived"
    if name == "mongo_delete":
        coll = pick(TABLES)
        return f"Delete from {coll} where status='deleted'.", {"collection": coll, "filter": {"status":"deleted"}}, '{"deleted":'+str(rint(1,500))+'}', "rows deleted"
    if name == "bq_run_query":
        ds = pick(DATASETS); sql = f"SELECT COUNT(*) FROM `{ds}.{pick(TABLES)}` WHERE _PARTITIONDATE = CURRENT_DATE()"
        return f"Run on BigQuery: {sql}", {"sql": sql, "dry_run": False}, '{"rows":[{"f0_":'+str(rint(10000,9999999))+'}],"bytes_billed":'+str(rint(1024,10**8))+'}', "BigQuery returned count"
    if name == "bq_create_dataset":
        d = "lyra_"+pick(["sandbox","exp","ml","tmp"])+"_"+str(rint(1,99))
        return f"Create BigQuery dataset {d} in US.", {"dataset_id": d, "location": "US"}, '{"created":true,"dataset_id":"'+d+'"}', f"dataset `{d}` created"
    if name == "bq_load_csv":
        gcs = f"gs://lyra-imports/{pick(TABLES)}.csv"; tbl = f"{pick(DATASETS)}.{pick(TABLES)}"
        return f"Load {gcs} into {tbl}.", {"gcs_uri": gcs, "table": tbl}, '{"loaded_rows":'+str(rint(1000,9_000_000))+'}', f"loaded into `{tbl}`"
    if name == "snowflake_warehouse_resume":
        w = pick(WAREHOUSES)
        return f"Resume Snowflake warehouse {w}.", {"warehouse": w}, '{"state":"STARTED"}', f"`{w}` resumed"
    if name == "snowflake_warehouse_suspend":
        w = pick(WAREHOUSES)
        return f"Suspend Snowflake warehouse {w} to save credits.", {"warehouse": w}, '{"state":"SUSPENDED"}', f"`{w}` suspended"
    if name == "snowflake_run_query":
        w = pick(WAREHOUSES); sql = f"SELECT COUNT(DISTINCT user_id) FROM {pick(SCHEMAS).upper()}.{pick(TABLES).upper()}"
        return f"Run on Snowflake ({w}): {sql}", {"sql": sql, "warehouse": w}, '{"rows":[[' + str(rint(1000,2_000_000)) + ']]}', "Snowflake count returned"
    if name == "dashboard_create":
        nm = pick(DASHBOARDS) + "_" + str(rint(1,99))
        return f"Create dashboard {nm} with revenue and churn tiles.", {"name": nm, "tiles": ["revenue","churn"]}, '{"dashboard_id":"dash_'+str(rint(10000,99999))+'","name":"'+nm+'"}', f"dashboard `{nm}` created"
    if name == "dashboard_refresh":
        d = "dash_"+str(rint(10000,99999))
        return f"Refresh dashboard {d}.", {"dashboard_id": d}, '{"refreshed":true,"duration_ms":'+str(rint(500,15000))+'}', f"dashboard `{d}` refreshed"
    if name == "metric_get":
        m = pick(METRICS); v = rfloat(0.1, 99999)
        return f"What's the current value of {m} (last 7d)?", {"metric": m, "window": "7d"}, '{"metric":"'+m+'","value":'+str(v)+'}', f"{m} = {v}"
    if name == "metric_alert_set":
        m = pick(METRICS); op = pick([">", "<", ">="]); th = rfloat(1, 5000)
        return f"Alert when {m} {op} {th}.", {"metric": m, "operator": op, "threshold": th}, '{"alert_id":"al_'+str(rint(1000,99999))+'","active":true}', f"alert on `{m}` set"
    if name == "etl_trigger_pipeline":
        p = pick(PIPELINES)
        return f"Kick off the {p} pipeline.", {"pipeline": p}, '{"run_id":"run_'+str(rint(10**6,10**7))+'","status":"queued"}', f"`{p}` queued"
    if name == "etl_status":
        rid = "run_"+str(rint(10**6, 10**7))
        return f"Status of {rid}?", {"run_id": rid}, '{"run_id":"'+rid+'","status":"'+pick(["running","success","failed"])+'"}', "status returned"
    if name == "kafka_publish":
        tp = pick(TOPICS); pl = {"id": rint(1,9999), "type": "evt"}
        return f"Publish a test message to {tp}.", {"topic": tp, "key": "k1", "payload": pl}, '{"offset":'+str(rint(1000,999999))+',"partition":'+str(rint(0,7))+'}', f"published to `{tp}`"
    if name == "kafka_consume":
        tp = pick(TOPICS); g = "lyra-consumer-"+str(rint(1,99))
        return f"Read up to 5 messages from {tp} (group {g}).", {"topic": tp, "group": g, "max_messages": 5}, '{"messages":[{"offset":4421,"value":"{}"}],"count":1}', "messages fetched"
    if name == "clickhouse_query":
        sql = f"SELECT count() FROM {pick(TABLES)} WHERE event_date = today()"
        return f"Run on ClickHouse: {sql}", {"sql": sql}, '{"rows":[['+str(rint(10000,99_000_000))+']],"elapsed_ms":'+str(rint(20,800))+'}', "ClickHouse count returned"
    if name == "pg_show_locks":
        db = pick(DBS)
        return f"Show current locks on {db}.", {"database": db}, '{"locks":[{"pid":'+str(rint(100,99999))+',"mode":"AccessExclusiveLock","relation":"'+pick(TABLES)+'"}]}', "locks listed"
    if name == "pg_kill_query":
        pid = rint(1000, 99999)
        return f"Kill the runaway query, pid {pid}.", {"pid": pid}, '{"cancelled":true,"pid":'+str(pid)+'}', f"pid {pid} cancelled"
    if name == "db_backup":
        db = pick(DBS); dest = f"s3://lyra-backups/{db}/{rint(2026,2027)}-05-08.dump"
        return f"Back up {db} to {dest}.", {"database": db, "destination": dest}, '{"backup_id":"bk_'+str(rint(10000,99999))+'","size_mb":'+str(rint(50,50000))+'}', "backup created"
    if name == "db_restore":
        db = pick(DBS); bk = "bk_"+str(rint(10000,99999))
        return f"Restore {db} from backup {bk}.", {"database": db, "backup_id": bk}, '{"restored":true}', f"`{db}` restored"
    if name == "migration_apply":
        tg = pick(DBS)
        return f"Apply pending migrations to {tg}.", {"target": tg}, '{"applied":'+str(rint(1,8))+',"current_version":"20260508_'+str(rint(1,9))+'"}', "migrations applied"
    if name == "migration_rollback":
        tg = pick(DBS)
        return f"Roll back the last migration on {tg}.", {"target": tg, "steps": 1}, '{"rolled_back":1}', "rollback complete"
    if name == "cube_query":
        return "Get total revenue and order count by week, last 30d.", {"measures":["Orders.totalRevenue","Orders.count"],"dimensions":["Orders.week"],"time_range":"last 30 days"}, '{"data":[{"Orders.week":"2026-04-27","Orders.count":'+str(rint(1000,9000))+',"Orders.totalRevenue":'+str(rfloat(50000,500000))+'}]}', "cube data returned"
    if name == "looker_run_query":
        return "Run Looker query: orders explore, fields user_id, total_revenue.", {"model": "ecommerce", "explore": "orders", "fields": ["users.id","orders.total_revenue"]}, '{"rows":[{"users.id":42,"orders.total_revenue":'+str(rfloat(10,9999))+'}]}', "Looker rows returned"
    if name == "tableau_refresh_extract":
        wb = pick(["FinanceDaily","GrowthExec","OpsRealtime"])
        return f"Refresh the Tableau extract for {wb}.", {"workbook": wb}, '{"job_id":"tab_'+str(rint(1000,99999))+'","status":"queued"}', f"`{wb}` refresh queued"
    if name == "powerbi_refresh_dataset":
        ds = "ds_"+str(rint(10000,99999))
        return f"Refresh Power BI dataset {ds}.", {"dataset_id": ds}, '{"status":"InProgress"}', f"`{ds}` refresh started"
    if name == "elastic_search":
        idx = pick(["logs-2026.05","events-prod","orders-search"]); q = {"match": {"status": "error"}}
        return f"Search {idx} for status=error.", {"index": idx, "query": q, "size": 10}, '{"hits":'+str(rint(1,9999))+',"top_score":'+str(rfloat(0.5,5))+'}', f"hits in `{idx}`"
    if name == "dbt_run":
        sel = pick(["staging.*","marts.finance.*","tag:nightly","fct_orders+"])
        return f"Run dbt models: {sel}.", {"select": sel, "target": "prod"}, '{"models_run":'+str(rint(1,40))+',"errors":0}', "dbt run ok"
    if name == "dbt_test":
        sel = pick(["staging.*","marts.*","fct_orders"])
        return f"Run dbt tests for {sel}.", {"select": sel}, '{"passed":'+str(rint(10,200))+',"failed":0}', "dbt tests passed"
    if name == "airflow_trigger_dag":
        dg = pick(["nightly_etl","hourly_metrics","churn_scoring","embedding_refresh"])
        return f"Trigger the {dg} DAG.", {"dag_id": dg, "conf": {}}, '{"run_id":"manual__2026-05-08T'+str(rint(0,23)).zfill(2)+':00:00","state":"queued"}', f"`{dg}` triggered"
    if name == "duckdb_query":
        sql = f"SELECT COUNT(*) FROM read_parquet('s3://lyra/{pick(TABLES)}/*.parquet')"
        return f"Run in DuckDB: {sql}", {"sql": sql}, '{"rows":[['+str(rint(1000,9_000_000))+']]}', "DuckDB count returned"
    if name == "cassandra_query":
        ks = pick(["events","sessions","analytics"]); cql = f"SELECT * FROM {pick(TABLES)} WHERE user_id = {rint(1,9999)} LIMIT 10"
        return f"Run on Cassandra ({ks}): {cql}", {"keyspace": ks, "cql": cql}, '{"rows":'+str(rint(0,10))+'}', "CQL rows returned"
    if name == "create_view":
        nm = f"v_{pick(TABLES)}_{pick(['daily','weekly','active'])}"
        sql = f"SELECT user_id, COUNT(*) AS n FROM {pick(SCHEMAS)}.{pick(TABLES)} GROUP BY 1"
        return f"Create view {nm} as: {sql}", {"name": nm, "sql": sql}, '{"created":true}', f"view `{nm}` created"
    if name == "grant_privileges":
        role = pick(["analyst_ro","etl_writer","reporting","data_science"])
        privs = random.sample(["SELECT","INSERT","UPDATE","DELETE","REFERENCES"], 2)
        obj = f"{pick(SCHEMAS)}.{pick(TABLES)}"
        return f"Grant {','.join(privs)} on {obj} to {role}.", {"role": role, "privileges": privs, "object": obj}, '{"granted":true}', f"granted to `{role}`"
    raise KeyError(name)


SYSTEM_VARIANTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a helpful database assistant. Always call tools to inspect or modify data.",
    "You are a data warehouse assistant. Prefer tool calls over guessing.",
    "You are a SQL and analytics assistant.",
    "You are a BI and dashboards assistant.",
    "You are an ETL and pipelines assistant.",
    "You are an infrastructure assistant for databases and caches.",
]
