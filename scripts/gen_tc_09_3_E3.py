"""TC-E batch-07: 500 tool-calling samples for databases/SQL/analytics/DW/BI domain."""
import json
import os
import random
import hashlib

SEED = "1009309E"
random.seed(int(hashlib.md5(SEED.encode()).hexdigest(), 16) % (2**32))

OUT = "/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-09-E.jsonl"

SUFFIXES = [
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

# 50 distinct tools across DB/SQL/analytics/DW/BI/ETL/cache/streaming
TOOLS = {
    "sql_query": ("Run a read-only SQL query.", {"sql": "string"}, ["sql"]),
    "sql_explain": ("Return the EXPLAIN plan for a SQL statement.", {"sql": "string"}, ["sql"]),
    "sql_count_rows": ("Count rows in a table.", {"schema": "string", "table": "string"}, ["table"]),
    "truncate_table": ("Truncate a table, removing all rows.", {"schema": "string", "table": "string"}, ["table"]),
    "drop_table": ("Drop a table.", {"schema": "string", "table": "string", "cascade": "boolean"}, ["table"]),
    "create_index": ("Create an index on a column.", {"table": "string", "columns": "array", "name": "string"}, ["table", "columns"]),
    "drop_index": ("Drop an index by name.", {"name": "string"}, ["name"]),
    "list_tables": ("List tables in a schema.", {"schema": "string"}, ["schema"]),
    "describe_table": ("Return columns and types for a table.", {"schema": "string", "table": "string"}, ["table"]),
    "vacuum_analyze": ("Run VACUUM ANALYZE on a Postgres table.", {"table": "string"}, ["table"]),
    "redis_get": ("Get a Redis key value.", {"key": "string"}, ["key"]),
    "redis_set": ("Set a Redis key.", {"key": "string", "value": "string", "ttl_s": "integer"}, ["key", "value"]),
    "redis_keys": ("List Redis keys matching a pattern.", {"pattern": "string"}, ["pattern"]),
    "redis_expire": ("Set TTL on a Redis key.", {"key": "string", "ttl_s": "integer"}, ["key", "ttl_s"]),
    "redis_lpush": ("Push value onto a Redis list head.", {"key": "string", "value": "string"}, ["key", "value"]),
    "redis_zadd": ("Add member to Redis sorted set with score.", {"key": "string", "member": "string", "score": "number"}, ["key", "member", "score"]),
    "redis_del": ("Delete a Redis key.", {"key": "string"}, ["key"]),
    "mongo_find": ("Find Mongo documents matching a filter.", {"collection": "string", "filter": "object", "limit": "integer"}, ["collection", "filter"]),
    "mongo_insert": ("Insert a Mongo document.", {"collection": "string", "document": "object"}, ["collection", "document"]),
    "mongo_aggregate": ("Run a Mongo aggregation pipeline.", {"collection": "string", "pipeline": "array"}, ["collection", "pipeline"]),
    "mongo_update": ("Update Mongo documents matching filter.", {"collection": "string", "filter": "object", "update": "object"}, ["collection", "filter", "update"]),
    "mongo_delete": ("Delete Mongo documents matching filter.", {"collection": "string", "filter": "object"}, ["collection", "filter"]),
    "bq_run_query": ("Run a BigQuery SQL query.", {"sql": "string", "dry_run": "boolean"}, ["sql"]),
    "bq_create_dataset": ("Create a BigQuery dataset.", {"dataset_id": "string", "location": "string"}, ["dataset_id"]),
    "bq_load_csv": ("Load a CSV from GCS into a BigQuery table.", {"gcs_uri": "string", "table": "string"}, ["gcs_uri", "table"]),
    "snowflake_warehouse_resume": ("Resume a Snowflake warehouse.", {"warehouse": "string"}, ["warehouse"]),
    "snowflake_warehouse_suspend": ("Suspend a Snowflake warehouse.", {"warehouse": "string"}, ["warehouse"]),
    "snowflake_run_query": ("Run a query on Snowflake.", {"sql": "string", "warehouse": "string"}, ["sql"]),
    "dashboard_create": ("Create a BI dashboard.", {"name": "string", "tiles": "array"}, ["name"]),
    "dashboard_refresh": ("Refresh a dashboard.", {"dashboard_id": "string"}, ["dashboard_id"]),
    "metric_get": ("Fetch a metric value.", {"metric": "string", "window": "string"}, ["metric"]),
    "metric_alert_set": ("Set an alert threshold on a metric.", {"metric": "string", "operator": "string", "threshold": "number"}, ["metric", "operator", "threshold"]),
    "etl_trigger_pipeline": ("Trigger an ETL pipeline run.", {"pipeline": "string"}, ["pipeline"]),
    "etl_status": ("Check ETL pipeline run status.", {"run_id": "string"}, ["run_id"]),
    "kafka_publish": ("Publish a message to a Kafka topic.", {"topic": "string", "key": "string", "payload": "object"}, ["topic", "payload"]),
    "kafka_consume": ("Consume messages from a Kafka topic.", {"topic": "string", "group": "string", "max_messages": "integer"}, ["topic", "group"]),
    "clickhouse_query": ("Run a ClickHouse query.", {"sql": "string"}, ["sql"]),
    "pg_show_locks": ("Show current Postgres locks.", {"database": "string"}, ["database"]),
    "pg_kill_query": ("Cancel a running Postgres query by pid.", {"pid": "integer"}, ["pid"]),
    "db_backup": ("Create a backup of a database.", {"database": "string", "destination": "string"}, ["database"]),
    "db_restore": ("Restore a database from a backup.", {"database": "string", "backup_id": "string"}, ["database", "backup_id"]),
    "migration_apply": ("Apply pending schema migrations.", {"target": "string"}, ["target"]),
    "migration_rollback": ("Roll back the last migration.", {"target": "string", "steps": "integer"}, ["target"]),
    "cube_query": ("Run a Cube.js semantic-layer query.", {"measures": "array", "dimensions": "array", "time_range": "string"}, ["measures"]),
    "looker_run_query": ("Run a Looker query against an explore.", {"model": "string", "explore": "string", "fields": "array"}, ["model", "explore", "fields"]),
    "tableau_refresh_extract": ("Refresh a Tableau extract.", {"workbook": "string"}, ["workbook"]),
    "powerbi_refresh_dataset": ("Refresh a Power BI dataset.", {"dataset_id": "string"}, ["dataset_id"]),
    "elastic_search": ("Search an Elasticsearch index.", {"index": "string", "query": "object", "size": "integer"}, ["index", "query"]),
    "dbt_run": ("Run dbt models.", {"select": "string", "target": "string"}, ["select"]),
    "dbt_test": ("Run dbt tests.", {"select": "string"}, ["select"]),
    "airflow_trigger_dag": ("Trigger an Airflow DAG run.", {"dag_id": "string", "conf": "object"}, ["dag_id"]),
    "duckdb_query": ("Run a DuckDB query.", {"sql": "string"}, ["sql"]),
    "cassandra_query": ("Run a CQL query on Cassandra.", {"keyspace": "string", "cql": "string"}, ["keyspace", "cql"]),
    "create_view": ("Create a SQL view.", {"name": "string", "sql": "string"}, ["name", "sql"]),
    "grant_privileges": ("Grant DB privileges to a role.", {"role": "string", "privileges": "array", "object": "string"}, ["role", "privileges", "object"]),
}

# Build tool spec dict for emission
def tool_spec(name):
    desc, props, required = TOOLS[name]
    properties = {}
    for k, t in props.items():
        if t == "array":
            properties[k] = {"type": "array", "items": {"type": "string"}}
        elif t == "object":
            properties[k] = {"type": "object"}
        else:
            properties[k] = {"type": t}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


import sys
sys.path.insert(0, os.path.dirname(__file__))
from gen_tc_09_3_E_data import gen_for_tool, SYSTEM_VARIANTS

TOTAL = 500
SINGLE_TURN_COUNT = 75   # 15%
MULTI_TURN_COUNT = TOTAL - SINGLE_TURN_COUNT

# Plan tool distribution: max 5% of 500 = 25 per tool
TOOL_NAMES = list(TOOLS.keys())  # 55 tools
# Build assignment list: assign samples to tools so no tool exceeds 25 (5%)
# Fairly uniform: 500 / 55 ~= 9.1
assignments = []
per_tool_caps = {t: 0 for t in TOOL_NAMES}
# Round-robin shuffle approach: shuffle a pool of (tool repeated) up to cap
pool = []
target_per_tool = TOTAL // len(TOOL_NAMES)  # 9
extra = TOTAL - target_per_tool * len(TOOL_NAMES)  # 500 - 495 = 5
counts = {t: target_per_tool for t in TOOL_NAMES}
# distribute extras
for t in random.sample(TOOL_NAMES, extra):
    counts[t] += 1
for t, c in counts.items():
    pool.extend([t] * c)
random.shuffle(pool)
assert len(pool) == TOTAL

# Decide single vs multi for each slot
indices = list(range(TOTAL))
random.shuffle(indices)
single_turn_idx = set(indices[:SINGLE_TURN_COUNT])

# Suffix assignment for multi-turn: ~14 uses each across 425 → ceil
suffix_pool_extended = []
per = MULTI_TURN_COUNT // len(SUFFIXES)  # 425/30 = 14
rem = MULTI_TURN_COUNT - per * len(SUFFIXES)  # 425 - 420 = 5
for s in SUFFIXES:
    suffix_pool_extended.extend([s] * per)
for s in random.sample(SUFFIXES, rem):
    suffix_pool_extended.append(s)
random.shuffle(suffix_pool_extended)
assert len(suffix_pool_extended) == MULTI_TURN_COUNT

BLACKLIST = [
    "I've gathered all the information",
    "I've completed the task",
    "Here's what I found:",
    "Based on the results,",
    "The results show that",
]


def make_sample(tool_name, is_single, suffix=None):
    prompt, args, result_json, frag = gen_for_tool(tool_name)
    sys_msg = random.choice(SYSTEM_VARIANTS)
    tool_call = {
        "type": "function",
        "function": {"name": tool_name, "arguments": args},
    }
    if is_single:
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        ]
    else:
        # final assistant message: starts with suffix phrase
        final = f"{suffix} {frag}."
        # safety: ensure no blacklisted opener
        for b in BLACKLIST:
            assert not final.startswith(b)
        msgs = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "name": tool_name, "content": result_json},
            {"role": "assistant", "content": final},
        ]
    return {
        "messages": msgs,
        "tools": [tool_spec(tool_name)],
        "domain": "tool-calling",
    }


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    samples = []
    suffix_iter = iter(suffix_pool_extended)
    for i, tool in enumerate(pool):
        is_single = i in single_turn_idx
        suf = None if is_single else next(suffix_iter)
        samples.append(make_sample(tool, is_single, suf))
    random.shuffle(samples)
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # diagnostics
    from collections import Counter
    tcounts = Counter()
    single = 0
    suff_counts = Counter()
    for s in samples:
        msgs = s["messages"]
        tc = msgs[2]["tool_calls"][0]["function"]["name"]
        tcounts[tc] += 1
        if len(msgs) == 3:
            single += 1
        else:
            final = msgs[-1]["content"]
            for sx in SUFFIXES:
                if final.startswith(sx):
                    suff_counts[sx] += 1
                    break
    print(f"total: {len(samples)}")
    print(f"distinct tools: {len(tcounts)}")
    print(f"max tool count: {max(tcounts.values())} ({max(tcounts.values())/len(samples)*100:.2f}%)")
    print(f"single-turn: {single}")
    print(f"suffix coverage: {len(suff_counts)}/{len(SUFFIXES)}")
    print(f"min/max suffix uses: {min(suff_counts.values())}/{max(suff_counts.values())}")


if __name__ == "__main__":
    main()
