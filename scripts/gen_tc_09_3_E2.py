"""TC-E2 batch-08: 500 tool-calling samples for databases/analytics wave 2.

Wave 2 themes (fresh from wave 1): vector DBs, graph DBs, time-series,
search engines, CDC/streaming connectors, feature stores, data catalogs,
data quality, MDM/dedup, advanced migrations, query plans, partitions,
materialized views, Iceberg/Delta lake operations.
"""
import json
import os
import random
import hashlib

SEED = "1009308E"
random.seed(int(hashlib.md5(SEED.encode()).hexdigest(), 16) % (2**32))

OUT = "/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-08-E.jsonl"

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

# 62 distinct tools — none overlap wave-1 batch-07-E
# Each entry: (description, properties_dict, required_list)
TOOLS = {
    # Vector DBs
    "pinecone_upsert": ("Upsert vectors into a Pinecone index.", {"index": "string", "namespace": "string", "vectors": "array", "count": "integer"}, ["index", "vectors"]),
    "pinecone_query": ("Query a Pinecone index for nearest neighbors.", {"index": "string", "namespace": "string", "top_k": "integer", "vector_dim": "integer"}, ["index", "top_k"]),
    "pinecone_delete": ("Delete vectors from a Pinecone namespace.", {"index": "string", "namespace": "string", "delete_all": "boolean"}, ["index"]),
    "weaviate_search": ("Run a hybrid/vector search on a Weaviate class.", {"class_name": "string", "near_text": "string", "limit": "integer"}, ["class_name"]),
    "weaviate_create_class": ("Create a Weaviate class with a vectorizer.", {"class_name": "string", "vectorizer": "string", "properties": "array"}, ["class_name"]),
    "qdrant_upsert_points": ("Upsert points into a Qdrant collection.", {"collection": "string", "points": "array", "count": "integer"}, ["collection"]),
    "qdrant_search": ("Run a vector similarity search in Qdrant.", {"collection": "string", "limit": "integer", "vector_dim": "integer"}, ["collection", "limit"]),
    "qdrant_create_collection": ("Create a Qdrant collection.", {"collection": "string", "vector_size": "integer", "distance": "string"}, ["collection", "vector_size"]),
    "pgvector_similarity": ("Run a pgvector similarity query.", {"table": "string", "limit": "integer", "operator": "string"}, ["table", "limit"]),
    "pgvector_create_index": ("Create a pgvector ANN index (IVFFlat or HNSW).", {"table": "string", "column": "string", "index_type": "string", "lists": "integer"}, ["table", "column", "index_type"]),
    # Graph DBs
    "neo4j_run_cypher": ("Execute a Cypher query against Neo4j.", {"cypher": "string", "params": "object"}, ["cypher"]),
    "neo4j_create_node": ("Create a Neo4j node with a label.", {"label": "string", "properties": "object"}, ["label", "properties"]),
    "neo4j_create_relationship": ("Create a relationship between two Neo4j nodes.", {"from_label": "string", "to_label": "string", "rel_type": "string", "from_id": "integer", "to_id": "integer"}, ["from_label", "to_label", "rel_type"]),
    "neptune_gremlin": ("Run a Gremlin traversal on AWS Neptune.", {"query": "string"}, ["query"]),
    "graph_shortest_path": ("Compute the shortest path between two nodes.", {"graph": "string", "source_id": "integer", "target_id": "integer", "max_depth": "integer"}, ["graph", "source_id", "target_id"]),
    # Time-series
    "influx_write_points": ("Write points into an InfluxDB bucket.", {"bucket": "string", "measurement": "string", "point_count": "integer"}, ["bucket", "measurement"]),
    "influx_flux_query": ("Run a Flux query against InfluxDB.", {"flux": "string"}, ["flux"]),
    "timescale_create_hypertable": ("Convert a TimescaleDB table to a hypertable.", {"table": "string", "time_column": "string", "chunk_interval": "string"}, ["table", "time_column"]),
    "timescale_compress_chunks": ("Compress old chunks on a TimescaleDB hypertable.", {"hypertable": "string", "older_than": "string"}, ["hypertable", "older_than"]),
    "prometheus_query": ("Run a Prometheus instant query.", {"query": "string"}, ["query"]),
    "prometheus_range_query": ("Run a Prometheus range query.", {"query": "string", "start": "string", "end": "string", "step": "string"}, ["query", "start", "end", "step"]),
    # Search engines
    "elastic_create_index": ("Create an Elasticsearch index.", {"index": "string", "shards": "integer", "replicas": "integer"}, ["index"]),
    "elastic_bulk_index": ("Bulk-index documents into Elasticsearch.", {"index": "string", "doc_count": "integer"}, ["index", "doc_count"]),
    "elastic_reindex": ("Reindex from one Elasticsearch index to another.", {"source": "string", "dest": "string"}, ["source", "dest"]),
    "opensearch_query": ("Query an OpenSearch index.", {"index": "string", "query": "object", "size": "integer", "sort": "array"}, ["index", "query"]),
    "meilisearch_search": ("Search a Meilisearch index.", {"index": "string", "q": "string", "limit": "integer"}, ["index", "q"]),
    "typesense_search": ("Search a Typesense collection.", {"collection": "string", "q": "string", "query_by": "string", "per_page": "integer"}, ["collection", "q", "query_by"]),
    # CDC / streaming
    "debezium_create_connector": ("Create a Debezium source connector.", {"name": "string", "config": "object"}, ["name", "config"]),
    "debezium_pause_connector": ("Pause a Debezium/Kafka Connect connector.", {"connector": "string"}, ["connector"]),
    "kafka_connect_status": ("Check Kafka Connect connector status.", {"connector": "string"}, ["connector"]),
    "kafka_topic_create": ("Create a Kafka topic.", {"topic": "string", "partitions": "integer", "replication_factor": "integer"}, ["topic", "partitions"]),
    "kafka_topic_describe": ("Describe a Kafka topic.", {"topic": "string"}, ["topic"]),
    # Feature stores
    "feast_get_online_features": ("Fetch online features from Feast.", {"entity": "string", "entity_id": "integer", "features": "array"}, ["entity", "entity_id", "features"]),
    "feast_materialize": ("Materialize a Feast feature view to the online store.", {"feature_view": "string", "start": "string", "end": "string"}, ["feature_view", "start", "end"]),
    "feast_apply": ("Apply Feast feature definitions from a repo.", {"repo_path": "string"}, ["repo_path"]),
    "tecton_get_features": ("Get features from a Tecton feature service.", {"feature_service": "string", "join_keys": "object"}, ["feature_service", "join_keys"]),
    # Data catalog
    "datahub_lookup_dataset": ("Look up dataset metadata in DataHub.", {"urn": "string"}, ["urn"]),
    "datahub_set_owner": ("Set ownership on a DataHub dataset.", {"urn": "string", "owner": "string"}, ["urn", "owner"]),
    "amundsen_search_table": ("Search Amundsen for tables.", {"query": "string", "page_size": "integer"}, ["query"]),
    # Data quality
    "ge_run_checkpoint": ("Run a Great Expectations checkpoint.", {"checkpoint": "string"}, ["checkpoint"]),
    "ge_validate_expectation": ("Run a single GE expectation against a column.", {"expectation": "string", "column": "string", "asset": "string"}, ["expectation", "column", "asset"]),
    "dbt_test_freshness": ("Run dbt source freshness checks.", {"source": "string"}, ["source"]),
    "soda_scan": ("Run a Soda data quality scan.", {"data_source": "string", "scan_definition": "string"}, ["data_source", "scan_definition"]),
    # MDM / dedup
    "mdm_match_records": ("Run an MDM match pass on golden records.", {"entity_type": "string", "threshold": "number", "algorithm": "string"}, ["entity_type"]),
    "mdm_merge_records": ("Merge two records as duplicates.", {"survivor_id": "string", "merged_id": "string", "rule": "string"}, ["survivor_id", "merged_id"]),
    "dedup_table": ("Deduplicate rows in a table by key columns.", {"table": "string", "key_columns": "array", "keep": "string"}, ["table", "key_columns"]),
    # Schema migrations
    "alembic_upgrade": ("Apply Alembic migrations to a target revision.", {"revision": "string"}, ["revision"]),
    "alembic_downgrade": ("Roll back Alembic migrations.", {"revision": "string"}, ["revision"]),
    "flyway_info": ("Show Flyway migration status.", {"schema": "string"}, ["schema"]),
    "schema_compare": ("Compare schemas between two environments.", {"left": "string", "right": "string"}, ["left", "right"]),
    # Query plans / partitions / MV
    "explain_analyze": ("Run EXPLAIN ANALYZE on a SQL statement.", {"sql": "string", "format": "string"}, ["sql"]),
    "list_partitions": ("List partitions of a partitioned table.", {"table": "string"}, ["table"]),
    "drop_partition": ("Drop a partition from a table.", {"table": "string", "partition": "string"}, ["table", "partition"]),
    "create_partition": ("Create a partition on a partitioned table.", {"table": "string", "partition_name": "string", "range_start": "string", "range_end": "string"}, ["table", "partition_name"]),
    "refresh_materialized_view": ("Refresh a materialized view.", {"view": "string", "concurrently": "boolean"}, ["view"]),
    "create_materialized_view": ("Create a materialized view.", {"name": "string", "sql": "string", "with_data": "boolean"}, ["name", "sql"]),
    # Iceberg / Delta
    "iceberg_create_table": ("Create an Apache Iceberg table.", {"table": "string", "schema": "array", "partition_spec": "string"}, ["table", "schema"]),
    "iceberg_snapshot_history": ("Show snapshot history for an Iceberg table.", {"table": "string"}, ["table"]),
    "iceberg_expire_snapshots": ("Expire old Iceberg snapshots.", {"table": "string", "older_than": "string"}, ["table", "older_than"]),
    "delta_optimize": ("Run OPTIMIZE on a Delta Lake table.", {"path": "string", "zorder_by": "array"}, ["path"]),
    "delta_vacuum": ("VACUUM a Delta Lake table.", {"path": "string", "retention_hours": "integer"}, ["path"]),
    "delta_time_travel": ("Read a Delta table as of a previous version.", {"path": "string", "version": "integer"}, ["path", "version"]),
}


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
from gen_tc_09_3_E2_data import gen_for_tool, SYSTEM_VARIANTS

TOTAL = 500
SINGLE_TURN_COUNT = 75
MULTI_TURN_COUNT = TOTAL - SINGLE_TURN_COUNT  # 425

TOOL_NAMES = list(TOOLS.keys())  # 62
assert len(TOOL_NAMES) >= 40, f"need >=40 tools, got {len(TOOL_NAMES)}"

# Distribute 500 across 62 tools, max 25 per tool (5%).
target_per_tool = TOTAL // len(TOOL_NAMES)  # 8
extra = TOTAL - target_per_tool * len(TOOL_NAMES)  # 4
counts = {t: target_per_tool for t in TOOL_NAMES}
for t in random.sample(TOOL_NAMES, extra):
    counts[t] += 1
assert max(counts.values()) <= 25, "max tool count would exceed 5%"
pool = []
for t, c in counts.items():
    pool.extend([t] * c)
random.shuffle(pool)
assert len(pool) == TOTAL

# Decide single vs multi
indices = list(range(TOTAL))
random.shuffle(indices)
single_turn_idx = set(indices[:SINGLE_TURN_COUNT])

# Suffix assignment
suffix_pool_extended = []
per = MULTI_TURN_COUNT // len(SUFFIXES)  # 14
rem = MULTI_TURN_COUNT - per * len(SUFFIXES)  # 5
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


def make_sample(tool_name, is_single, suffix=None, salt=0):
    # Try a few times if generated user prompt collides at outer level
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
        final = f"{suffix} {frag}."
        for b in BLACKLIST:
            assert not final.startswith(b), f"blacklist hit: {b}"
            assert b not in final, f"blacklist substring: {b}"
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
    seen_user = set()
    for i, tool in enumerate(pool):
        is_single = i in single_turn_idx
        suf = None if is_single else next(suffix_iter)
        # regenerate up to 12 times to dedupe user messages within batch
        for attempt in range(12):
            sample = make_sample(tool, is_single, suf)
            user_msg = sample["messages"][1]["content"]
            if user_msg not in seen_user:
                seen_user.add(user_msg)
                samples.append(sample)
                break
        else:
            # fallback: append a small unique tag
            sample = make_sample(tool, is_single, suf)
            sample["messages"][1]["content"] += f" (req #{i})"
            seen_user.add(sample["messages"][1]["content"])
            samples.append(sample)

    random.shuffle(samples)
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Diagnostics
    from collections import Counter
    tcounts = Counter()
    single = 0
    suff_counts = Counter()
    user_msgs = []
    blacklist_hits = 0
    for s in samples:
        msgs = s["messages"]
        tc = msgs[2]["tool_calls"][0]["function"]["name"]
        tcounts[tc] += 1
        user_msgs.append(msgs[1]["content"])
        if len(msgs) == 3:
            single += 1
        else:
            final = msgs[-1]["content"]
            for b in BLACKLIST:
                if b in final:
                    blacklist_hits += 1
            for sx in SUFFIXES:
                if final.startswith(sx):
                    suff_counts[sx] += 1
                    break
    multi = len(samples) - single
    dup_users = len(user_msgs) - len(set(user_msgs))
    print(f"total: {len(samples)}")
    print(f"distinct tools: {len(tcounts)}")
    print(f"max tool count: {max(tcounts.values())} ({max(tcounts.values())/len(samples)*100:.2f}%)")
    print(f"min tool count: {min(tcounts.values())}")
    print(f"single-turn: {single}  multi-turn: {multi}")
    print(f"suffix coverage: {len(suff_counts)}/{len(SUFFIXES)}")
    if suff_counts:
        print(f"min/max suffix uses: {min(suff_counts.values())}/{max(suff_counts.values())}")
    print(f"blacklist hits: {blacklist_hits}")
    print(f"duplicate user msgs: {dup_users}")


if __name__ == "__main__":
    main()
