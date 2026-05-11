"""TC-H batch 07: 500 tool-calling samples for cloud/devops/iot/media/ecommerce.

Domain: AWS/GCP/Azure, k8s, Docker, Terraform, CI/CD, monitoring, IoT,
media (Spotify/YouTube-like), e-commerce (Shopify/Stripe-like).

Output: /Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-11-H.jsonl
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path

SEED = "1009311H"
OUT = Path("/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-11-H.jsonl")
TOTAL = 500
SINGLE_TURN_TARGET = 75  # ~15%
MULTI_TURN_TARGET = TOTAL - SINGLE_TURN_TARGET

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a helpful assistant. Use tools when they would help answer the user.",
    "You are a helpful assistant. Call tools whenever they let you give a precise answer.",
    "You are a helpful assistant. Tools are available; use them rather than fabricating data.",
]

SUFFIX_POOL = [
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

BLACKLISTED = (
    "I've gathered all the information",
    "I've completed the task",
    "Here's what I found:",
    "Based on the results,",
    "The results show that",
)


def rand_id(rng, prefix, n=8):
    chars = "0123456789abcdef"
    return prefix + "".join(rng.choice(chars) for _ in range(n))


def rand_int(rng, lo, hi):
    return rng.randint(lo, hi)


# ----- helper generators for realistic resource ids -----
REGIONS = ["us-east-1", "us-east-2", "us-west-1", "us-west-2", "eu-west-1", "eu-central-1", "ap-southeast-1", "ap-northeast-1"]
GCP_REGIONS = ["us-central1", "us-east1", "europe-west1", "asia-east1"]
AZURE_REGIONS = ["eastus", "westus2", "northeurope", "southeastasia"]
CLUSTERS = ["prod-east", "prod-west", "staging", "dev-eu", "qa-us", "edge-1", "edge-2", "analytics"]
NAMESPACES = ["default", "platform", "billing", "auth", "search", "ingest", "frontend", "checkout", "media", "iot-core"]
APPS = ["api-prod", "api-staging", "worker", "scheduler", "ingest-svc", "media-encoder", "checkout-api", "search-svc", "auth-svc", "notification-svc", "stream-relay", "image-resizer", "telemetry-agg"]
BRANCHES = ["main", "develop", "release/2.4", "feature/auth-refactor", "hotfix/checkout-bug", "feature/iot-bridge"]
INSTANCE_TYPES = ["t3.micro", "t3.medium", "t3.large", "m5.large", "m5.xlarge", "c5.large", "r5.xlarge"]
SONG_TITLES = ["Midnight Drive", "Cobalt Sky", "Paper Lanterns", "Static Bloom", "Glass Garden", "Neon Tide", "Slow Burn", "Echo Chamber", "Velvet Hour", "Ember and Ash"]
ARTISTS = ["Lina Vega", "The Owls Below", "Marlowe & Sons", "Kestrel", "Yusra Khan", "Cold Atlas", "Hana Iwata", "Junot Reyes"]
PRODUCTS = ["organic-cotton-tee", "ceramic-mug-12oz", "wool-beanie", "leather-wallet", "bamboo-toothbrush", "linen-apron", "stainless-bottle", "hemp-tote"]


def aws_arn(rng, service, region, acct, resource):
    return f"arn:aws:{service}:{region}:{acct}:{resource}"


# ============================================================
# Tool definitions: 60+ tools across the domain
# ============================================================

def make_tools():
    """Return mapping: tool_name -> (description, args_schema, builder).

    builder(rng) -> (user_text, args_dict, tool_result_str, final_summary_fragment)
    The final_summary_fragment is a short follow-up sentence describing the result.
    """

    T = {}

    # ---------- AWS EC2 ----------
    def t_ec2_list_instances(rng):
        region = rng.choice(REGIONS)
        state = rng.choice(["running", "stopped", "pending"])
        user = rng.choice([
            f"List all EC2 instances in {region} that are {state}.",
            f"Show me {state} EC2 instances in the {region} region.",
            f"Can you pull up EC2 instances in {region} with state={state}?",
        ])
        args = {"region": region, "state": state}
        n = rng.randint(2, 12)
        result = json.dumps({"instances": [f"i-0{rng.randint(10**14, 10**15-1):x}" for _ in range(min(n,3))], "count": n})
        summary = f"Found {n} {state} instances in {region}."
        return user, args, result, summary
    T["ec2_list_instances"] = ("List EC2 instances by region and state.", t_ec2_list_instances)

    def t_ec2_start_instance(rng):
        iid = f"i-0{rng.randint(10**14, 10**15-1):x}"
        region = rng.choice(REGIONS)
        user = rng.choice([
            f"Start EC2 instance {iid} in {region}.",
            f"Spin up {iid} ({region}) please.",
            f"Bring {iid} back online in {region}.",
        ])
        args = {"instance_id": iid, "region": region}
        result = json.dumps({"instance_id": iid, "previous_state": "stopped", "current_state": "pending"})
        summary = f"{iid} is transitioning from stopped to pending."
        return user, args, result, summary
    T["ec2_start_instance"] = ("Start a stopped EC2 instance.", t_ec2_start_instance)

    def t_ec2_stop_instance(rng):
        iid = f"i-0{rng.randint(10**14, 10**15-1):x}"
        region = rng.choice(REGIONS)
        user = rng.choice([
            f"Stop {iid} in {region} for the night.",
            f"Shut down EC2 {iid} ({region}).",
            f"Take {iid} offline in {region} please.",
        ])
        args = {"instance_id": iid, "region": region}
        result = json.dumps({"instance_id": iid, "current_state": "stopping"})
        summary = f"{iid} is now stopping."
        return user, args, result, summary
    T["ec2_stop_instance"] = ("Stop a running EC2 instance.", t_ec2_stop_instance)

    def t_ec2_terminate_instance(rng):
        iid = f"i-0{rng.randint(10**14, 10**15-1):x}"
        region = rng.choice(REGIONS)
        user = rng.choice([
            f"Terminate {iid} in {region} — we're done with it.",
            f"Kill EC2 {iid} ({region}).",
            f"Decommission instance {iid} in {region}.",
        ])
        args = {"instance_id": iid, "region": region}
        result = json.dumps({"instance_id": iid, "current_state": "shutting-down"})
        summary = f"Termination underway for {iid}."
        return user, args, result, summary
    T["ec2_terminate_instance"] = ("Terminate an EC2 instance permanently.", t_ec2_terminate_instance)

    def t_ec2_describe_instance(rng):
        iid = f"i-0{rng.randint(10**14, 10**15-1):x}"
        user = rng.choice([
            f"What instance type is {iid}?",
            f"Describe {iid} for me.",
            f"Pull metadata on EC2 {iid}.",
        ])
        args = {"instance_id": iid}
        itype = rng.choice(INSTANCE_TYPES)
        result = json.dumps({"id": iid, "type": itype, "az": rng.choice(REGIONS) + rng.choice(["a","b","c"])})
        summary = f"{iid} is a {itype}."
        return user, args, result, summary
    T["ec2_describe_instance"] = ("Describe an EC2 instance.", t_ec2_describe_instance)

    # ---------- AWS S3 ----------
    def t_s3_list_objects(rng):
        bucket = rng.choice(["lyra-prod-logs", "user-uploads-eu", "ml-artifacts-2026", "static-assets-cdn", "billing-exports"])
        prefix = rng.choice(["logs/2026-05/", "raw/", "exports/q1/", "thumbnails/", "models/v3/"])
        user = rng.choice([
            f"List objects under {prefix} in s3://{bucket}.",
            f"What's in s3://{bucket}/{prefix}?",
            f"Show contents of bucket {bucket} with prefix {prefix}.",
        ])
        args = {"bucket": bucket, "prefix": prefix}
        n = rng.randint(3, 240)
        result = json.dumps({"bucket": bucket, "prefix": prefix, "count": n})
        summary = f"{n} objects under {prefix} in {bucket}."
        return user, args, result, summary
    T["s3_list_objects"] = ("List objects in an S3 bucket prefix.", t_s3_list_objects)

    def t_s3_get_object(rng):
        bucket = rng.choice(["lyra-prod-logs", "config-store", "user-uploads-eu"])
        key = rng.choice(["config/app.yaml", "logs/2026-05-08.json.gz", "manifests/release-2.4.json", "users/profile.json"])
        user = rng.choice([
            f"Fetch s3://{bucket}/{key}.",
            f"Download object {key} from {bucket}.",
            f"Get me the contents of {bucket}/{key}.",
        ])
        args = {"bucket": bucket, "key": key}
        sz = rng.randint(120, 90000)
        result = json.dumps({"bucket": bucket, "key": key, "size_bytes": sz, "etag": rand_id(rng, "", 16)})
        summary = f"Pulled {key} ({sz} bytes)."
        return user, args, result, summary
    T["s3_get_object"] = ("Retrieve an object from S3.", t_s3_get_object)

    def t_s3_put_object(rng):
        bucket = rng.choice(["lyra-prod-logs", "config-store", "ml-artifacts-2026"])
        key = rng.choice(["config/feature-flags.json", "models/v4/weights.bin", "exports/2026-05-08-summary.csv"])
        user = rng.choice([
            f"Upload the local file as {bucket}/{key}.",
            f"Push the new flags to s3://{bucket}/{key}.",
            f"Save it under s3://{bucket}/{key}.",
        ])
        args = {"bucket": bucket, "key": key}
        result = json.dumps({"bucket": bucket, "key": key, "version_id": rand_id(rng, "v", 12)})
        summary = f"Uploaded to s3://{bucket}/{key}."
        return user, args, result, summary
    T["s3_put_object"] = ("Upload an object to S3.", t_s3_put_object)

    def t_s3_delete_object(rng):
        bucket = rng.choice(["lyra-prod-logs", "user-uploads-eu", "tmp-bucket"])
        key = rng.choice(["tmp/scratch.bin", "logs/2025-12-old.gz", "thumbnails/orphan-7821.jpg"])
        user = rng.choice([
            f"Delete s3://{bucket}/{key}.",
            f"Remove {key} from bucket {bucket}.",
            f"Clean up {bucket}/{key} please.",
        ])
        args = {"bucket": bucket, "key": key}
        result = json.dumps({"bucket": bucket, "key": key, "deleted": True})
        summary = f"Removed {key} from {bucket}."
        return user, args, result, summary
    T["s3_delete_object"] = ("Delete an object from S3.", t_s3_delete_object)

    # ---------- AWS Lambda / CloudWatch / IAM ----------
    def t_lambda_invoke(rng):
        fn = rng.choice(["resize-image", "billing-rollup", "auth-token-refresh", "iot-event-router", "checkout-webhook"])
        user = rng.choice([
            f"Invoke the {fn} lambda with a sample payload.",
            f"Trigger lambda {fn} now.",
            f"Run {fn} once please.",
        ])
        args = {"function_name": fn, "payload": {"sample": True}}
        result = json.dumps({"function": fn, "status_code": 200, "duration_ms": rng.randint(40, 980)})
        summary = f"{fn} returned 200."
        return user, args, result, summary
    T["lambda_invoke"] = ("Invoke an AWS Lambda function.", t_lambda_invoke)

    def t_lambda_get_logs(rng):
        fn = rng.choice(["resize-image", "billing-rollup", "iot-event-router"])
        user = rng.choice([
            f"Pull recent logs for the {fn} lambda.",
            f"What's in CloudWatch for lambda {fn}?",
            f"Tail logs of {fn} please.",
        ])
        args = {"function_name": fn, "minutes": 15}
        n = rng.randint(20, 400)
        result = json.dumps({"function": fn, "log_lines": n, "errors": rng.randint(0, 5)})
        summary = f"{n} log lines for {fn} in the last 15min."
        return user, args, result, summary
    T["lambda_get_logs"] = ("Fetch recent CloudWatch logs for a Lambda.", t_lambda_get_logs)

    def t_cloudwatch_get_metric(rng):
        metric = rng.choice(["CPUUtilization", "MemoryUtilization", "NetworkIn", "RequestCount", "TargetResponseTime"])
        ns = rng.choice(["AWS/EC2", "AWS/ECS", "AWS/ELB", "AWS/Lambda"])
        user = rng.choice([
            f"What's the {metric} on our prod ALB right now?",
            f"Get current {metric} from {ns}.",
            f"Pull {metric} for the last 5 minutes.",
        ])
        args = {"namespace": ns, "metric": metric, "period_s": 300}
        val = round(rng.uniform(2, 92), 2)
        result = json.dumps({"metric": metric, "value": val, "unit": "Percent" if "Utilization" in metric else "Count"})
        summary = f"{metric} is at {val}."
        return user, args, result, summary
    T["cloudwatch_get_metric"] = ("Read a CloudWatch metric.", t_cloudwatch_get_metric)

    def t_iam_list_users(rng):
        user = rng.choice([
            "List all IAM users in our account.",
            "Who has IAM access right now?",
            "Pull the IAM user roster.",
        ])
        args = {}
        n = rng.randint(8, 120)
        result = json.dumps({"users": n, "with_mfa": rng.randint(5, n)})
        summary = f"{n} IAM users total."
        return user, args, result, summary
    T["iam_list_users"] = ("List IAM users in the AWS account.", t_iam_list_users)

    def t_iam_create_role(rng):
        rolename = rng.choice(["LambdaInvokeRole", "S3ReadOnlyAuditor", "CICDDeployer", "IoTIngestRole"])
        user = rng.choice([
            f"Create an IAM role named {rolename}.",
            f"Provision role {rolename} for me.",
            f"Set up the {rolename} IAM role.",
        ])
        args = {"role_name": rolename, "trust_service": "lambda.amazonaws.com"}
        arn = aws_arn(rng, "iam", "", f"{rng.randint(10**11, 10**12-1)}", f"role/{rolename}")
        result = json.dumps({"role_arn": arn})
        summary = f"Created {rolename}: {arn}."
        return user, args, result, summary
    T["iam_create_role"] = ("Create an IAM role.", t_iam_create_role)

    def t_cloudfront_invalidate(rng):
        dist = rand_id(rng, "E", 13).upper()
        path = rng.choice(["/static/*", "/index.html", "/assets/css/*", "/api/v2/*"])
        user = rng.choice([
            f"Invalidate {path} on CloudFront {dist}.",
            f"Bust the CDN cache for {path} (dist {dist}).",
            f"Push a CloudFront invalidation for {path} on {dist}.",
        ])
        args = {"distribution_id": dist, "paths": [path]}
        invid = rand_id(rng, "I", 12).upper()
        result = json.dumps({"distribution": dist, "invalidation_id": invid, "status": "InProgress"})
        summary = f"Invalidation {invid} queued."
        return user, args, result, summary
    T["cloudfront_invalidate"] = ("Invalidate paths on a CloudFront distribution.", t_cloudfront_invalidate)

    def t_rds_describe_instance(rng):
        db = rng.choice(["billing-prod", "auth-prod", "analytics-warehouse", "iot-tsdb"])
        user = rng.choice([
            f"What's the storage used on RDS {db}?",
            f"Describe RDS instance {db}.",
            f"Get RDS details for {db}.",
        ])
        args = {"db_instance_id": db}
        result = json.dumps({"db": db, "engine": rng.choice(["postgres","mysql","aurora-postgres"]), "storage_gb": rng.randint(50, 4000)})
        summary = f"{db} status retrieved."
        return user, args, result, summary
    T["rds_describe_instance"] = ("Describe an RDS instance.", t_rds_describe_instance)

    def t_sqs_send_message(rng):
        queue = rng.choice(["order-events", "iot-ingest", "media-encode-jobs", "billing-retry"])
        user = rng.choice([
            f"Send a test message to SQS queue {queue}.",
            f"Push an event onto {queue}.",
            f"Enqueue this payload on {queue}.",
        ])
        args = {"queue_name": queue, "body": "{\"test\": true}"}
        mid = rand_id(rng, "msg-", 16)
        result = json.dumps({"queue": queue, "message_id": mid})
        summary = f"Enqueued on {queue} as {mid}."
        return user, args, result, summary
    T["sqs_send_message"] = ("Send a message to an SQS queue.", t_sqs_send_message)

    def t_sns_publish(rng):
        topic = rng.choice(["alerts-prod", "deployment-events", "iot-state-change"])
        user = rng.choice([
            f"Publish to SNS topic {topic} about the rollout.",
            f"Broadcast deployment status on {topic}.",
            f"Send a notice to {topic}.",
        ])
        args = {"topic": topic, "subject": "Deploy", "message": "v2.4 rollout complete"}
        result = json.dumps({"topic": topic, "message_id": rand_id(rng, "", 24)})
        summary = f"Published to {topic}."
        return user, args, result, summary
    T["sns_publish"] = ("Publish a message to an SNS topic.", t_sns_publish)

    # ---------- GCP ----------
    def t_gcp_compute_list(rng):
        proj = rng.choice(["lyra-prod-348201", "media-pipeline-771204", "iot-fleet-902184"])
        zone = rng.choice(GCP_REGIONS) + "-" + rng.choice(["a","b","c"])
        user = rng.choice([
            f"List GCE instances in project {proj} zone {zone}.",
            f"What VMs are running in {proj}/{zone}?",
            f"Pull GCP compute instances for {proj} ({zone}).",
        ])
        args = {"project": proj, "zone": zone}
        n = rng.randint(1, 30)
        result = json.dumps({"project": proj, "zone": zone, "count": n})
        summary = f"{n} VMs in {proj}/{zone}."
        return user, args, result, summary
    T["gcp_compute_list"] = ("List GCP Compute Engine instances.", t_gcp_compute_list)

    def t_gcp_pubsub_publish(rng):
        topic = rng.choice(["projects/lyra/topics/orders", "projects/media-pipeline/topics/encode-jobs", "projects/iot-fleet/topics/sensor-readings"])
        user = rng.choice([
            f"Publish a test event to {topic}.",
            f"Drop a message on Pub/Sub topic {topic}.",
            f"Send a payload to {topic}.",
        ])
        args = {"topic": topic, "data": "{}"}
        result = json.dumps({"topic": topic, "message_id": str(rng.randint(10**16, 10**17-1))})
        summary = f"Published to {topic}."
        return user, args, result, summary
    T["gcp_pubsub_publish"] = ("Publish a message to a Pub/Sub topic.", t_gcp_pubsub_publish)

    def t_gcp_storage_list(rng):
        bucket = rng.choice(["gs://lyra-archive-eu", "gs://ml-features-prod", "gs://media-raw-uploads"])
        prefix = rng.choice(["snapshots/", "audio/", "raw/2026-05/"])
        user = rng.choice([
            f"List GCS objects under {bucket}/{prefix}.",
            f"What's in {bucket}/{prefix}?",
            f"Pull a listing of {bucket}/{prefix}.",
        ])
        args = {"bucket": bucket, "prefix": prefix}
        n = rng.randint(1, 800)
        result = json.dumps({"bucket": bucket, "prefix": prefix, "count": n})
        summary = f"{n} objects under {bucket}/{prefix}."
        return user, args, result, summary
    T["gcp_storage_list"] = ("List objects in a GCS bucket.", t_gcp_storage_list)

    def t_gcp_bigquery_query(rng):
        sql = rng.choice([
            "SELECT COUNT(*) FROM `lyra.events.orders` WHERE date = CURRENT_DATE()",
            "SELECT region, AVG(latency_ms) FROM `lyra.metrics.requests` GROUP BY 1",
            "SELECT customer_id FROM `billing.subscriptions` WHERE status='past_due'",
        ])
        user = rng.choice([
            "Run that BQ query for today's order count.",
            "Query BigQuery for average latency by region.",
            "Pull past-due subscribers from BigQuery.",
        ])
        args = {"query": sql}
        rows = rng.randint(1, 12000)
        result = json.dumps({"job_id": rand_id(rng, "bq-job-", 12), "rows": rows})
        summary = f"Returned {rows} rows."
        return user, args, result, summary
    T["gcp_bigquery_query"] = ("Run a BigQuery SQL query.", t_gcp_bigquery_query)

    # ---------- Azure ----------
    def t_azure_vm_restart(rng):
        rg = rng.choice(["rg-prod-eus", "rg-staging-weu", "rg-iot-sea"])
        vm = rng.choice(["vm-api-01", "vm-worker-03", "vm-edge-iot-12", "vm-mediaenc-2"])
        user = rng.choice([
            f"Restart Azure VM {vm} in {rg}.",
            f"Reboot {rg}/{vm}.",
            f"Cycle the Azure VM {vm} ({rg}).",
        ])
        args = {"resource_group": rg, "vm_name": vm}
        result = json.dumps({"vm": vm, "resource_group": rg, "status": "restarting"})
        summary = f"{vm} is restarting."
        return user, args, result, summary
    T["azure_vm_restart"] = ("Restart an Azure VM.", t_azure_vm_restart)

    def t_azure_blob_list(rng):
        sa = rng.choice(["lyraprodsa", "mediarawweu", "iotcoldstorage"])
        container = rng.choice(["logs", "uploads", "snapshots", "exports"])
        user = rng.choice([
            f"List blobs in {sa}/{container}.",
            f"What's in Azure container {container} on {sa}?",
            f"Pull {sa}/{container} listing.",
        ])
        args = {"storage_account": sa, "container": container}
        n = rng.randint(2, 1500)
        result = json.dumps({"storage": sa, "container": container, "count": n})
        summary = f"{n} blobs in {sa}/{container}."
        return user, args, result, summary
    T["azure_blob_list"] = ("List blobs in an Azure container.", t_azure_blob_list)

    def t_azure_function_invoke(rng):
        fn = rng.choice(["OrderWebhook", "IotTelemetryRouter", "MediaTranscodeKick"])
        user = rng.choice([
            f"Invoke Azure Function {fn}.",
            f"Trigger the {fn} function.",
            f"Run {fn} once.",
        ])
        args = {"function_name": fn}
        result = json.dumps({"function": fn, "status": 200, "exec_id": rand_id(rng, "exec-", 10)})
        summary = f"{fn} returned 200."
        return user, args, result, summary
    T["azure_function_invoke"] = ("Invoke an Azure Function.", t_azure_function_invoke)

    # ---------- Kubernetes ----------
    def t_k8s_get_pods(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        user = rng.choice([
            f"List pods in {cluster}/{ns}.",
            f"What pods are running in namespace {ns} on {cluster}?",
            f"Show me {cluster} pods in {ns}.",
        ])
        args = {"cluster": cluster, "namespace": ns}
        n = rng.randint(1, 40)
        ready = rng.randint(max(0, n-3), n)
        result = json.dumps({"cluster": cluster, "namespace": ns, "pods": n, "ready": ready})
        summary = f"{ready}/{n} pods ready in {ns}."
        return user, args, result, summary
    T["k8s_get_pods"] = ("List pods in a Kubernetes namespace.", t_k8s_get_pods)

    def t_k8s_describe_pod(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        pod = f"{rng.choice(APPS)}-{rand_id(rng,'',5)}-{rand_id(rng,'',5)}"
        user = rng.choice([
            f"Describe pod {pod} in {ns} on {cluster}.",
            f"Why is {pod} unhealthy? ({cluster}/{ns})",
            f"Pull details on pod {pod} ({cluster}/{ns}).",
        ])
        args = {"cluster": cluster, "namespace": ns, "pod": pod}
        result = json.dumps({"pod": pod, "phase": rng.choice(["Running","Pending","CrashLoopBackOff"]), "restarts": rng.randint(0, 12)})
        summary = f"{pod} status retrieved."
        return user, args, result, summary
    T["k8s_describe_pod"] = ("Describe a specific pod.", t_k8s_describe_pod)

    def t_k8s_logs(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        pod = f"{rng.choice(APPS)}-{rand_id(rng,'',5)}-{rand_id(rng,'',5)}"
        user = rng.choice([
            f"Tail logs of {pod} in {ns}.",
            f"What's {pod} logging right now? ({cluster}/{ns})",
            f"Pull last 100 log lines from {pod} ({cluster}/{ns}).",
        ])
        args = {"cluster": cluster, "namespace": ns, "pod": pod, "tail": 100}
        n = rng.randint(20, 100)
        result = json.dumps({"pod": pod, "lines": n, "errors": rng.randint(0, 8)})
        summary = f"Pulled {n} log lines from {pod}."
        return user, args, result, summary
    T["k8s_logs"] = ("Fetch pod logs.", t_k8s_logs)

    def t_k8s_rollout_restart(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        dep = rng.choice(APPS)
        user = rng.choice([
            f"Restart the {dep} deployment in cluster {cluster}.",
            f"Roll {dep} on {cluster}/{ns}.",
            f"Trigger a rollout-restart of {dep} ({cluster}/{ns}).",
        ])
        args = {"cluster": cluster, "namespace": ns, "deployment": dep}
        result = json.dumps({"deployment": dep, "rollout": "triggered", "rev": rng.randint(2, 80)})
        summary = f"Rollout triggered on {dep}."
        return user, args, result, summary
    T["k8s_rollout_restart"] = ("Trigger a rollout restart of a deployment.", t_k8s_rollout_restart)

    def t_k8s_apply_manifest(rng):
        cluster = rng.choice(CLUSTERS)
        path = rng.choice(["k8s/api-prod-deploy.yaml", "k8s/iot-bridge-svc.yaml", "k8s/checkout-hpa.yaml"])
        user = rng.choice([
            f"Apply {path} to {cluster}.",
            f"kubectl apply {path} on {cluster}.",
            f"Push {path} to cluster {cluster}.",
        ])
        args = {"cluster": cluster, "manifest_path": path}
        result = json.dumps({"applied": path, "objects": rng.randint(1, 6), "cluster": cluster})
        summary = f"Applied {path} to {cluster}."
        return user, args, result, summary
    T["k8s_apply_manifest"] = ("Apply a YAML manifest to a cluster.", t_k8s_apply_manifest)

    def t_k8s_scale_deployment(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        dep = rng.choice(APPS)
        n = rng.randint(2, 20)
        user = rng.choice([
            f"Scale {dep} to {n} replicas in {cluster}/{ns}.",
            f"Bump {dep} to {n} pods ({cluster}/{ns}).",
            f"Set {dep} replicas={n} on {cluster}/{ns}.",
        ])
        args = {"cluster": cluster, "namespace": ns, "deployment": dep, "replicas": n}
        result = json.dumps({"deployment": dep, "replicas": n, "status": "scaling"})
        summary = f"Scaling {dep} to {n}."
        return user, args, result, summary
    T["k8s_scale_deployment"] = ("Scale a deployment to N replicas.", t_k8s_scale_deployment)

    def t_k8s_delete_pod(rng):
        cluster = rng.choice(CLUSTERS)
        ns = rng.choice(NAMESPACES)
        pod = f"{rng.choice(APPS)}-{rand_id(rng,'',5)}-{rand_id(rng,'',5)}"
        user = rng.choice([
            f"Delete pod {pod} in {ns} on {cluster}.",
            f"Kill {pod} ({cluster}/{ns}).",
            f"Force-restart {pod} by deleting it ({cluster}/{ns}).",
        ])
        args = {"cluster": cluster, "namespace": ns, "pod": pod}
        result = json.dumps({"pod": pod, "deleted": True})
        summary = f"{pod} deleted; controller will reschedule."
        return user, args, result, summary
    T["k8s_delete_pod"] = ("Delete a pod (controller will recreate).", t_k8s_delete_pod)

    # ---------- Docker ----------
    def t_docker_ps(rng):
        host = rng.choice(["build-runner-1", "edge-node-7", "ci-runner-eu-3"])
        user = rng.choice([
            f"What containers are running on {host}?",
            f"docker ps on {host}.",
            f"List containers on {host}.",
        ])
        args = {"host": host}
        n = rng.randint(1, 12)
        result = json.dumps({"host": host, "running": n})
        summary = f"{n} containers on {host}."
        return user, args, result, summary
    T["docker_ps"] = ("List running Docker containers on a host.", t_docker_ps)

    def t_docker_logs(rng):
        cid = rand_id(rng, "", 12)
        host = rng.choice(["build-runner-1", "edge-node-7"])
        user = rng.choice([
            f"Pull logs for container {cid} on {host}.",
            f"docker logs {cid} on {host}.",
            f"What's container {cid} printing on {host}?",
        ])
        args = {"host": host, "container_id": cid, "tail": 200}
        result = json.dumps({"container": cid, "lines": rng.randint(50, 200)})
        summary = f"Logs for {cid} retrieved."
        return user, args, result, summary
    T["docker_logs"] = ("Get logs for a container.", t_docker_logs)

    def t_docker_exec(rng):
        cid = rand_id(rng, "", 12)
        cmd = rng.choice(["ls /var/log", "ps aux", "cat /etc/hostname", "env | grep AWS"])
        user = rng.choice([
            f"Exec '{cmd}' inside container {cid}.",
            f"Run {cmd!r} in container {cid}.",
            f"docker exec {cid} -- {cmd}.",
        ])
        args = {"container_id": cid, "command": cmd}
        result = json.dumps({"container": cid, "exit_code": 0, "output_lines": rng.randint(1, 30)})
        summary = f"Command exited 0 in {cid}."
        return user, args, result, summary
    T["docker_exec"] = ("Exec a command inside a container.", t_docker_exec)

    def t_docker_image_pull(rng):
        img = rng.choice(["nginx:1.27", "redis:7.2-alpine", "ghcr.io/lyra/api-prod:v2.4.1", "postgres:16"])
        host = rng.choice(["build-runner-1", "edge-node-7"])
        user = rng.choice([
            f"Pull {img} on {host}.",
            f"docker pull {img} ({host}).",
            f"Download image {img} to {host}.",
        ])
        args = {"host": host, "image": img}
        result = json.dumps({"image": img, "host": host, "status": "pulled", "digest": "sha256:" + rand_id(rng, "", 24)})
        summary = f"Pulled {img} on {host}."
        return user, args, result, summary
    T["docker_image_pull"] = ("Pull a Docker image to a host.", t_docker_image_pull)

    # ---------- Terraform ----------
    def t_terraform_plan(rng):
        ws = rng.choice(["prod-network", "iot-fleet", "media-pipeline-staging", "checkout-stack"])
        user = rng.choice([
            f"Run terraform plan on workspace {ws}.",
            f"Plan changes for the {ws} workspace.",
            f"What would terraform change in {ws}?",
        ])
        args = {"workspace": ws}
        result = json.dumps({"workspace": ws, "to_add": rng.randint(0, 8), "to_change": rng.randint(0, 6), "to_destroy": rng.randint(0, 3)})
        summary = f"Plan complete for {ws}."
        return user, args, result, summary
    T["terraform_plan"] = ("Run terraform plan on a workspace.", t_terraform_plan)

    def t_terraform_apply(rng):
        ws = rng.choice(["prod-network", "iot-fleet", "media-pipeline-staging"])
        user = rng.choice([
            f"Apply the pending plan on {ws}.",
            f"terraform apply for {ws}.",
            f"Roll out the {ws} terraform changes.",
        ])
        args = {"workspace": ws, "auto_approve": True}
        result = json.dumps({"workspace": ws, "applied": rng.randint(1, 12), "duration_s": rng.randint(20, 600)})
        summary = f"Apply complete on {ws}."
        return user, args, result, summary
    T["terraform_apply"] = ("Apply terraform changes.", t_terraform_apply)

    def t_terraform_state_list(rng):
        ws = rng.choice(["prod-network", "iot-fleet", "checkout-stack"])
        user = rng.choice([
            f"List resources in {ws} terraform state.",
            f"What's in tf state for {ws}?",
            f"Show the {ws} state inventory.",
        ])
        args = {"workspace": ws}
        n = rng.randint(10, 240)
        result = json.dumps({"workspace": ws, "resources": n})
        summary = f"{n} resources tracked in {ws}."
        return user, args, result, summary
    T["terraform_state_list"] = ("List resources in terraform state.", t_terraform_state_list)

    # ---------- CI/CD ----------
    def t_ci_trigger_build(rng):
        repo = rng.choice(["lyra/api", "lyra/web", "lyra/iot-bridge", "lyra/media-encoder"])
        branch = rng.choice(BRANCHES)
        user = rng.choice([
            f"Trigger a CI build on {repo}@{branch}.",
            f"Kick off CI for {repo} branch {branch}.",
            f"Start a fresh build of {repo} on {branch}.",
        ])
        args = {"repo": repo, "branch": branch}
        rid = rng.randint(40000, 99999)
        result = json.dumps({"repo": repo, "branch": branch, "run_id": rid, "status": "queued"})
        summary = f"Run #{rid} queued."
        return user, args, result, summary
    T["ci_trigger_build"] = ("Trigger a CI build.", t_ci_trigger_build)

    def t_ci_get_latest_run(rng):
        branch = rng.choice(BRANCHES)
        user = rng.choice([
            f"What's the status of the latest CI run on {branch}?",
            f"Last build on {branch}?",
            f"How did the most recent {branch} CI run go?",
        ])
        args = {"branch": branch}
        rid = rng.randint(40000, 99999)
        status = rng.choice(["success", "failure", "in_progress"])
        result = json.dumps({"id": rid, "status": status, "duration_s": rng.randint(60, 1200)})
        summary = f"Run {rid} on {branch}: {status}."
        return user, args, result, summary
    T["ci_get_latest_run"] = ("Get the latest CI run for a branch.", t_ci_get_latest_run)

    def t_ci_cancel_run(rng):
        rid = rng.randint(40000, 99999)
        user = rng.choice([
            f"Cancel CI run {rid}.",
            f"Stop run #{rid} — wrong branch.",
            f"Kill the in-progress build {rid}.",
        ])
        args = {"run_id": rid}
        result = json.dumps({"run_id": rid, "status": "cancelled"})
        summary = f"Run {rid} cancelled."
        return user, args, result, summary
    T["ci_cancel_run"] = ("Cancel a running CI build.", t_ci_cancel_run)

    def t_pr_status_checks(rng):
        repo = rng.choice(["lyra/api", "lyra/web", "lyra/iot-bridge"])
        prn = rng.randint(100, 4000)
        user = rng.choice([
            f"What checks are passing on {repo}#{prn}?",
            f"Status of PR {prn} in {repo}.",
            f"Pull check results for {repo} PR #{prn}.",
        ])
        args = {"repo": repo, "pr_number": prn}
        passed = rng.randint(2, 9)
        failed = rng.randint(0, 2)
        result = json.dumps({"repo": repo, "pr": prn, "passed": passed, "failed": failed})
        summary = f"PR #{prn}: {passed} passed, {failed} failed."
        return user, args, result, summary
    T["pr_status_checks"] = ("Get status checks on a pull request.", t_pr_status_checks)

    def t_artifact_publish(rng):
        pkg = rng.choice(["lyra-api:v2.4.1", "lyra-iot-bridge:v0.9.3", "lyra-web:v3.1.0"])
        registry = rng.choice(["ghcr.io/lyra", "registry.lyra.dev", "docker.io/lyra"])
        user = rng.choice([
            f"Publish {pkg} to {registry}.",
            f"Push the artifact {pkg} to {registry}.",
            f"Release {pkg} on {registry}.",
        ])
        args = {"package": pkg, "registry": registry}
        result = json.dumps({"package": pkg, "registry": registry, "digest": "sha256:" + rand_id(rng, "", 24)})
        summary = f"Published {pkg} to {registry}."
        return user, args, result, summary
    T["artifact_publish"] = ("Publish a build artifact to a registry.", t_artifact_publish)

    def t_helm_release_status(rng):
        rel = rng.choice(["api-prod", "checkout", "iot-bridge", "media-encoder"])
        ns = rng.choice(NAMESPACES)
        user = rng.choice([
            f"What's the helm release status of {rel} in {ns}?",
            f"helm status {rel} -n {ns}.",
            f"Show the helm release {rel} ({ns}).",
        ])
        args = {"release": rel, "namespace": ns}
        result = json.dumps({"release": rel, "namespace": ns, "status": rng.choice(["deployed","pending-upgrade","failed"]), "rev": rng.randint(1, 40)})
        summary = f"Helm release {rel} status retrieved."
        return user, args, result, summary
    T["helm_release_status"] = ("Get a Helm release status.", t_helm_release_status)

    # ---------- Monitoring / Alerts ----------
    def t_monitor_get_alert(rng):
        aid = rand_id(rng, "alt-", 10)
        user = rng.choice([
            f"Pull the current state of alert {aid}.",
            f"Is alert {aid} firing?",
            f"Get details on monitor {aid}.",
        ])
        args = {"alert_id": aid}
        result = json.dumps({"alert_id": aid, "state": rng.choice(["firing","resolved","silenced"]), "severity": rng.choice(["P1","P2","P3"])})
        summary = f"Alert {aid} retrieved."
        return user, args, result, summary
    T["monitor_get_alert"] = ("Get an alert's current state.", t_monitor_get_alert)

    def t_monitor_silence_alert(rng):
        aid = rand_id(rng, "alt-", 10)
        mins = rng.choice([15, 30, 60, 120, 240])
        user = rng.choice([
            f"Silence alert {aid} for {mins} minutes.",
            f"Mute monitor {aid} for {mins}m while we deploy.",
            f"Suppress {aid} for the next {mins} minutes.",
        ])
        args = {"alert_id": aid, "duration_min": mins}
        result = json.dumps({"alert_id": aid, "silenced_for_min": mins, "expires_at": "2026-05-08T18:00:00Z"})
        summary = f"{aid} silenced for {mins}m."
        return user, args, result, summary
    T["monitor_silence_alert"] = ("Silence an alert for a duration.", t_monitor_silence_alert)

    def t_pagerduty_create_incident(rng):
        svc = rng.choice(["api-prod", "checkout", "iot-ingest", "media-pipeline"])
        title = rng.choice(["High error rate on /checkout", "RDS replica lag spike", "IoT MQTT broker unreachable"])
        user = rng.choice([
            f"Open a PagerDuty incident on {svc} for: {title}.",
            f"Page the {svc} team about {title!r}.",
            f"Create incident on {svc}: {title}.",
        ])
        args = {"service": svc, "title": title, "urgency": "high"}
        iid = "PD-" + rand_id(rng, "", 8).upper()
        result = json.dumps({"incident_id": iid, "service": svc, "status": "triggered"})
        summary = f"Incident {iid} opened on {svc}."
        return user, args, result, summary
    T["pagerduty_create_incident"] = ("Create a PagerDuty incident.", t_pagerduty_create_incident)

    def t_pagerduty_resolve_incident(rng):
        iid = "PD-" + rand_id(rng, "", 8).upper()
        user = rng.choice([
            f"Resolve PagerDuty incident {iid} — fix is in.",
            f"Close out {iid}.",
            f"Mark {iid} resolved.",
        ])
        args = {"incident_id": iid}
        result = json.dumps({"incident_id": iid, "status": "resolved"})
        summary = f"{iid} resolved."
        return user, args, result, summary
    T["pagerduty_resolve_incident"] = ("Resolve a PagerDuty incident.", t_pagerduty_resolve_incident)

    def t_datadog_query(rng):
        q = rng.choice([
            "avg:system.cpu.user{env:prod} by {host}",
            "sum:trace.web.request.errors{service:checkout}.as_rate()",
            "avg:kubernetes.memory.usage{cluster:prod-east}",
        ])
        user = rng.choice([
            f"Run a Datadog query for {q}.",
            f"Pull metrics: {q}.",
            f"What does DD show for {q}?",
        ])
        args = {"query": q, "from_min_ago": 30}
        result = json.dumps({"query": q, "series": rng.randint(1, 32), "max_value": round(rng.uniform(0.1, 95.4), 2)})
        summary = f"Datadog series retrieved."
        return user, args, result, summary
    T["datadog_query"] = ("Run a Datadog metric query.", t_datadog_query)

    def t_grafana_get_dashboard(rng):
        uid = rand_id(rng, "", 10)
        user = rng.choice([
            f"Open the Grafana dashboard with uid {uid}.",
            f"Pull dashboard {uid} from Grafana.",
            f"What's on dashboard {uid}?",
        ])
        args = {"uid": uid}
        result = json.dumps({"uid": uid, "title": rng.choice(["Prod Overview","IoT Fleet","Checkout SLO"]), "panels": rng.randint(4, 24)})
        summary = f"Dashboard {uid} retrieved."
        return user, args, result, summary
    T["grafana_get_dashboard"] = ("Get a Grafana dashboard by uid.", t_grafana_get_dashboard)

    # ---------- IoT ----------
    def t_iot_set_thermostat(rng):
        device = "thermo-" + rand_id(rng, "", 6)
        temp = rng.randint(60, 78)
        user = rng.choice([
            f"Set thermostat {device} to {temp}F.",
            f"Bump {device} to {temp} degrees.",
            f"Change {device} target to {temp}F.",
        ])
        args = {"device_id": device, "target_f": temp}
        result = json.dumps({"device": device, "target_f": temp, "current_f": temp + rng.randint(-3, 3)})
        summary = f"{device} set to {temp}F."
        return user, args, result, summary
    T["iot_set_thermostat"] = ("Set a thermostat target temperature.", t_iot_set_thermostat)

    def t_iot_get_sensor_reading(rng):
        device = "sensor-" + rand_id(rng, "", 6)
        kind = rng.choice(["temperature", "humidity", "co2", "soil_moisture", "occupancy"])
        user = rng.choice([
            f"What's {device} reading right now?",
            f"Pull the latest {kind} from sensor {device}.",
            f"Read {device} ({kind}).",
        ])
        args = {"device_id": device, "metric": kind}
        val = round(rng.uniform(10, 90), 2)
        result = json.dumps({"device": device, "metric": kind, "value": val, "ts": "2026-05-08T14:23:11Z"})
        summary = f"{device} {kind}: {val}."
        return user, args, result, summary
    T["iot_get_sensor_reading"] = ("Read a sensor's latest value.", t_iot_get_sensor_reading)

    def t_iot_lock_door(rng):
        device = "lock-" + rand_id(rng, "", 6)
        user = rng.choice([
            f"Lock the door {device}.",
            f"Engage lock {device}.",
            f"Make sure {device} is locked.",
        ])
        args = {"device_id": device}
        result = json.dumps({"device": device, "state": "locked", "battery_pct": rng.randint(20, 100)})
        summary = f"{device} is now locked."
        return user, args, result, summary
    T["iot_lock_door"] = ("Lock a smart door lock.", t_iot_lock_door)

    def t_iot_turn_on_light(rng):
        device = "light-" + rand_id(rng, "", 6)
        bright = rng.randint(20, 100)
        user = rng.choice([
            f"Turn on {device} at {bright}%.",
            f"Light up {device} ({bright}% brightness).",
            f"Switch {device} on, {bright}%.",
        ])
        args = {"device_id": device, "brightness_pct": bright}
        result = json.dumps({"device": device, "state": "on", "brightness_pct": bright})
        summary = f"{device} on at {bright}%."
        return user, args, result, summary
    T["iot_turn_on_light"] = ("Turn on a smart light.", t_iot_turn_on_light)

    def t_iot_turn_off_light(rng):
        device = "light-" + rand_id(rng, "", 6)
        user = rng.choice([
            f"Turn off {device}.",
            f"Switch off {device}.",
            f"Kill the light {device}.",
        ])
        args = {"device_id": device}
        result = json.dumps({"device": device, "state": "off"})
        summary = f"{device} is off."
        return user, args, result, summary
    T["iot_turn_off_light"] = ("Turn off a smart light.", t_iot_turn_off_light)

    def t_iot_arm_security(rng):
        site = rng.choice(["home", "office", "cabin", "warehouse-3"])
        mode = rng.choice(["away", "home", "night"])
        user = rng.choice([
            f"Arm the {site} security system in {mode} mode.",
            f"Set {site} alarm to {mode}.",
            f"Engage {mode}-mode alarm at {site}.",
        ])
        args = {"site": site, "mode": mode}
        result = json.dumps({"site": site, "mode": mode, "armed": True})
        summary = f"{site} armed: {mode}."
        return user, args, result, summary
    T["iot_arm_security"] = ("Arm a home/site security system.", t_iot_arm_security)

    def t_iot_list_devices(rng):
        site = rng.choice(["home", "office", "warehouse-3", "cabin"])
        user = rng.choice([
            f"List all IoT devices at {site}.",
            f"What smart devices are paired at {site}?",
            f"Pull the device roster for {site}.",
        ])
        args = {"site": site}
        n = rng.randint(3, 60)
        result = json.dumps({"site": site, "devices": n, "online": rng.randint(max(0, n-5), n)})
        summary = f"{n} devices at {site}."
        return user, args, result, summary
    T["iot_list_devices"] = ("List IoT devices at a site.", t_iot_list_devices)

    # ---------- Media ----------
    def t_play_song(rng):
        title = rng.choice(SONG_TITLES)
        artist = rng.choice(ARTISTS)
        user = rng.choice([
            f"Play '{title}' by {artist}.",
            f"Put on {title} by {artist}.",
            f"Start playback of {artist}'s {title}.",
        ])
        args = {"title": title, "artist": artist}
        result = json.dumps({"now_playing": f"{title} - {artist}", "duration_s": rng.randint(150, 320)})
        summary = f"Now playing: {title} by {artist}."
        return user, args, result, summary
    T["play_song"] = ("Play a song by title and artist.", t_play_song)

    def t_pause_playback(rng):
        device = rng.choice(["living-room-speaker", "kitchen-hub", "headphones-bt-12", "office-soundbar"])
        user = rng.choice([
            f"Pause playback on {device}.",
            f"Hit pause on {device}.",
            f"Stop the music on {device} for a sec.",
        ])
        args = {"device": device}
        result = json.dumps({"device": device, "state": "paused"})
        summary = f"Paused on {device}."
        return user, args, result, summary
    T["pause_playback"] = ("Pause music playback on a device.", t_pause_playback)

    def t_resume_playback(rng):
        device = rng.choice(["living-room-speaker", "kitchen-hub", "headphones-bt-12"])
        user = rng.choice([
            f"Resume playback on {device}.",
            f"Un-pause {device}.",
            f"Pick the song back up on {device}.",
        ])
        args = {"device": device}
        result = json.dumps({"device": device, "state": "playing"})
        summary = f"Resumed on {device}."
        return user, args, result, summary
    T["resume_playback"] = ("Resume paused music playback.", t_resume_playback)

    def t_add_to_playlist(rng):
        title = rng.choice(SONG_TITLES)
        artist = rng.choice(ARTISTS)
        playlist = rng.choice(["Coding Focus", "Sunday Morning", "Roadtrip 2026", "Bedtime Wind-Down"])
        user = rng.choice([
            f"Add '{title}' by {artist} to my {playlist} playlist.",
            f"Drop {title} ({artist}) into {playlist}.",
            f"Save {artist}'s {title} on {playlist}.",
        ])
        args = {"playlist": playlist, "title": title, "artist": artist}
        result = json.dumps({"playlist": playlist, "added": f"{title} - {artist}", "track_count": rng.randint(20, 220)})
        summary = f"Added to {playlist}."
        return user, args, result, summary
    T["add_to_playlist"] = ("Add a track to a playlist.", t_add_to_playlist)

    def t_youtube_play_video(rng):
        vid = rand_id(rng, "", 11)
        user = rng.choice([
            f"Play YouTube video {vid} on the TV.",
            f"Cast YT video {vid} to the living room.",
            f"Open YouTube {vid} on the screen.",
        ])
        args = {"video_id": vid, "device": "tv-livingroom"}
        result = json.dumps({"video_id": vid, "device": "tv-livingroom", "state": "playing"})
        summary = f"YT {vid} playing on the TV."
        return user, args, result, summary
    T["youtube_play_video"] = ("Cast a YouTube video to a device.", t_youtube_play_video)

    def t_youtube_search(rng):
        q = rng.choice(["lo-fi study beats", "cherry tomato pruning tutorial", "rust async basics", "bouldering crash course"])
        user = rng.choice([
            f"Search YouTube for '{q}'.",
            f"Find me YouTube videos about {q}.",
            f"Look up '{q}' on YouTube.",
        ])
        args = {"query": q, "limit": 5}
        result = json.dumps({"query": q, "results": 5, "top_views": rng.randint(10000, 9000000)})
        summary = f"5 results for '{q}'."
        return user, args, result, summary
    T["youtube_search"] = ("Search YouTube for videos.", t_youtube_search)

    def t_volume_set(rng):
        device = rng.choice(["living-room-speaker", "kitchen-hub", "office-soundbar", "patio-speakers"])
        vol = rng.randint(5, 90)
        user = rng.choice([
            f"Set {device} volume to {vol}.",
            f"Turn {device} to {vol}.",
            f"Volume {vol} on {device}.",
        ])
        args = {"device": device, "volume": vol}
        result = json.dumps({"device": device, "volume": vol})
        summary = f"{device} volume: {vol}."
        return user, args, result, summary
    T["volume_set"] = ("Set the volume on a media device.", t_volume_set)

    # ---------- E-commerce / Stripe / Shopify ----------
    def t_shopify_create_order(rng):
        sku = rng.choice(PRODUCTS)
        qty = rng.randint(1, 8)
        cust = "cust-" + rand_id(rng, "", 8)
        user = rng.choice([
            f"Create a Shopify order for {qty}x {sku} for {cust}.",
            f"Place order: {qty} {sku} ({cust}).",
            f"New order — {qty} of {sku} for customer {cust}.",
        ])
        args = {"sku": sku, "quantity": qty, "customer_id": cust}
        oid = rng.randint(40000, 999999)
        result = json.dumps({"order_id": oid, "sku": sku, "qty": qty, "total_usd": round(qty * rng.uniform(8, 65), 2)})
        summary = f"Order #{oid} created."
        return user, args, result, summary
    T["shopify_create_order"] = ("Create a Shopify order.", t_shopify_create_order)

    def t_shopify_get_inventory(rng):
        sku = rng.choice(PRODUCTS)
        user = rng.choice([
            f"How many {sku} do we have in stock?",
            f"Inventory level for {sku}?",
            f"Pull stock count for SKU {sku}.",
        ])
        args = {"sku": sku}
        n = rng.randint(0, 1500)
        result = json.dumps({"sku": sku, "available": n, "warehouse": rng.choice(["wh-east","wh-west","wh-eu"])})
        summary = f"{n} units of {sku} in stock."
        return user, args, result, summary
    T["shopify_get_inventory"] = ("Get inventory level for a SKU.", t_shopify_get_inventory)

    def t_shopify_update_product(rng):
        sku = rng.choice(PRODUCTS)
        price = round(rng.uniform(5, 90), 2)
        user = rng.choice([
            f"Update the price of {sku} to ${price}.",
            f"Set {sku} price to ${price}.",
            f"Change {sku} retail to ${price}.",
        ])
        args = {"sku": sku, "price_usd": price}
        result = json.dumps({"sku": sku, "price_usd": price, "updated": True})
        summary = f"{sku} now ${price}."
        return user, args, result, summary
    T["shopify_update_product"] = ("Update a Shopify product price.", t_shopify_update_product)

    def t_stripe_create_charge(rng):
        amt = round(rng.uniform(5, 950), 2)
        cust = "cus_" + rand_id(rng, "", 14)
        user = rng.choice([
            f"Charge ${amt} to {cust}.",
            f"Run a Stripe charge of ${amt} on customer {cust}.",
            f"Bill {cust} ${amt}.",
        ])
        args = {"customer_id": cust, "amount_usd": amt}
        chid = "ch_" + rand_id(rng, "", 18)
        result = json.dumps({"charge_id": chid, "amount_usd": amt, "status": "succeeded"})
        summary = f"Charge {chid} succeeded."
        return user, args, result, summary
    T["stripe_create_charge"] = ("Create a Stripe charge.", t_stripe_create_charge)

    def t_stripe_refund(rng):
        chid = "ch_" + rand_id(rng, "", 18)
        amt = round(rng.uniform(5, 200), 2)
        user = rng.choice([
            f"Refund ${amt} from charge {chid}.",
            f"Issue a partial refund of ${amt} on {chid}.",
            f"Push ${amt} back to the customer for charge {chid}.",
        ])
        args = {"charge_id": chid, "amount_usd": amt}
        rid = "re_" + rand_id(rng, "", 18)
        result = json.dumps({"refund_id": rid, "amount_usd": amt, "status": "succeeded"})
        summary = f"Refund {rid} processed."
        return user, args, result, summary
    T["stripe_refund"] = ("Refund a Stripe charge.", t_stripe_refund)

    def t_stripe_subscription_cancel(rng):
        sid = "sub_" + rand_id(rng, "", 14)
        user = rng.choice([
            f"Cancel subscription {sid}.",
            f"End {sid} at period end.",
            f"Kill the subscription {sid}.",
        ])
        args = {"subscription_id": sid, "at_period_end": True}
        result = json.dumps({"subscription": sid, "cancel_at_period_end": True, "current_period_end": "2026-06-01T00:00:00Z"})
        summary = f"{sid} set to cancel at period end."
        return user, args, result, summary
    T["stripe_subscription_cancel"] = ("Cancel a Stripe subscription.", t_stripe_subscription_cancel)

    def t_stripe_get_balance(rng):
        user = rng.choice([
            "What's our Stripe balance?",
            "Pull current Stripe available balance.",
            "How much is sitting in Stripe right now?",
        ])
        args = {}
        avail = round(rng.uniform(1000, 250000), 2)
        result = json.dumps({"available_usd": avail, "pending_usd": round(rng.uniform(100, 8000), 2)})
        summary = f"Stripe available balance: ${avail}."
        return user, args, result, summary
    T["stripe_get_balance"] = ("Get current Stripe balance.", t_stripe_get_balance)

    def t_stripe_create_customer(rng):
        email = rng.choice(["amaya.okafor@globex.com", "j.tomlinson@initech.io", "mira@orbital-coffee.shop"])
        user = rng.choice([
            f"Create a Stripe customer for {email}.",
            f"Add {email} as a Stripe customer.",
            f"Provision Stripe customer record for {email}.",
        ])
        args = {"email": email}
        cust = "cus_" + rand_id(rng, "", 14)
        result = json.dumps({"customer_id": cust, "email": email})
        summary = f"Created customer {cust}."
        return user, args, result, summary
    T["stripe_create_customer"] = ("Create a Stripe customer.", t_stripe_create_customer)

    return T


# ============================================================
# Schema generation for tool definitions (for the "tools" field)
# ============================================================

def schema_for(name):
    """Best-effort generic JSON-schema-ish for tool args we generate."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


# ============================================================
# Main generation
# ============================================================

def main():
    rng = random.Random(SEED)
    tools = make_tools()
    tool_names = list(tools.keys())
    assert len(tool_names) >= 40, f"Need >=40 tools, have {len(tool_names)}"

    OUT.parent.mkdir(parents=True, exist_ok=True)

    # plan single vs multi
    indices = list(range(TOTAL))
    rng.shuffle(indices)
    single_set = set(indices[:SINGLE_TURN_TARGET])

    # Cap each tool at <= 5% of TOTAL = 25
    cap_per_tool = max(1, int(TOTAL * 0.05))  # 25
    tool_counts = Counter()

    # Suffix usage tracking — aim ~14 each (30 phrases * ~14 = 420 multi-turn samples we have 425).
    suffix_counts = Counter()
    suffix_target = MULTI_TURN_TARGET // len(SUFFIX_POOL)  # 14
    suffix_extra = MULTI_TURN_TARGET - suffix_target * len(SUFFIX_POOL)  # 425-420=5

    # Build a suffix queue: each phrase appears suffix_target times, then suffix_extra extras
    suffix_queue = []
    for s in SUFFIX_POOL:
        suffix_queue.extend([s] * suffix_target)
    extras = rng.sample(SUFFIX_POOL, suffix_extra)
    suffix_queue.extend(extras)
    rng.shuffle(suffix_queue)

    samples = []
    suffix_iter = iter(suffix_queue)

    # Tool selection: weighted random with cap enforcement
    def pick_tool():
        candidates = [t for t in tool_names if tool_counts[t] < cap_per_tool]
        if not candidates:
            return rng.choice(tool_names)
        return rng.choice(candidates)

    for i in range(TOTAL):
        tname = pick_tool()
        tool_counts[tname] += 1
        desc, builder = tools[tname]
        user_text, args, result_str, summary = builder(rng)
        sysprompt = rng.choice(SYSTEM_PROMPTS)

        if i in single_set:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": tname, "arguments": args}}
                ]},
            ]
        else:
            suffix = next(suffix_iter)
            # ensure no blacklisted opener
            for bl in BLACKLISTED:
                assert not suffix.startswith(bl)
            final = f"{suffix} {summary}"
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": tname, "arguments": args}}
                ]},
                {"role": "tool", "name": tname, "content": result_str},
                {"role": "assistant", "content": final},
            ]
            suffix_counts[suffix] += 1

        sample = {
            "messages": messages,
            "tools": [{
                "type": "function",
                "function": {
                    "name": tname,
                    "description": desc,
                    "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in args}, "required": list(args.keys())},
                },
            }],
            "domain": "tool-calling",
        }
        samples.append(sample)

    # write
    with OUT.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # diagnostics
    line_count = sum(1 for _ in OUT.open())
    distinct_tools = len({s["messages"][2]["tool_calls"][0]["function"]["name"] for s in samples})
    single_count = sum(1 for s in samples if len(s["messages"]) == 3)
    suffix_used = len(suffix_counts)

    # check no tool > 5%
    max_tool, max_n = tool_counts.most_common(1)[0]
    pct = max_n / TOTAL * 100

    print(f"Wrote {OUT}")
    print(f"line_count={line_count}")
    print(f"distinct_tools={distinct_tools}")
    print(f"single_turn={single_count}  multi_turn={TOTAL - single_count}")
    print(f"max_tool={max_tool} count={max_n} ({pct:.1f}%)  cap={cap_per_tool}")
    print(f"suffix_pool_coverage={suffix_used}/{len(SUFFIX_POOL)}")
    print("suffix_distribution:", sorted(suffix_counts.values()))


if __name__ == "__main__":
    main()
