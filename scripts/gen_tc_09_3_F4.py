"""TC-F: Tool-calling samples for task management, project mgmt, ticketing, HR, CRM, expenses.

Generates exactly 500 ShareGPT-format samples with mode-collapse safeguards.
Run: python gen_tc_09_3_F.py
Output: /Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-07-F.jsonl
"""
import json
import os
import random
from collections import Counter

SEED = "1009310F"
OUT_PATH = "/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-10-F.jsonl"
N_TOTAL = 500
SINGLE_TURN_FRAC = 0.15  # ~15%

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

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a helpful assistant. Use tools when they would help answer the user.",
    "You are a helpful assistant that calls functions to complete tasks accurately.",
    "You are an assistant with access to internal tools. Call them when needed.",
]

PROJECT_KEYS = ["PROJ", "ENG", "PLAT", "CORE", "API", "WEB", "MOB", "DATA", "INFRA", "SEC",
                "DEV", "OPS", "QA", "DOC", "GROW", "REV", "MKT", "SUP", "FIN", "HR",
                "ACME", "GLOBEX", "NORTH", "WIDGET", "ZAP", "FORGE", "PIVOT", "ATLAS"]
LIN_PREFIXES = ["LIN", "LNR", "LR", "TKT", "BUG", "FEAT"]

FIRST_NAMES = ["alice", "bob", "carla", "dmitri", "elena", "farouk", "gita", "hiro", "ines",
               "jamal", "kira", "luca", "mei", "nadia", "omar", "priya", "quentin", "rina",
               "sven", "tomas", "uma", "vikram", "wei", "xan", "yara", "zane",
               "amaya", "ben", "chloe", "diego", "emma", "frank", "grace", "henrik",
               "isla", "javier", "kenji", "lina", "marco", "nora", "ola", "paulo"]
LAST_NAMES = ["okafor", "rivera", "patel", "kim", "schmidt", "nguyen", "lopez", "tanaka",
              "ivanov", "andersen", "moreau", "santos", "kowalski", "fernandez", "haddad",
              "abebe", "kaur", "yamamoto", "petersen", "marchetti", "novak"]
COMPANIES = ["acme", "globex", "initech", "umbrella", "stark", "wayne", "hooli",
             "pied-piper", "vandelay", "soylent", "cyberdyne", "tyrell", "weyland"]
SPRINT_NAMES = ["Atlas", "Borealis", "Comet", "Delta", "Eclipse", "Falcon", "Galaxy",
                "Horizon", "Ion", "Jupiter", "Kestrel", "Lumen", "Mercury", "Nebula",
                "Orion", "Pulsar", "Quasar", "Raven", "Saturn", "Titan"]
TASK_TITLES = [
    "Refactor billing module", "Migrate legacy auth flow", "Audit S3 bucket policies",
    "Draft Q3 OKRs", "Update onboarding checklist", "Fix flaky integration tests",
    "Review vendor SOC2 report", "Plan team offsite agenda", "Triage support backlog",
    "Roll out feature flags v2", "Backfill analytics pipeline", "Patch CVE-2025-1184",
    "Write postmortem for incident 4451", "Compile FY26 budget request",
    "Schedule design review", "Cut release notes for 3.7", "Sunset legacy webhook",
    "Polish landing page hero", "Audit IAM permissions", "Tune Postgres slow queries",
    "Refresh API rate limits", "Implement CSV export", "Set up canary deploy",
    "Recalibrate Datadog alerts", "Rotate prod credentials",
]
LEAD_SOURCES = ["webform", "tradeshow", "cold_outbound", "referral", "linkedin",
                "partner", "event", "inbound_demo", "content_dl"]
DEAL_STAGES = ["prospecting", "qualification", "discovery", "proposal",
               "negotiation", "closed_won", "closed_lost"]
EXPENSE_CATS = ["travel", "meals", "lodging", "software", "office_supplies",
                "client_entertainment", "training", "subscription"]
EMP_DEPTS = ["engineering", "sales", "marketing", "people", "finance",
             "support", "design", "data", "legal"]

# 50 distinct tools — comfortably above 40
TOOLS = [
    # task management
    "create_task", "complete_task", "update_task_due", "assign_task", "list_tasks",
    "delete_task", "reopen_task", "add_subtask", "set_task_priority",
    # project mgmt
    "create_project", "archive_project", "add_project_member", "remove_project_member",
    "rename_project", "set_project_owner",
    # ticketing
    "create_issue", "update_issue", "close_issue", "comment_on_issue", "get_issue",
    "link_issues", "move_issue_status", "add_label", "remove_label", "assign_issue",
    # sprints
    "start_sprint", "end_sprint", "velocity_report", "add_to_sprint", "remove_from_sprint",
    # HR
    "request_pto", "cancel_pto", "get_remaining_pto", "submit_timesheet",
    "lookup_employee", "update_org_chart", "approve_timesheet",
    # expenses
    "submit_expense", "approve_expense", "reject_expense",
    # CRM
    "add_lead", "convert_lead_to_opportunity", "update_opportunity_stage",
    "log_call", "log_email_to_crm", "create_deal", "forecast_pipeline",
    # invoicing / OKRs / 1on1
    "send_invoice", "mark_invoice_paid", "set_okr", "update_okr_progress",
    "record_one_on_one_note",
]
assert len(set(TOOLS)) >= 40

# Tool schemas (description + arg shape) and a generator that produces (user, args, result, summary_tail)
def rid(rng, lo=100, hi=99999):
    return rng.randint(lo, hi)

def issue_key(rng):
    return f"{rng.choice(PROJECT_KEYS)}-{rid(rng, 100, 9999)}"

def lin_key(rng):
    return f"{rng.choice(LIN_PREFIXES)}-{rid(rng, 100, 9999)}"

def employee_id(rng):
    return f"EMP-{rid(rng, 1000, 99999)}"

def email(rng):
    return f"{rng.choice(FIRST_NAMES)}.{rng.choice(LAST_NAMES)}@{rng.choice(COMPANIES)}.com"

def name(rng):
    return f"{rng.choice(FIRST_NAMES).capitalize()} {rng.choice(LAST_NAMES).capitalize()}"

def date(rng):
    return f"2026-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"

def money(rng, lo=10, hi=5000):
    return round(rng.uniform(lo, hi), 2)

# ----- per-tool generators -----
def gen_create_task(rng):
    title = rng.choice(TASK_TITLES)
    due = date(rng)
    args = {"title": title, "due": due, "assignee": rng.choice(FIRST_NAMES)}
    user = f"Create a task '{title}' due {due}, assign to {args['assignee']}."
    tid = f"task_{rid(rng)}"
    result = {"id": tid, "title": title, "due": due, "assignee": args["assignee"], "status": "open"}
    tail = f"Task {tid} created and queued."
    return user, args, result, tail

def gen_complete_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Mark {tid} as complete."
    result = {"task_id": tid, "status": "done"}
    return user, args, result, f"{tid} is now marked done."

def gen_update_task_due(rng):
    tid = f"task_{rid(rng)}"
    new_due = date(rng)
    args = {"task_id": tid, "due": new_due}
    user = f"Push {tid}'s due date to {new_due}."
    result = {"task_id": tid, "due": new_due, "updated": True}
    return user, args, result, f"Due date for {tid} now {new_due}."

def gen_assign_task(rng):
    tid = f"task_{rid(rng)}"
    person = rng.choice(FIRST_NAMES)
    args = {"task_id": tid, "assignee": person}
    user = f"Assign {tid} to {person}."
    result = {"task_id": tid, "assignee": person}
    return user, args, result, f"{person} now owns {tid}."

def gen_list_tasks(rng):
    proj = rng.choice(PROJECT_KEYS).lower()
    args = {"project": proj, "status": "open"}
    user = f"List open tasks in the {proj} project."
    result = {"project": proj, "count": rng.randint(2, 18),
              "tasks": [f"task_{rid(rng)}" for _ in range(3)]}
    return user, args, result, f"Found {result['count']} open tasks in {proj}."

def gen_delete_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Delete task {tid}, it's a duplicate."
    result = {"task_id": tid, "deleted": True}
    return user, args, result, f"{tid} removed."

def gen_reopen_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Reopen {tid}, it wasn't actually finished."
    result = {"task_id": tid, "status": "open"}
    return user, args, result, f"{tid} is open again."

def gen_add_subtask(rng):
    tid = f"task_{rid(rng)}"
    sub = rng.choice(["write tests", "update docs", "review with PM", "ship behind flag", "QA pass"])
    args = {"parent_id": tid, "title": sub}
    user = f"Add a subtask '{sub}' under {tid}."
    sid = f"task_{rid(rng)}"
    result = {"id": sid, "parent": tid, "title": sub}
    return user, args, result, f"Subtask {sid} added under {tid}."

def gen_set_task_priority(rng):
    tid = f"task_{rid(rng)}"
    pri = rng.choice(["low", "medium", "high", "urgent"])
    args = {"task_id": tid, "priority": pri}
    user = f"Bump {tid} to {pri} priority."
    result = {"task_id": tid, "priority": pri}
    return user, args, result, f"{tid} now {pri} priority."

def gen_create_project(rng):
    nm = f"{rng.choice(['Atlas', 'Beacon', 'Cipher', 'Delta', 'Echo', 'Forge'])} {rng.choice(['Migration', 'Rollout', 'Audit', 'Refresh'])}"
    args = {"name": nm, "owner": rng.choice(FIRST_NAMES)}
    user = f"Spin up a new project called '{nm}', owner {args['owner']}."
    pid = f"proj_{rid(rng)}"
    result = {"id": pid, "name": nm, "owner": args["owner"]}
    return user, args, result, f"Project {pid} created."

def gen_archive_project(rng):
    pid = f"proj_{rid(rng)}"
    args = {"project_id": pid}
    user = f"Archive {pid}, it's been done for months."
    result = {"project_id": pid, "archived": True}
    return user, args, result, f"{pid} archived."

def gen_add_project_member(rng):
    pid = f"proj_{rid(rng)}"
    person = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "user": person, "role": rng.choice(["member", "admin", "viewer"])}
    user = f"Add {person} to {pid} as {args['role']}."
    result = {"project_id": pid, "user": person, "role": args["role"]}
    return user, args, result, f"{person} added to {pid}."

def gen_remove_project_member(rng):
    pid = f"proj_{rid(rng)}"
    person = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "user": person}
    user = f"Remove {person} from {pid}."
    result = {"project_id": pid, "user": person, "removed": True}
    return user, args, result, f"{person} removed from {pid}."

def gen_rename_project(rng):
    pid = f"proj_{rid(rng)}"
    new = rng.choice(["Phoenix", "Tundra", "Helix", "Mosaic"])
    args = {"project_id": pid, "name": new}
    user = f"Rename {pid} to '{new}'."
    result = {"project_id": pid, "name": new}
    return user, args, result, f"{pid} renamed to {new}."

def gen_set_project_owner(rng):
    pid = f"proj_{rid(rng)}"
    person = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "owner": person}
    user = f"Hand {pid} ownership to {person}."
    result = {"project_id": pid, "owner": person}
    return user, args, result, f"{person} now owns {pid}."

def gen_create_issue(rng):
    proj = rng.choice(PROJECT_KEYS)
    title = rng.choice(["Login fails on Safari", "Webhook retries 502", "Memory leak in worker",
                        "Stale cache on profile page", "CSV export truncates rows", "Search timeout"])
    args = {"project": proj, "title": title, "type": rng.choice(["bug", "task", "story"])}
    user = f"Open a {args['type']} in {proj}: {title}"
    key = f"{proj}-{rid(rng, 1000, 9999)}"
    result = {"key": key, "title": title}
    return user, args, result, f"Filed as {key}."

def gen_update_issue(rng):
    key = issue_key(rng)
    args = {"key": key, "fields": {"priority": rng.choice(["P0", "P1", "P2", "P3"])}}
    user = f"Bump {key} to {args['fields']['priority']}."
    result = {"key": key, "updated": True}
    return user, args, result, f"{key} updated."

def gen_close_issue(rng):
    key = issue_key(rng)
    res = rng.choice(["fixed", "wontfix", "duplicate", "cannot_reproduce"])
    args = {"issue_key": key, "resolution": res}
    user = f"Close {key} as {res.replace('_', '-')}."
    result = {"key": key, "status": "closed", "resolution": res}
    return user, args, result, f"{key} closed ({res})."

def gen_comment_on_issue(rng):
    key = issue_key(rng)
    body = rng.choice(["Repro on staging confirmed.", "Patched in PR #482.",
                       "Waiting on infra team.", "Will retest after deploy."])
    args = {"key": key, "body": body}
    user = f"Drop a note on {key}: {body}"
    result = {"key": key, "comment_id": rid(rng)}
    return user, args, result, f"Comment posted on {key}."

def gen_get_issue(rng):
    key = lin_key(rng)
    args = {"key": key}
    user = f"Who's assigned to {key}?"
    result = {"key": key, "assignee": rng.choice(FIRST_NAMES), "status": rng.choice(["in_progress", "review", "todo"])}
    return user, args, result, f"{result['assignee'].capitalize()} is on {key}, currently {result['status']}."

def gen_link_issues(rng):
    a, b = issue_key(rng), issue_key(rng)
    rel = rng.choice(["blocks", "blocked_by", "duplicates", "relates_to"])
    args = {"from_key": a, "to_key": b, "relation": rel}
    user = f"Link {a} as {rel.replace('_',' ')} {b}."
    result = {"from": a, "to": b, "relation": rel}
    return user, args, result, f"Link recorded: {a} {rel} {b}."

def gen_move_issue_status(rng):
    key = issue_key(rng)
    status = rng.choice(["todo", "in_progress", "review", "done", "blocked"])
    args = {"key": key, "status": status}
    user = f"Move {key} to {status.replace('_',' ')}."
    result = {"key": key, "status": status}
    return user, args, result, f"{key} now in {status}."

def gen_add_label(rng):
    key = issue_key(rng)
    label = rng.choice(["needs-triage", "good-first-issue", "regression", "customer-reported", "tech-debt"])
    args = {"key": key, "label": label}
    user = f"Tag {key} with '{label}'."
    result = {"key": key, "label": label, "added": True}
    return user, args, result, f"Label '{label}' applied to {key}."

def gen_remove_label(rng):
    key = issue_key(rng)
    label = rng.choice(["needs-triage", "blocked", "wip", "draft"])
    args = {"key": key, "label": label}
    user = f"Strip '{label}' off {key}."
    result = {"key": key, "label": label, "removed": True}
    return user, args, result, f"Label '{label}' removed from {key}."

def gen_assign_issue(rng):
    key = issue_key(rng)
    person = rng.choice(FIRST_NAMES)
    args = {"key": key, "assignee": person}
    user = f"Hand {key} over to {person}."
    result = {"key": key, "assignee": person}
    return user, args, result, f"{key} assigned to {person}."

def gen_start_sprint(rng):
    nm = f"Sprint {rng.choice(SPRINT_NAMES)}"
    args = {"name": nm, "start": date(rng), "end": date(rng)}
    user = f"Kick off {nm}."
    result = {"name": nm, "status": "active", "id": f"sprint_{rid(rng)}"}
    return user, args, result, f"{nm} is live."

def gen_end_sprint(rng):
    sid = f"sprint_{rid(rng)}"
    args = {"sprint_id": sid}
    user = f"Wrap up {sid}."
    result = {"sprint_id": sid, "status": "completed", "completed_points": rng.randint(15, 60)}
    return user, args, result, f"{sid} closed at {result['completed_points']} points."

def gen_velocity_report(rng):
    team = rng.choice(["platform", "growth", "core", "data", "infra"])
    args = {"team": team, "last_n_sprints": rng.choice([3, 5, 10])}
    user = f"What's {team} team velocity over the last {args['last_n_sprints']} sprints?"
    result = {"team": team, "avg_velocity": rng.randint(20, 50)}
    return user, args, result, f"{team} averages {result['avg_velocity']} pts/sprint."

def gen_add_to_sprint(rng):
    key = issue_key(rng)
    sid = f"sprint_{rid(rng)}"
    args = {"key": key, "sprint_id": sid}
    user = f"Pull {key} into {sid}."
    result = {"key": key, "sprint_id": sid}
    return user, args, result, f"{key} added to {sid}."

def gen_remove_from_sprint(rng):
    key = issue_key(rng)
    args = {"key": key}
    user = f"Drop {key} from the active sprint."
    result = {"key": key, "removed_from_sprint": True}
    return user, args, result, f"{key} removed from sprint."

def gen_request_pto(rng):
    sd, ed = date(rng), date(rng)
    reason = rng.choice(["vacation", "personal", "wedding", "family", "medical"])
    args = {"start": sd, "end": ed, "reason": reason}
    user = f"Submit PTO from {sd} to {ed} for {reason}."
    result = {"request_id": f"pto_{rid(rng)}", "status": "pending"}
    return user, args, result, f"PTO request {result['request_id']} submitted."

def gen_cancel_pto(rng):
    pid = f"pto_{rid(rng)}"
    args = {"request_id": pid}
    user = f"Cancel my PTO request {pid}."
    result = {"request_id": pid, "status": "cancelled"}
    return user, args, result, f"{pid} cancelled."

def gen_get_remaining_pto(rng):
    eid = employee_id(rng)
    args = {"employee_id": eid}
    user = f"How many PTO days does {eid} have left?"
    result = {"employee_id": eid, "remaining_days": rng.randint(0, 25)}
    return user, args, result, f"{eid} has {result['remaining_days']} days left."

def gen_submit_timesheet(rng):
    week = f"2026-W{rng.randint(1,52):02d}"
    hrs = rng.randint(35, 50)
    args = {"week": week, "hours": hrs}
    user = f"File my timesheet for {week} at {hrs} hours."
    result = {"week": week, "hours": hrs, "status": "submitted"}
    return user, args, result, f"Timesheet for {week} submitted."

def gen_lookup_employee(rng):
    em = email(rng)
    args = {"email": em}
    user = f"Pull employee record for {em}."
    result = {"email": em, "id": employee_id(rng), "department": rng.choice(EMP_DEPTS),
              "manager": rng.choice(FIRST_NAMES)}
    return user, args, result, f"{em} is in {result['department']}, manager {result['manager']}."

def gen_update_org_chart(rng):
    eid = employee_id(rng)
    mgr = rng.choice(FIRST_NAMES)
    args = {"employee_id": eid, "new_manager": mgr}
    user = f"Update {eid}'s manager to {mgr}."
    result = {"employee_id": eid, "manager": mgr}
    return user, args, result, f"{eid} now reports to {mgr}."

def gen_approve_timesheet(rng):
    tsid = f"ts_{rid(rng)}"
    args = {"timesheet_id": tsid}
    user = f"Approve timesheet {tsid}."
    result = {"timesheet_id": tsid, "status": "approved"}
    return user, args, result, f"Timesheet {tsid} approved."

def gen_submit_expense(rng):
    cat = rng.choice(EXPENSE_CATS)
    amt = money(rng)
    args = {"category": cat, "amount": amt, "currency": "USD", "date": date(rng)}
    user = f"File a ${amt} {cat} expense for {args['date']}."
    result = {"expense_id": f"exp_{rid(rng)}", "status": "pending"}
    return user, args, result, f"Expense {result['expense_id']} filed."

def gen_approve_expense(rng):
    eid = f"exp_{rid(rng)}"
    args = {"expense_id": eid}
    user = f"Approve expense {eid}."
    result = {"expense_id": eid, "status": "approved"}
    return user, args, result, f"{eid} approved."

def gen_reject_expense(rng):
    eid = f"exp_{rid(rng)}"
    reason = rng.choice(["missing_receipt", "out_of_policy", "duplicate", "wrong_category"])
    args = {"expense_id": eid, "reason": reason}
    user = f"Reject {eid}: {reason.replace('_',' ')}."
    result = {"expense_id": eid, "status": "rejected", "reason": reason}
    return user, args, result, f"{eid} rejected ({reason})."

def gen_add_lead(rng):
    nm = name(rng)
    src = rng.choice(LEAD_SOURCES)
    co = rng.choice(COMPANIES).capitalize()
    args = {"name": nm, "company": co, "source": src, "email": email(rng)}
    user = f"Add a new lead: {nm} from {co}, source {src}."
    result = {"lead_id": f"lead_{rid(rng)}", "status": "new"}
    return user, args, result, f"Lead {result['lead_id']} created."

def gen_convert_lead_to_opportunity(rng):
    lid = f"lead_{rid(rng)}"
    args = {"lead_id": lid, "amount": money(rng, 1000, 50000)}
    user = f"Convert {lid} to an opportunity at ${args['amount']}."
    result = {"lead_id": lid, "opportunity_id": f"opp_{rid(rng)}", "amount": args["amount"]}
    return user, args, result, f"Opportunity {result['opportunity_id']} created from {lid}."

def gen_update_opportunity_stage(rng):
    oid = f"opp_{rid(rng)}"
    stg = rng.choice(DEAL_STAGES)
    args = {"opportunity_id": oid, "stage": stg}
    user = f"Move {oid} to {stg.replace('_',' ')}."
    result = {"opportunity_id": oid, "stage": stg}
    return user, args, result, f"{oid} is at {stg}."

def gen_log_call(rng):
    contact = name(rng)
    dur = rng.randint(5, 60)
    args = {"contact": contact, "duration_min": dur, "notes": "Discussed pilot scope."}
    user = f"Log a {dur}-minute call with {contact} about pilot scope."
    result = {"activity_id": f"act_{rid(rng)}", "type": "call"}
    return user, args, result, f"Call logged as {result['activity_id']}."

def gen_log_email_to_crm(rng):
    em = email(rng)
    subj = rng.choice(["Follow-up on demo", "Pricing question", "Renewal terms", "Q3 roadmap share"])
    args = {"contact_email": em, "subject": subj, "direction": rng.choice(["inbound", "outbound"])}
    user = f"Log {args['direction']} email to {em}: '{subj}'."
    result = {"activity_id": f"act_{rid(rng)}", "type": "email"}
    return user, args, result, f"Email logged on {em}'s record."

def gen_create_deal(rng):
    co = rng.choice(COMPANIES).capitalize()
    amt = money(rng, 5000, 250000)
    args = {"company": co, "amount": amt, "stage": "prospecting", "close_date": date(rng)}
    user = f"Create a deal for {co} at ${amt}, target close {args['close_date']}."
    result = {"deal_id": f"deal_{rid(rng)}", "amount": amt}
    return user, args, result, f"Deal {result['deal_id']} created for {co}."

def gen_forecast_pipeline(rng):
    q = f"Q{rng.randint(1,4)} 2026"
    args = {"quarter": q, "team": rng.choice(["enterprise", "smb", "midmarket"])}
    user = f"Forecast {args['team']} pipeline for {q}."
    result = {"quarter": q, "weighted": round(money(rng, 100000, 2000000), 2),
              "deals": rng.randint(8, 60)}
    return user, args, result, f"{q} weighted forecast: ${result['weighted']:,.0f} across {result['deals']} deals."

def gen_send_invoice(rng):
    co = rng.choice(COMPANIES).capitalize()
    amt = money(rng, 500, 25000)
    args = {"customer": co, "amount": amt, "due_date": date(rng)}
    user = f"Send an invoice to {co} for ${amt}, due {args['due_date']}."
    result = {"invoice_id": f"inv_{rid(rng)}", "status": "sent"}
    return user, args, result, f"Invoice {result['invoice_id']} sent to {co}."

def gen_mark_invoice_paid(rng):
    iid = f"inv_{rid(rng)}"
    args = {"invoice_id": iid, "paid_date": date(rng)}
    user = f"Mark {iid} as paid as of {args['paid_date']}."
    result = {"invoice_id": iid, "status": "paid"}
    return user, args, result, f"{iid} marked paid."

def gen_set_okr(rng):
    obj = rng.choice(["Improve activation by 15%", "Cut p99 latency below 200ms",
                      "Land 5 enterprise pilots", "Reduce support backlog by 40%"])
    args = {"team": rng.choice(["growth", "platform", "sales", "support"]),
            "objective": obj, "quarter": f"Q{rng.randint(1,4)} 2026"}
    user = f"Set OKR for {args['team']}: {obj} ({args['quarter']})."
    result = {"okr_id": f"okr_{rid(rng)}", "progress": 0}
    return user, args, result, f"OKR {result['okr_id']} recorded."

def gen_update_okr_progress(rng):
    oid = f"okr_{rid(rng)}"
    pct = rng.randint(10, 100)
    args = {"okr_id": oid, "progress": pct}
    user = f"Set {oid} progress to {pct}%."
    result = {"okr_id": oid, "progress": pct}
    return user, args, result, f"{oid} now at {pct}%."

def gen_record_one_on_one_note(rng):
    person = rng.choice(FIRST_NAMES)
    note = rng.choice(["Wants stretch project on infra.", "Concerned about scope creep.",
                       "Ready for promo case Q3.", "Needs more design feedback loops."])
    args = {"report": person, "date": date(rng), "note": note}
    user = f"Log a 1:1 note for {person}: {note}"
    result = {"note_id": f"note_{rid(rng)}"}
    return user, args, result, f"1:1 note saved as {result['note_id']}."

# Tool descriptions and minimal schemas
TOOL_SCHEMAS = {
    "create_task": ("Create a new task.", {"title": "string", "due": "date", "assignee": "string"}, ["title"]),
    "complete_task": ("Mark a task complete.", {"task_id": "string"}, ["task_id"]),
    "update_task_due": ("Change a task's due date.", {"task_id": "string", "due": "date"}, ["task_id", "due"]),
    "assign_task": ("Assign a task to a user.", {"task_id": "string", "assignee": "string"}, ["task_id", "assignee"]),
    "list_tasks": ("List tasks in a project.", {"project": "string", "status": "string"}, ["project"]),
    "delete_task": ("Delete a task.", {"task_id": "string"}, ["task_id"]),
    "reopen_task": ("Reopen a completed task.", {"task_id": "string"}, ["task_id"]),
    "add_subtask": ("Add a subtask to a parent task.", {"parent_id": "string", "title": "string"}, ["parent_id", "title"]),
    "set_task_priority": ("Set task priority.", {"task_id": "string", "priority": "low|medium|high|urgent"}, ["task_id", "priority"]),
    "create_project": ("Create a project.", {"name": "string", "owner": "string"}, ["name"]),
    "archive_project": ("Archive a project.", {"project_id": "string"}, ["project_id"]),
    "add_project_member": ("Add a member to a project.", {"project_id": "string", "user": "string", "role": "string"}, ["project_id", "user"]),
    "remove_project_member": ("Remove a member from a project.", {"project_id": "string", "user": "string"}, ["project_id", "user"]),
    "rename_project": ("Rename a project.", {"project_id": "string", "name": "string"}, ["project_id", "name"]),
    "set_project_owner": ("Set the owner of a project.", {"project_id": "string", "owner": "string"}, ["project_id", "owner"]),
    "create_issue": ("Open a new issue/ticket.", {"project": "string", "title": "string", "type": "bug|task|story"}, ["project", "title"]),
    "update_issue": ("Update fields on an issue.", {"key": "string", "fields": "object"}, ["key", "fields"]),
    "close_issue": ("Close an issue with a resolution.", {"issue_key": "string", "resolution": "string"}, ["issue_key", "resolution"]),
    "comment_on_issue": ("Add a comment to an issue.", {"key": "string", "body": "string"}, ["key", "body"]),
    "get_issue": ("Fetch issue details.", {"key": "string"}, ["key"]),
    "link_issues": ("Create a relationship between two issues.", {"from_key": "string", "to_key": "string", "relation": "string"}, ["from_key", "to_key", "relation"]),
    "move_issue_status": ("Transition an issue to a new status.", {"key": "string", "status": "string"}, ["key", "status"]),
    "add_label": ("Add a label to an issue.", {"key": "string", "label": "string"}, ["key", "label"]),
    "remove_label": ("Remove a label from an issue.", {"key": "string", "label": "string"}, ["key", "label"]),
    "assign_issue": ("Assign an issue to a user.", {"key": "string", "assignee": "string"}, ["key", "assignee"]),
    "start_sprint": ("Start a sprint.", {"name": "string", "start": "date", "end": "date"}, ["name"]),
    "end_sprint": ("End a sprint.", {"sprint_id": "string"}, ["sprint_id"]),
    "velocity_report": ("Compute team velocity.", {"team": "string", "last_n_sprints": "integer"}, ["team"]),
    "add_to_sprint": ("Add an issue to a sprint.", {"key": "string", "sprint_id": "string"}, ["key", "sprint_id"]),
    "remove_from_sprint": ("Remove an issue from its sprint.", {"key": "string"}, ["key"]),
    "request_pto": ("Submit a PTO request.", {"start": "date", "end": "date", "reason": "string"}, ["start", "end"]),
    "cancel_pto": ("Cancel a PTO request.", {"request_id": "string"}, ["request_id"]),
    "get_remaining_pto": ("Get remaining PTO days for an employee.", {"employee_id": "string"}, ["employee_id"]),
    "submit_timesheet": ("Submit a timesheet for a week.", {"week": "string", "hours": "number"}, ["week", "hours"]),
    "lookup_employee": ("Look up an employee record.", {"email": "string"}, ["email"]),
    "update_org_chart": ("Update reporting line.", {"employee_id": "string", "new_manager": "string"}, ["employee_id", "new_manager"]),
    "approve_timesheet": ("Approve a timesheet.", {"timesheet_id": "string"}, ["timesheet_id"]),
    "submit_expense": ("Submit an expense report line.", {"category": "string", "amount": "number", "currency": "string", "date": "date"}, ["category", "amount"]),
    "approve_expense": ("Approve an expense.", {"expense_id": "string"}, ["expense_id"]),
    "reject_expense": ("Reject an expense.", {"expense_id": "string", "reason": "string"}, ["expense_id", "reason"]),
    "add_lead": ("Add a CRM lead.", {"name": "string", "company": "string", "source": "string", "email": "string"}, ["name", "company"]),
    "convert_lead_to_opportunity": ("Convert a lead to an opportunity.", {"lead_id": "string", "amount": "number"}, ["lead_id"]),
    "update_opportunity_stage": ("Update opportunity stage.", {"opportunity_id": "string", "stage": "string"}, ["opportunity_id", "stage"]),
    "log_call": ("Log a sales/CRM call.", {"contact": "string", "duration_min": "integer", "notes": "string"}, ["contact"]),
    "log_email_to_crm": ("Log an email activity to CRM.", {"contact_email": "string", "subject": "string", "direction": "inbound|outbound"}, ["contact_email", "subject"]),
    "create_deal": ("Create a CRM deal.", {"company": "string", "amount": "number", "stage": "string", "close_date": "date"}, ["company", "amount"]),
    "forecast_pipeline": ("Forecast pipeline for a team/quarter.", {"quarter": "string", "team": "string"}, ["quarter"]),
    "send_invoice": ("Send an invoice to a customer.", {"customer": "string", "amount": "number", "due_date": "date"}, ["customer", "amount"]),
    "mark_invoice_paid": ("Mark an invoice as paid.", {"invoice_id": "string", "paid_date": "date"}, ["invoice_id"]),
    "set_okr": ("Record a new OKR.", {"team": "string", "objective": "string", "quarter": "string"}, ["team", "objective"]),
    "update_okr_progress": ("Update OKR progress percent.", {"okr_id": "string", "progress": "integer"}, ["okr_id", "progress"]),
    "record_one_on_one_note": ("Record a 1:1 meeting note.", {"report": "string", "date": "date", "note": "string"}, ["report", "note"]),
}

GENERATORS = {
    "create_task": gen_create_task, "complete_task": gen_complete_task,
    "update_task_due": gen_update_task_due, "assign_task": gen_assign_task,
    "list_tasks": gen_list_tasks, "delete_task": gen_delete_task,
    "reopen_task": gen_reopen_task, "add_subtask": gen_add_subtask,
    "set_task_priority": gen_set_task_priority,
    "create_project": gen_create_project, "archive_project": gen_archive_project,
    "add_project_member": gen_add_project_member, "remove_project_member": gen_remove_project_member,
    "rename_project": gen_rename_project, "set_project_owner": gen_set_project_owner,
    "create_issue": gen_create_issue, "update_issue": gen_update_issue,
    "close_issue": gen_close_issue, "comment_on_issue": gen_comment_on_issue,
    "get_issue": gen_get_issue, "link_issues": gen_link_issues,
    "move_issue_status": gen_move_issue_status, "add_label": gen_add_label,
    "remove_label": gen_remove_label, "assign_issue": gen_assign_issue,
    "start_sprint": gen_start_sprint, "end_sprint": gen_end_sprint,
    "velocity_report": gen_velocity_report, "add_to_sprint": gen_add_to_sprint,
    "remove_from_sprint": gen_remove_from_sprint,
    "request_pto": gen_request_pto, "cancel_pto": gen_cancel_pto,
    "get_remaining_pto": gen_get_remaining_pto, "submit_timesheet": gen_submit_timesheet,
    "lookup_employee": gen_lookup_employee, "update_org_chart": gen_update_org_chart,
    "approve_timesheet": gen_approve_timesheet,
    "submit_expense": gen_submit_expense, "approve_expense": gen_approve_expense,
    "reject_expense": gen_reject_expense,
    "add_lead": gen_add_lead, "convert_lead_to_opportunity": gen_convert_lead_to_opportunity,
    "update_opportunity_stage": gen_update_opportunity_stage,
    "log_call": gen_log_call, "log_email_to_crm": gen_log_email_to_crm,
    "create_deal": gen_create_deal, "forecast_pipeline": gen_forecast_pipeline,
    "send_invoice": gen_send_invoice, "mark_invoice_paid": gen_mark_invoice_paid,
    "set_okr": gen_set_okr, "update_okr_progress": gen_update_okr_progress,
    "record_one_on_one_note": gen_record_one_on_one_note,
}

assert set(TOOLS) == set(GENERATORS.keys()) == set(TOOL_SCHEMAS.keys())

BLACKLISTED = [
    "i've gathered all the information",
    "i've completed the task",
    "here's what i found:",
    "based on the results,",
    "the results show that",
]

def assert_no_blacklist(text):
    low = text.lower()
    for b in BLACKLISTED:
        assert b not in low, f"Blacklisted opener: {b!r} in {text!r}"

def tool_schema_obj(name):
    desc, props, required = TOOL_SCHEMAS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": ("integer" if v == "integer" else
                                            "number" if v == "number" else
                                            "object" if v == "object" else "string"),
                                   "description": v}
                               for k, v in props.items()},
                "required": required,
            },
        },
    }

def build_tool_quota():
    """Return a list of tool names of length N_TOTAL respecting <=5% per tool (=25 max)."""
    cap = 25  # 5% of 500
    base = N_TOTAL // len(TOOLS)  # 10
    rem = N_TOTAL - base * len(TOOLS)  # 0 with 50 tools
    counts = {t: base for t in TOOLS}
    # distribute remainder
    extras = list(TOOLS)
    rng = random.Random(SEED + "quota")
    rng.shuffle(extras)
    for i in range(rem):
        counts[extras[i]] += 1
    # ensure cap respected
    for t, c in counts.items():
        assert c <= cap, f"{t}={c} exceeds cap {cap}"
    out = []
    for t, c in counts.items():
        out.extend([t] * c)
    rng.shuffle(out)
    assert len(out) == N_TOTAL
    return out

def main():
    rng = random.Random(SEED)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    quota = build_tool_quota()
    # Decide single-turn vs multi-turn slots
    n_single = int(round(N_TOTAL * SINGLE_TURN_FRAC))  # 75
    indices = list(range(N_TOTAL))
    rng.shuffle(indices)
    single_idx = set(indices[:n_single])

    # Suffix pool quota for multi-turn samples
    n_multi = N_TOTAL - n_single  # 425
    base = n_multi // len(SUFFIX_POOL)  # 14
    rem = n_multi - base * len(SUFFIX_POOL)
    suffix_counts = {i: base for i in range(len(SUFFIX_POOL))}
    extras = list(range(len(SUFFIX_POOL)))
    rng_suf = random.Random(SEED + "suf")
    rng_suf.shuffle(extras)
    for i in range(rem):
        suffix_counts[extras[i]] += 1
    suffix_bag = []
    for idx, c in suffix_counts.items():
        suffix_bag.extend([idx] * c)
    rng_suf.shuffle(suffix_bag)

    samples = []
    suffix_pos = 0
    tool_count = Counter()
    suffix_use = Counter()

    for i in range(N_TOTAL):
        tool_name = quota[i]
        gen = GENERATORS[tool_name]
        user, args, result, tail = gen(rng)
        sysprompt = rng.choice(SYSTEM_PROMPTS)
        result_str = json.dumps(result, separators=(", ", ": "))
        # keep tool result <=200 chars
        if len(result_str) > 200:
            # trim by dropping fields (best-effort)
            for k in list(result.keys())[3:]:
                result.pop(k, None)
            result_str = json.dumps(result, separators=(", ", ": "))
        # Pad short results occasionally? not needed; they're 10-200 already typically
        if len(result_str) < 10:
            result["ok"] = True
            result_str = json.dumps(result, separators=(", ", ": "))

        is_single = i in single_idx
        if is_single:
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": tool_name, "arguments": args}}]},
            ]
        else:
            sidx = suffix_bag[suffix_pos]
            suffix_pos += 1
            suffix = SUFFIX_POOL[sidx]
            suffix_use[sidx] += 1
            final = f"{suffix} {tail}"
            assert_no_blacklist(final)
            messages = [
                {"role": "system", "content": sysprompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"type": "function",
                                 "function": {"name": tool_name, "arguments": args}}]},
                {"role": "tool", "name": tool_name, "content": result_str},
                {"role": "assistant", "content": final},
            ]
        sample = {
            "messages": messages,
            "tools": [tool_schema_obj(tool_name)],
            "domain": "tool-calling",
        }
        samples.append(sample)
        tool_count[tool_name] += 1

    # Validate
    assert len(samples) == N_TOTAL
    for t, c in tool_count.items():
        assert c <= 25, f"{t} count {c} exceeds 5%"
    assert len(tool_count) >= 40

    with open(OUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    # Report
    print(f"Wrote {len(samples)} samples to {OUT_PATH}")
    print(f"Distinct tools: {len(tool_count)}")
    print(f"Single-turn: {n_single}; Multi-turn: {n_multi}")
    print(f"Max tool freq: {max(tool_count.values())} ({max(tool_count.values())/N_TOTAL:.1%})")
    print(f"Suffix-pool unique used: {len(suffix_use)}/{len(SUFFIX_POOL)}")
    print(f"Suffix usage min/max: {min(suffix_use.values())}/{max(suffix_use.values())}")

if __name__ == "__main__":
    main()
