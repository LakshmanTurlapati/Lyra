"""TC-F2: Fresh tool-calling batch for task/project/HR/CRM/ticketing domain.

500 ShareGPT samples, distinct fixtures from TC-F wave 1.
Run: python gen_tc_09_3_F2.py
Output: /Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-08-F.jsonl
"""
import json
import os
import random
from collections import Counter

from gen_tc_09_3_F2_data import (
    PROJECT_KEYS, LIN_PREFIXES, FIRST_NAMES, LAST_NAMES, COMPANIES,
    SPRINT_NAMES, TASK_TITLES, LEAD_SOURCES, DEAL_STAGES, EXPENSE_CATS, EMP_DEPTS,
)

SEED = "1009308F"
OUT_PATH = "/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-08-F.jsonl"
N_TOTAL = 500
N_SINGLE = 75
N_MULTI = 425

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
assert len(SUFFIX_POOL) == 30

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a helpful assistant. Use tools when they would help answer the user.",
    "You are a helpful assistant that calls functions to complete tasks accurately.",
    "You are an assistant with access to internal tools. Call them when needed.",
    "You are a productivity assistant connected to backend systems via tools.",
]

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


def rid(rng, lo=100, hi=99999):
    return rng.randint(lo, hi)


def issue_key(rng):
    return f"{rng.choice(PROJECT_KEYS)}-{rid(rng, 100, 9999)}"


def lin_key(rng):
    return f"{rng.choice(LIN_PREFIXES)}-{rid(rng, 100, 9999)}"


def employee_id(rng):
    return f"EMP-{rid(rng, 1000, 99999)}"


def email_addr(rng):
    return f"{rng.choice(FIRST_NAMES)}.{rng.choice(LAST_NAMES)}@{rng.choice(COMPANIES)}.com"


def person(rng):
    return f"{rng.choice(FIRST_NAMES).capitalize()} {rng.choice(LAST_NAMES).capitalize()}"


def date_str(rng):
    return f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def money(rng, lo=10, hi=5000):
    return round(rng.uniform(lo, hi), 2)


# ----- per-tool generators (fresh phrasings) -----
def gen_create_task(rng):
    title = rng.choice(TASK_TITLES)
    due = date_str(rng)
    args = {"title": title, "due": due, "assignee": rng.choice(FIRST_NAMES)}
    user = f"Spin up a task — '{title}', due {due}, owner is {args['assignee']}."
    tid = f"task_{rid(rng)}"
    result = {"id": tid, "title": title, "due": due, "assignee": args["assignee"], "status": "open"}
    return user, args, result, f"Task {tid} is on the board."


def gen_complete_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Flip {tid} to done for me."
    result = {"task_id": tid, "status": "done"}
    return user, args, result, f"{tid} is closed out."


def gen_update_task_due(rng):
    tid = f"task_{rid(rng)}"
    new_due = date_str(rng)
    args = {"task_id": tid, "due": new_due}
    user = f"Slide {tid}'s deadline to {new_due}."
    result = {"task_id": tid, "due": new_due, "updated": True}
    return user, args, result, f"{tid} is now due {new_due}."


def gen_assign_task(rng):
    tid = f"task_{rid(rng)}"
    p = rng.choice(FIRST_NAMES)
    args = {"task_id": tid, "assignee": p}
    user = f"Hand {tid} over to {p}."
    result = {"task_id": tid, "assignee": p}
    return user, args, result, f"{p} is on {tid}."


def gen_list_tasks(rng):
    proj = rng.choice(PROJECT_KEYS).lower()
    args = {"project": proj, "status": "open"}
    user = f"Show me what's still open on {proj}."
    result = {"project": proj, "count": rng.randint(2, 18),
              "tasks": [f"task_{rid(rng)}" for _ in range(3)]}
    return user, args, result, f"{result['count']} open tasks on {proj}."


def gen_delete_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Trash {tid} — it's a dupe of another one."
    result = {"task_id": tid, "deleted": True}
    return user, args, result, f"{tid} gone."


def gen_reopen_task(rng):
    tid = f"task_{rid(rng)}"
    args = {"task_id": tid}
    user = f"Bring {tid} back — turns out it's not done."
    result = {"task_id": tid, "status": "open"}
    return user, args, result, f"{tid} reopened."


def gen_add_subtask(rng):
    tid = f"task_{rid(rng)}"
    sub = rng.choice(["draft the design doc", "schedule a sync",
                      "spike on alternatives", "loop in security",
                      "update the changelog", "request a PR review"])
    args = {"parent_id": tid, "title": sub}
    user = f"Tack on a subtask under {tid}: '{sub}'."
    sid = f"task_{rid(rng)}"
    result = {"id": sid, "parent": tid, "title": sub}
    return user, args, result, f"Subtask {sid} attached."


def gen_set_task_priority(rng):
    tid = f"task_{rid(rng)}"
    pri = rng.choice(["low", "medium", "high", "urgent"])
    args = {"task_id": tid, "priority": pri}
    user = f"Set {tid} to {pri}."
    result = {"task_id": tid, "priority": pri}
    return user, args, result, f"{tid} now {pri}."


def gen_create_project(rng):
    nm = f"{rng.choice(['Halcyon', 'Ironwood', 'Juniper', 'Kestrel', 'Larkspur', 'Mariner'])} {rng.choice(['Initiative', 'Effort', 'Programme', 'Track'])}"
    args = {"name": nm, "owner": rng.choice(FIRST_NAMES)}
    user = f"Stand up a new project '{nm}', led by {args['owner']}."
    pid = f"proj_{rid(rng)}"
    result = {"id": pid, "name": nm, "owner": args["owner"]}
    return user, args, result, f"{pid} is created."


def gen_archive_project(rng):
    pid = f"proj_{rid(rng)}"
    args = {"project_id": pid}
    user = f"Mothball {pid}; it's been wrapped for a quarter."
    result = {"project_id": pid, "archived": True}
    return user, args, result, f"{pid} archived."


def gen_add_project_member(rng):
    pid = f"proj_{rid(rng)}"
    p = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "user": p, "role": rng.choice(["member", "admin", "viewer"])}
    user = f"Loop {p} into {pid} as a {args['role']}."
    result = {"project_id": pid, "user": p, "role": args["role"]}
    return user, args, result, f"{p} added to {pid}."


def gen_remove_project_member(rng):
    pid = f"proj_{rid(rng)}"
    p = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "user": p}
    user = f"Pull {p} off {pid}."
    result = {"project_id": pid, "user": p, "removed": True}
    return user, args, result, f"{p} removed from {pid}."


def gen_rename_project(rng):
    pid = f"proj_{rid(rng)}"
    new = rng.choice(["Polaris", "Quasar", "Redwood", "Saffron", "Tundra-2"])
    args = {"project_id": pid, "name": new}
    user = f"Retitle {pid} to '{new}'."
    result = {"project_id": pid, "name": new}
    return user, args, result, f"{pid} now '{new}'."


def gen_set_project_owner(rng):
    pid = f"proj_{rid(rng)}"
    p = rng.choice(FIRST_NAMES)
    args = {"project_id": pid, "owner": p}
    user = f"Make {p} the new owner of {pid}."
    result = {"project_id": pid, "owner": p}
    return user, args, result, f"{p} owns {pid} now."


def gen_create_issue(rng):
    proj = rng.choice(PROJECT_KEYS)
    title = rng.choice([
        "Refunds API returns 500 intermittently",
        "Webhook signature mismatch on retries",
        "Onboarding wizard skips step 3 on Firefox",
        "Stripe Connect payouts delayed by a day",
        "Mobile app crashes when tapping receipts",
        "Search index missing recent records",
    ])
    args = {"project": proj, "title": title, "type": rng.choice(["bug", "task", "story"])}
    user = f"File a {args['type']} on {proj}: {title}"
    key = f"{proj}-{rid(rng, 1000, 9999)}"
    result = {"key": key, "title": title}
    return user, args, result, f"Filed {key}."


def gen_update_issue(rng):
    key = issue_key(rng)
    args = {"key": key, "fields": {"priority": rng.choice(["P0", "P1", "P2", "P3"])}}
    user = f"Set {key} to {args['fields']['priority']}."
    result = {"key": key, "updated": True}
    return user, args, result, f"{key} updated."


def gen_close_issue(rng):
    key = issue_key(rng)
    res = rng.choice(["fixed", "wontfix", "duplicate", "cannot_reproduce", "by_design"])
    args = {"issue_key": key, "resolution": res}
    user = f"Resolve {key} as {res.replace('_', ' ')}."
    result = {"key": key, "status": "closed", "resolution": res}
    return user, args, result, f"{key} closed ({res})."


def gen_comment_on_issue(rng):
    key = issue_key(rng)
    body = rng.choice([
        "Verified locally; reproducible on main.",
        "Owner is on PTO until Monday — leaving as-is.",
        "Linking the related PR for visibility.",
        "Customer pinged again; bumping priority.",
    ])
    args = {"key": key, "body": body}
    user = f"Add a comment on {key}: {body}"
    result = {"key": key, "comment_id": rid(rng)}
    return user, args, result, f"Comment posted to {key}."


def gen_get_issue(rng):
    key = lin_key(rng)
    args = {"key": key}
    user = f"What's the status on {key}?"
    result = {"key": key, "assignee": rng.choice(FIRST_NAMES),
              "status": rng.choice(["in_progress", "review", "todo", "blocked"])}
    return user, args, result, f"{result['assignee'].capitalize()} on {key}, {result['status']}."


def gen_link_issues(rng):
    a, b = issue_key(rng), issue_key(rng)
    rel = rng.choice(["blocks", "blocked_by", "duplicates", "relates_to", "caused_by"])
    args = {"from_key": a, "to_key": b, "relation": rel}
    user = f"Mark {a} as {rel.replace('_', ' ')} {b}."
    result = {"from": a, "to": b, "relation": rel}
    return user, args, result, f"Linked {a} {rel} {b}."


def gen_move_issue_status(rng):
    key = issue_key(rng)
    status = rng.choice(["todo", "in_progress", "review", "done", "blocked", "qa"])
    args = {"key": key, "status": status}
    user = f"Slide {key} into {status.replace('_', ' ')}."
    result = {"key": key, "status": status}
    return user, args, result, f"{key} → {status}."


def gen_add_label(rng):
    key = issue_key(rng)
    label = rng.choice(["needs-design", "blocked-on-vendor", "production-impact",
                        "low-effort", "spike", "compliance"])
    args = {"key": key, "label": label}
    user = f"Slap a '{label}' label on {key}."
    result = {"key": key, "label": label, "added": True}
    return user, args, result, f"'{label}' on {key}."


def gen_remove_label(rng):
    key = issue_key(rng)
    label = rng.choice(["needs-design", "wip", "draft", "spike", "blocked"])
    args = {"key": key, "label": label}
    user = f"Take '{label}' off {key}."
    result = {"key": key, "label": label, "removed": True}
    return user, args, result, f"'{label}' removed from {key}."


def gen_assign_issue(rng):
    key = issue_key(rng)
    p = rng.choice(FIRST_NAMES)
    args = {"key": key, "assignee": p}
    user = f"Assign {key} over to {p}."
    result = {"key": key, "assignee": p}
    return user, args, result, f"{key} now {p}'s."


def gen_start_sprint(rng):
    nm = f"Sprint {rng.choice(SPRINT_NAMES)}"
    args = {"name": nm, "start": date_str(rng), "end": date_str(rng)}
    user = f"Fire up {nm}."
    result = {"name": nm, "status": "active", "id": f"sprint_{rid(rng)}"}
    return user, args, result, f"{nm} is rolling."


def gen_end_sprint(rng):
    sid = f"sprint_{rid(rng)}"
    args = {"sprint_id": sid}
    user = f"Close out {sid}."
    result = {"sprint_id": sid, "status": "completed", "completed_points": rng.randint(15, 60)}
    return user, args, result, f"{sid} done at {result['completed_points']} pts."


def gen_velocity_report(rng):
    team = rng.choice(["payments", "trust", "search", "ml-platform", "billing"])
    args = {"team": team, "last_n_sprints": rng.choice([3, 5, 10])}
    user = f"What's velocity been on {team} the last {args['last_n_sprints']} sprints?"
    result = {"team": team, "avg_velocity": rng.randint(20, 50)}
    return user, args, result, f"{team} averaging {result['avg_velocity']} pts."


def gen_add_to_sprint(rng):
    key = issue_key(rng)
    sid = f"sprint_{rid(rng)}"
    args = {"key": key, "sprint_id": sid}
    user = f"Drop {key} into {sid}."
    result = {"key": key, "sprint_id": sid}
    return user, args, result, f"{key} in {sid}."


def gen_remove_from_sprint(rng):
    key = issue_key(rng)
    args = {"key": key}
    user = f"Yank {key} out of the current sprint."
    result = {"key": key, "removed_from_sprint": True}
    return user, args, result, f"{key} pulled."


def gen_request_pto(rng):
    sd, ed = date_str(rng), date_str(rng)
    reason = rng.choice(["honeymoon", "kid_sick", "moving", "conference", "bereavement"])
    args = {"start": sd, "end": ed, "reason": reason}
    user = f"Put in PTO {sd} through {ed} — {reason.replace('_', ' ')}."
    result = {"request_id": f"pto_{rid(rng)}", "status": "pending"}
    return user, args, result, f"PTO {result['request_id']} filed."


def gen_cancel_pto(rng):
    pid = f"pto_{rid(rng)}"
    args = {"request_id": pid}
    user = f"Pull back {pid} — plans changed."
    result = {"request_id": pid, "status": "cancelled"}
    return user, args, result, f"{pid} cancelled."


def gen_get_remaining_pto(rng):
    eid = employee_id(rng)
    args = {"employee_id": eid}
    user = f"Days remaining on {eid}'s PTO balance?"
    result = {"employee_id": eid, "remaining_days": rng.randint(0, 25)}
    return user, args, result, f"{eid}: {result['remaining_days']} days left."


def gen_submit_timesheet(rng):
    week = f"2026-W{rng.randint(1, 52):02d}"
    hrs = rng.randint(35, 50)
    args = {"week": week, "hours": hrs}
    user = f"Push my timesheet for {week} — {hrs} hrs."
    result = {"week": week, "hours": hrs, "status": "submitted"}
    return user, args, result, f"{week} timesheet in."


def gen_lookup_employee(rng):
    em = email_addr(rng)
    args = {"email": em}
    user = f"Look up the record for {em}."
    result = {"email": em, "id": employee_id(rng), "department": rng.choice(EMP_DEPTS),
              "manager": rng.choice(FIRST_NAMES)}
    return user, args, result, f"{em} sits in {result['department']}."


def gen_update_org_chart(rng):
    eid = employee_id(rng)
    mgr = rng.choice(FIRST_NAMES)
    args = {"employee_id": eid, "new_manager": mgr}
    user = f"Reassign {eid} under {mgr}."
    result = {"employee_id": eid, "manager": mgr}
    return user, args, result, f"{eid} → {mgr}."


def gen_approve_timesheet(rng):
    tsid = f"ts_{rid(rng)}"
    args = {"timesheet_id": tsid}
    user = f"Sign off on {tsid}."
    result = {"timesheet_id": tsid, "status": "approved"}
    return user, args, result, f"{tsid} approved."


def gen_submit_expense(rng):
    cat = rng.choice(EXPENSE_CATS)
    amt = money(rng)
    args = {"category": cat, "amount": amt, "currency": "USD", "date": date_str(rng)}
    user = f"Log a ${amt} {cat.replace('_', ' ')} expense from {args['date']}."
    result = {"expense_id": f"exp_{rid(rng)}", "status": "pending"}
    return user, args, result, f"Expense {result['expense_id']} in."


def gen_approve_expense(rng):
    eid = f"exp_{rid(rng)}"
    args = {"expense_id": eid}
    user = f"OK {eid} for me."
    result = {"expense_id": eid, "status": "approved"}
    return user, args, result, f"{eid} approved."


def gen_reject_expense(rng):
    eid = f"exp_{rid(rng)}"
    reason = rng.choice(["over_limit", "personal_charge", "no_business_purpose", "stale_date"])
    args = {"expense_id": eid, "reason": reason}
    user = f"Bounce {eid}: {reason.replace('_', ' ')}."
    result = {"expense_id": eid, "status": "rejected", "reason": reason}
    return user, args, result, f"{eid} rejected ({reason})."


def gen_add_lead(rng):
    nm = person(rng)
    src = rng.choice(LEAD_SOURCES)
    co = rng.choice(COMPANIES).capitalize()
    args = {"name": nm, "company": co, "source": src, "email": email_addr(rng)}
    user = f"New lead — {nm} at {co}, came in via {src.replace('_', ' ')}."
    result = {"lead_id": f"lead_{rid(rng)}", "status": "new"}
    return user, args, result, f"{result['lead_id']} added."


def gen_convert_lead_to_opportunity(rng):
    lid = f"lead_{rid(rng)}"
    args = {"lead_id": lid, "amount": money(rng, 1000, 50000)}
    user = f"Promote {lid} into an opportunity at ${args['amount']}."
    result = {"lead_id": lid, "opportunity_id": f"opp_{rid(rng)}", "amount": args["amount"]}
    return user, args, result, f"{result['opportunity_id']} created."


def gen_update_opportunity_stage(rng):
    oid = f"opp_{rid(rng)}"
    stg = rng.choice(DEAL_STAGES)
    args = {"opportunity_id": oid, "stage": stg}
    user = f"Advance {oid} to {stg.replace('_', ' ')}."
    result = {"opportunity_id": oid, "stage": stg}
    return user, args, result, f"{oid} at {stg}."


def gen_log_call(rng):
    contact = person(rng)
    dur = rng.randint(5, 60)
    args = {"contact": contact, "duration_min": dur, "notes": "Reviewed implementation timeline."}
    user = f"Log {dur} min with {contact} — talked through implementation timeline."
    result = {"activity_id": f"act_{rid(rng)}", "type": "call"}
    return user, args, result, f"Call logged ({result['activity_id']})."


def gen_log_email_to_crm(rng):
    em = email_addr(rng)
    subj = rng.choice([
        "Re: Security questionnaire", "Pilot kickoff next week",
        "Pricing addendum", "Champion change at the account",
    ])
    args = {"contact_email": em, "subject": subj, "direction": rng.choice(["inbound", "outbound"])}
    user = f"Log {args['direction']} email to {em}, subject '{subj}'."
    result = {"activity_id": f"act_{rid(rng)}", "type": "email"}
    return user, args, result, f"Email captured on {em}."


def gen_create_deal(rng):
    co = rng.choice(COMPANIES).capitalize()
    amt = money(rng, 5000, 250000)
    args = {"company": co, "amount": amt, "stage": "intro_call", "close_date": date_str(rng)}
    user = f"New deal — {co}, ${amt}, target close {args['close_date']}."
    result = {"deal_id": f"deal_{rid(rng)}", "amount": amt}
    return user, args, result, f"{result['deal_id']} created for {co}."


def gen_forecast_pipeline(rng):
    q = f"Q{rng.randint(1, 4)} 2026"
    args = {"quarter": q, "team": rng.choice(["named_accounts", "commercial", "self_serve"])}
    user = f"Forecast for the {args['team']} team in {q}."
    result = {"quarter": q, "weighted": round(money(rng, 100000, 2000000), 2),
              "deals": rng.randint(8, 60)}
    return user, args, result, f"{q}: ${result['weighted']:,.0f} weighted."


def gen_send_invoice(rng):
    co = rng.choice(COMPANIES).capitalize()
    amt = money(rng, 500, 25000)
    args = {"customer": co, "amount": amt, "due_date": date_str(rng)}
    user = f"Bill {co} ${amt}, net due {args['due_date']}."
    result = {"invoice_id": f"inv_{rid(rng)}", "status": "sent"}
    return user, args, result, f"{result['invoice_id']} sent."


def gen_mark_invoice_paid(rng):
    iid = f"inv_{rid(rng)}"
    args = {"invoice_id": iid, "paid_date": date_str(rng)}
    user = f"Settle {iid} as of {args['paid_date']}."
    result = {"invoice_id": iid, "status": "paid"}
    return user, args, result, f"{iid} paid."


def gen_set_okr(rng):
    obj = rng.choice([
        "Reduce churn by 20%",
        "Hit 99.95% uptime on payments",
        "Ship 3 design partner integrations",
        "Lower CAC by 25%",
    ])
    args = {"team": rng.choice(["payments", "marketing", "infra", "cs"]),
            "objective": obj, "quarter": f"Q{rng.randint(1, 4)} 2026"}
    user = f"Log an OKR for {args['team']}: {obj} in {args['quarter']}."
    result = {"okr_id": f"okr_{rid(rng)}", "progress": 0}
    return user, args, result, f"OKR {result['okr_id']} on the books."


def gen_update_okr_progress(rng):
    oid = f"okr_{rid(rng)}"
    pct = rng.randint(10, 100)
    args = {"okr_id": oid, "progress": pct}
    user = f"Update {oid} to {pct}%."
    result = {"okr_id": oid, "progress": pct}
    return user, args, result, f"{oid}: {pct}%."


def gen_record_one_on_one_note(rng):
    p = rng.choice(FIRST_NAMES)
    note = rng.choice([
        "Wants more cross-team exposure.",
        "Frustrated with ambiguous priorities.",
        "Excited about the new platform direction.",
        "Looking for mentorship outside the team.",
    ])
    args = {"report": p, "date": date_str(rng), "note": note}
    user = f"Drop a 1:1 note for {p}: {note}"
    result = {"note_id": f"note_{rid(rng)}"}
    return user, args, result, f"1:1 saved as {result['note_id']}."


TOOLS = [
    "create_task", "complete_task", "update_task_due", "assign_task", "list_tasks",
    "delete_task", "reopen_task", "add_subtask", "set_task_priority",
    "create_project", "archive_project", "add_project_member", "remove_project_member",
    "rename_project", "set_project_owner",
    "create_issue", "update_issue", "close_issue", "comment_on_issue", "get_issue",
    "link_issues", "move_issue_status", "add_label", "remove_label", "assign_issue",
    "start_sprint", "end_sprint", "velocity_report", "add_to_sprint", "remove_from_sprint",
    "request_pto", "cancel_pto", "get_remaining_pto", "submit_timesheet",
    "lookup_employee", "update_org_chart", "approve_timesheet",
    "submit_expense", "approve_expense", "reject_expense",
    "add_lead", "convert_lead_to_opportunity", "update_opportunity_stage",
    "log_call", "log_email_to_crm", "create_deal", "forecast_pipeline",
    "send_invoice", "mark_invoice_paid", "set_okr", "update_okr_progress",
    "record_one_on_one_note",
]
assert len(set(TOOLS)) >= 40

GENERATORS = {
    "create_task": gen_create_task, "complete_task": gen_complete_task,
    "update_task_due": gen_update_task_due, "assign_task": gen_assign_task,
    "list_tasks": gen_list_tasks, "delete_task": gen_delete_task,
    "reopen_task": gen_reopen_task, "add_subtask": gen_add_subtask,
    "set_task_priority": gen_set_task_priority,
    "create_project": gen_create_project, "archive_project": gen_archive_project,
    "add_project_member": gen_add_project_member,
    "remove_project_member": gen_remove_project_member,
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

assert set(TOOLS) == set(GENERATORS.keys()) == set(TOOL_SCHEMAS.keys())


def tool_schema_obj(name):
    desc, props, required = TOOL_SCHEMAS[name]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {
                        "type": ("integer" if v == "integer"
                                 else "number" if v == "number"
                                 else "object" if v == "object"
                                 else "string"),
                        "description": v,
                    }
                    for k, v in props.items()
                },
                "required": required,
            },
        },
    }


def build_tool_quota():
    cap = 25  # 5%
    base = N_TOTAL // len(TOOLS)
    rem = N_TOTAL - base * len(TOOLS)
    counts = {t: base for t in TOOLS}
    extras = list(TOOLS)
    rng = random.Random(SEED + "quota")
    rng.shuffle(extras)
    for i in range(rem):
        counts[extras[i]] += 1
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

    indices = list(range(N_TOTAL))
    rng.shuffle(indices)
    single_idx = set(indices[:N_SINGLE])

    base = N_MULTI // len(SUFFIX_POOL)  # 14
    rem = N_MULTI - base * len(SUFFIX_POOL)  # 5
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
        if len(result_str) > 200:
            for k in list(result.keys())[3:]:
                result.pop(k, None)
            result_str = json.dumps(result, separators=(", ", ": "))
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

    assert len(samples) == N_TOTAL
    for t, c in tool_count.items():
        assert c <= 25, f"{t} count {c} exceeds 5%"
    assert len(tool_count) >= 40

    with open(OUT_PATH, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(samples)} samples to {OUT_PATH}")
    print(f"Distinct tools: {len(tool_count)}")
    print(f"Single-turn: {N_SINGLE}; Multi-turn: {N_MULTI}")
    print(f"Max tool freq: {max(tool_count.values())} ({max(tool_count.values()) / N_TOTAL:.1%})")
    print(f"Suffix-pool unique used: {len(suffix_use)}/{len(SUFFIX_POOL)}")
    print(f"Suffix usage min/max: {min(suffix_use.values())}/{max(suffix_use.values())}")


if __name__ == "__main__":
    main()
