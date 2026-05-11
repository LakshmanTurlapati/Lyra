#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_B.py -- Generate 500 ShareGPT tool-calling samples (TC-B: messaging).

Subagent B of phase 09.3 tool-calling. Domain: email, SMS, chat, push,
contacts, voice/call routing.

Seed: 1009307B (deterministic shuffle).

Output: datasets/tool-calling/raw-09.3/batch-07-B.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from pathlib import Path

from gen_tc_09_3_B_data import (
    NAMES, EMAILS, PHONES, CHANNELS, SUBJECTS,
    SAMPLES as BASE_SAMPLES, SUFFIX_POOL, TOOLS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-11-B.jsonl"
SEED = "1009311B"

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a productivity assistant with access to messaging, email, and contact tools. Call tools when the user's request maps to one.",
    "You are an assistant that helps the user manage communications. Use tools to take actions; respond concisely after tool results.",
    "You are a helpful assistant. When the user requests an action you have a tool for, call the tool with realistic arguments.",
]


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


# ------------------------------------------------------------------
# Expansion: we need ~500 samples total. Base pool has fewer; we
# generate parameterized variants for the most natural axes (contact
# names, phone numbers, channels, subjects, email IDs, message bodies).
# ------------------------------------------------------------------

EXTRA_BODIES = [
    "Running a few minutes late, will be there shortly.",
    "On my way home, anything you need from the store?",
    "Could you send me the link from earlier?",
    "Picked up the package, thanks again.",
    "Confirming our 3pm tomorrow.",
    "Did the build finish?",
    "Let me know when you're free for a quick call.",
    "Stuck in traffic — pushed our slot to 4.",
    "Files uploaded, take a look when you can.",
    "Ping when the deploy hits prod.",
    "Are we still on for Friday?",
    "Got the docs, reviewing now.",
    "Heads up: customer escalation incoming.",
    "Please ack when received.",
    "Thanks for the heads up earlier.",
    "Saw your message — replying tonight.",
    "Need a quick eye on PR #4471 if you have a sec.",
    "Coffee at 3?",
    "Mail came — your package is at the front desk.",
    "All quiet on prod, going offline.",
]

EXTRA_SUBJECTS_BODIES = [
    ("Q3 forecast review", "Pulling together the Q3 forecast — can you send updated numbers by Wed?"),
    ("Reschedule our 1:1", "Hey — can we push our 1:1 to Thursday at 11?"),
    ("Welcome aboard!", "Excited to have you on the team. Onboarding doc attached."),
    ("Vendor renewal", "Renewal terms look reasonable but I have two questions on SLAs."),
    ("Demo feedback", "Demo went well overall — three small UI nits I want to flag."),
    ("Outage postmortem draft", "First pass at the postmortem is in the doc. Comments welcome."),
    ("Hiring loop next week", "Five candidates, four panels — I'll send the schedule shortly."),
    ("Press release review", "Legal cleared the draft. Final read needed before EOD Friday."),
    ("Travel approval", "Requesting approval for the Madrid trip — agenda attached."),
    ("Conference talk acceptance", "Talk got accepted! Slides due in 6 weeks."),
    ("Holiday party RSVP", "Counting heads for the Dec 14 party. RSVP by Friday."),
    ("Annual review timing", "Ready to schedule reviews — sending self-assessment template now."),
    ("Security training reminder", "Reminder: security training due by end of month."),
    ("Office move logistics", "We're consolidating to floor 14 starting June 1."),
    ("Benefits enrollment open", "Open enrollment runs through the 30th — see attached guide."),
    ("Quarterly metrics", "Quarter-over-quarter dashboard attached. Highlights at the top."),
    ("Roadmap input request", "Looking for input on the H2 roadmap themes by next Wednesday."),
    ("Customer reference check", "BigCo asked for two references — willing to be one?"),
    ("Legal review needed", "Need a legal pass on the new MSA before we send."),
    ("All-hands agenda", "Posting the all-hands agenda — speak up if anything's missing."),
]


def expand_samples():
    """Build the full 500-sample list from base + parametric expansion."""
    samples = list(BASE_SAMPLES)

    # ---- Email-flavored expansion: send_email with EXTRA_SUBJECTS_BODIES ----
    for i, (subj, body) in enumerate(EXTRA_SUBJECTS_BODIES):
        recipient_email = EMAILS[i % len(EMAILS)]
        recipient_name = NAMES[i % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_email",
            "args": {"to": recipient_email, "subject": subj, "body": body},
            "user": f"Email {recipient_name} about {subj.lower()}: {body[:60]}",
            "result": f'{{"sent":true,"id":"msg_out_e{i:03d}"}}',
        })

    # ---- SMS expansion ----
    for i, body in enumerate(EXTRA_BODIES):
        contact = NAMES[(i * 3) % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_sms",
            "args": {"contact": contact, "body": body},
            "user": f"Text {contact}: {body}",
            "result": f'{{"sent":true,"sid":"sms_x{i:03d}"}}',
        })

    # ---- Chat message expansion ----
    chat_msgs = [
        ("Heads up — staging is flaky again.",  "#engineering"),
        ("Slides are ready for review.",         "#product-launch"),
        ("Backfill kicked off, ETA 2 hours.",    "#data-eng"),
        ("Pager handoff: I've got it now.",      "#oncall-rotation"),
        ("Design sync moved to Tue 11am.",       "#design-review"),
        ("Standup notes posted in the doc.",     "#standup-eu"),
        ("Welcome to the team!",                 "#general"),
        ("Outage cleared, RCA tomorrow.",        "#alerts-prod"),
        ("Friendly reminder: PRs need 2 reviewers.", "#engineering"),
        ("Coffee in the kitchen.",               "#kitchen-fridge"),
        ("ML eval results posted.",              "#ml-research"),
        ("Frontend release ships Thursday.",     "#frontend"),
        ("Backend deploy paused for review.",    "#backend-platform"),
        ("Security patch landed.",               "#security-incidents"),
        ("Reminder: HR review forms due Friday.","#hr-announcements"),
        ("Heading out — back online tomorrow.",  "#general"),
        ("Lunch & learn 12:30 today.",           "#general"),
        ("Anyone seen the staging logs?",        "#engineering"),
        ("Tickets triaged — list in the doc.",   "#engineering"),
        ("Doc updated with new schema.",         "#data-eng"),
    ]
    for i, (text, ch) in enumerate(chat_msgs):
        samples.append({
            "tool": "send_chat_message",
            "args": {"channel": ch, "text": text},
            "user": f"Post in {ch}: {text}",
            "result": f'{{"sent":true,"ts":"2026-05-08T1{i%10}:00:00Z"}}',
        })

    # ---- search_emails varied queries ----
    queries = [
        ("from:diego", '{"results":[{"id":"msg_d1"},{"id":"msg_d2"}],"total":2}'),
        ("subject:roadmap", '{"results":[{"id":"msg_r1","subject":"Roadmap proposal"}],"total":1}'),
        ("has:attachment after:2026-04-01", '{"results":[{"id":"msg_a1"},{"id":"msg_a2"},{"id":"msg_a3"}],"total":3}'),
        ("is:starred", '{"results":[],"total":0}'),
        ("from:legal", '{"results":[{"id":"msg_l1"},{"id":"msg_l2"}],"total":2}'),
        ("invoice 2026", '{"results":[{"id":"msg_i1","amount":1200}],"total":1}'),
        ("subject:welcome", '{"results":[{"id":"msg_w1"}],"total":1}'),
        ("from:hr@acme.io", '{"results":[{"id":"msg_h1"},{"id":"msg_h2"},{"id":"msg_h3"}],"total":3}'),
        ("from:elena subject:auth", '{"results":[{"id":"msg_ea1"}],"total":1}'),
        ("subject:postmortem", '{"results":[{"id":"msg_p1"}],"total":1}'),
        ("vacation", '{"results":[{"id":"msg_v1"},{"id":"msg_v2"}],"total":2}'),
        ("from:vendor@bigco.com", '{"results":[{"id":"msg_vb1"}],"total":1}'),
        ("conference talk", '{"results":[{"id":"msg_c1"}],"total":1}'),
        ("travel approval", '{"results":[{"id":"msg_t1"}],"total":1}'),
        ("subject:RFP", '{"results":[{"id":"msg_rfp1"},{"id":"msg_rfp2"}],"total":2}'),
    ]
    user_phrasings = [
        "Search my mail for {q}.",
        "Find emails matching {q}.",
        "Pull up emails: {q}.",
        "Look through email for {q}.",
        "Run a search: {q}.",
    ]
    for i, (q, res) in enumerate(queries):
        samples.append({
            "tool": "search_emails",
            "args": {"query": q},
            "user": user_phrasings[i % len(user_phrasings)].format(q=q),
            "result": res,
        })

    # ---- count_unread variants ----
    folder_phrasings = [
        ("Inbox",     "How many unread in Inbox?"),
        ("Drafts",    "Unread count in Drafts?"),
        ("Sent",      "Anything unread in Sent?"),
        ("Spam",      "Unread spam count?"),
        ("Archive",   "How many unread in Archive?"),
        ("Important", "Unread Important?"),
        ("Receipts",  "Unread Receipts?"),
        ("Updates",   "Updates folder unread?"),
        ("Forums",    "Forums unread count?"),
        ("Social",    "How many unread in Social?"),
    ]
    for i, (folder, user) in enumerate(folder_phrasings):
        samples.append({
            "tool": "count_unread_emails",
            "args": {"folder": folder},
            "user": user,
            "result": f'{{"count":{(i*7+3)%50}}}',
        })

    # ---- Mark/archive/delete variants ----
    for i in range(15):
        eid = f"msg_{1000+i*37}"
        samples.append({
            "tool": "mark_email_read",
            "args": {"email_id": eid},
            "user": f"Mark {eid} as read.",
            "result": '{"marked":true}',
        })
    for i in range(15):
        eid = f"msg_{2000+i*41}"
        samples.append({
            "tool": "archive_email",
            "args": {"email_id": eid},
            "user": f"Archive {eid}.",
            "result": '{"archived":true}',
        })
    for i in range(12):
        eid = f"msg_{3000+i*43}"
        samples.append({
            "tool": "delete_email",
            "args": {"email_id": eid},
            "user": f"Delete email {eid}.",
            "result": '{"deleted":true}',
        })

    # ---- Voice / call expansions ----
    for i in range(12):
        contact = NAMES[(i * 5) % len(NAMES)].split()[0]
        samples.append({
            "tool": "start_voice_call",
            "args": {"contact": contact},
            "user": f"Call {contact}.",
            "result": f'{{"call_id":"call_v{i:03d}","status":"ringing"}}',
        })

    for i in range(10):
        vid = f"vm_{8000+i*13}"
        samples.append({
            "tool": "transcribe_voicemail",
            "args": {"voicemail_id": vid},
            "user": f"What does voicemail {vid} say?",
            "result": f'{{"transcript":"Hey it\'s {NAMES[i%len(NAMES)].split()[0]}, please call me back when you get a chance."}}',
        })

    for i in range(8):
        cid = f"call_{500+i*17}"
        samples.append({
            "tool": "transcribe_call",
            "args": {"call_id": cid},
            "user": f"Transcribe call {cid}.",
            "result": f'{{"transcript":"Discussed the proposal, agreed on next steps, and confirmed timeline.","duration_sec":{180+i*30}}}',
        })

    for i in range(8):
        samples.append({
            "tool": "get_call_log",
            "args": {"limit": 3 + i},
            "user": f"Show my last {3+i} calls.",
            "result": '{"calls":[{"id":"call_a","with":"Sarah","duration_sec":300},{"id":"call_b","with":"+1-555-0142","duration_sec":62}]}',
        })

    for i in range(6):
        vid = f"vm_{9000+i*11}"
        samples.append({
            "tool": "delete_voicemail",
            "args": {"voicemail_id": vid},
            "user": f"Delete voicemail {vid}.",
            "result": '{"deleted":true}',
        })

    for i in range(6):
        samples.append({
            "tool": "get_voicemail_list",
            "args": {"limit": 3 + i},
            "user": f"List my last {3+i} voicemails.",
            "result": '{"voicemails":[{"id":"vm_8821","from":"Marcus","unread":true},{"id":"vm_8820","from":"Mom","unread":false}]}',
        })

    # ---- Push, schedule, OOO, DND, status, signature, folder ----
    push_specs = [
        ("Build broke", "CI failed on main: 3 tests red."),
        ("PR ready",    "Your PR has 2 approvals — ready to merge."),
        ("Calendar",    "Standup in 5 minutes."),
        ("Delivery",    "Your package was delivered to the front desk."),
        ("Reminder",    "Take medication at 8pm."),
        ("Weather",     "Heavy rain expected this evening."),
        ("Bill due",    "Electric bill due in 3 days."),
        ("Workout",     "Time for your evening run."),
        ("Flight",      "Flight UA-441 boarding in 30 minutes."),
        ("Backup",      "Nightly backup completed successfully."),
    ]
    for i, (title, body) in enumerate(push_specs):
        samples.append({
            "tool": "send_push_notification",
            "args": {"device": f"device-{i%3}", "title": title, "body": body},
            "user": f"Push me a notification: {title} — {body}",
            "result": '{"delivered":true}',
        })

    for i in range(8):
        samples.append({
            "tool": "schedule_email",
            "args": {"to": EMAILS[i % len(EMAILS)],
                     "subject": EXTRA_SUBJECTS_BODIES[i % len(EXTRA_SUBJECTS_BODIES)][0],
                     "body": EXTRA_SUBJECTS_BODIES[i % len(EXTRA_SUBJECTS_BODIES)][1],
                     "send_at": f"2026-05-{10+i:02d}T09:00:00Z"},
            "user": f"Schedule that email for May {10+i} at 9am.",
            "result": f'{{"scheduled":true,"id":"sch_{i+200}"}}',
        })

    # ---- Tag, move, snooze ----
    tags = ["follow-up", "tax-2025", "urgent", "personal", "investments", "kids", "travel", "receipts"]
    for i, t in enumerate(tags):
        samples.append({
            "tool": "tag_email",
            "args": {"email_id": f"msg_t{i:03d}", "tag": t},
            "user": f"Tag msg_t{i:03d} as {t}.",
            "result": '{"tagged":true}',
        })

    folders_for_move = ["Receipts", "Receipts 2026", "Travel", "Tax 2025", "Personal", "Job applications", "Archive", "Important"]
    for i, fl in enumerate(folders_for_move):
        samples.append({
            "tool": "move_to_folder",
            "args": {"email_id": f"msg_m{i:03d}", "folder": fl},
            "user": f"Move msg_m{i:03d} to {fl}.",
            "result": '{"moved":true}',
        })

    for i in range(8):
        samples.append({
            "tool": "snooze_email",
            "args": {"email_id": f"msg_s{i:03d}", "until": f"2026-05-{15+i:02d}T08:00:00Z"},
            "user": f"Snooze msg_s{i:03d} until May {15+i} at 8am.",
            "result": f'{{"snoozed_until":"2026-05-{15+i:02d}T08:00:00Z"}}',
        })

    # ---- Contacts ----
    for i in range(15):
        n = NAMES[i]
        em = EMAILS[i % len(EMAILS)]
        samples.append({
            "tool": "add_contact",
            "args": {"name": n, "email": em},
            "user": f"Add {n} ({em}) to my contacts.",
            "result": f'{{"id":"ctc_n{i:03d}","saved":true}}',
        })
    for i in range(10):
        cid = f"ctc_u{i:03d}"
        field = ["phone", "email", "title", "company", "notes"][i % 5]
        val = ["+1-555-0001", "new@email.com", "Director", "NewCo", "VIP"][i % 5]
        samples.append({
            "tool": "update_contact",
            "args": {"contact_id": cid, "field": field, "value": val},
            "user": f"Update {cid}'s {field} to {val}.",
            "result": '{"updated":true}',
        })
    for i in range(8):
        em = EMAILS[i % len(EMAILS)]
        samples.append({
            "tool": "find_contact_by_email",
            "args": {"email": em},
            "user": f"Find the contact for {em}.",
            "result": f'{{"id":"ctc_{i+400:03d}","name":"{NAMES[i % len(NAMES)]}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "get_contact_details",
            "args": {"contact_id": f"ctc_d{i:03d}"},
            "user": f"Pull contact ctc_d{i:03d}.",
            "result": f'{{"name":"{NAMES[i]}","email":"{EMAILS[i % len(EMAILS)]}","phone":"{PHONES[i % len(PHONES)]}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "list_contacts",
            "args": {"limit": 5 + i * 3},
            "user": f"List {5+i*3} contacts.",
            "result": '{"contacts":[{"id":"ctc_1","name":"Sarah Chen"},{"id":"ctc_2","name":"Marcus Webb"}],"total":284}',
        })

    # ---- Chat ops ----
    for i in range(8):
        samples.append({
            "tool": "create_chat_channel",
            "args": {"name": f"#topic-{i}", "private": i % 2 == 0,
                     "purpose": f"Discussion of topic {i}"},
            "user": f"Create a {'private' if i%2==0 else 'public'} channel called topic-{i}.",
            "result": f'{{"channel_id":"C{i:04d}","created":true}}',
        })
    for i in range(8):
        samples.append({
            "tool": "add_user_to_channel",
            "args": {"channel": CHANNELS[i % len(CHANNELS)],
                     "user": f"@{NAMES[i % len(NAMES)].split()[0].lower()}"},
            "user": f"Add {NAMES[i % len(NAMES)].split()[0]} to {CHANNELS[i % len(CHANNELS)]}.",
            "result": '{"added":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "leave_channel",
            "args": {"channel": CHANNELS[i % len(CHANNELS)]},
            "user": f"Leave {CHANNELS[i % len(CHANNELS)]}.",
            "result": '{"left":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "mute_channel",
            "args": {"channel": CHANNELS[(i+3) % len(CHANNELS)], "duration_hours": 1 + i * 2},
            "user": f"Mute {CHANNELS[(i+3) % len(CHANNELS)]} for {1+i*2} hours.",
            "result": '{"muted":true}',
        })
    for i in range(6):
        samples.append({
            "tool": "pin_message",
            "args": {"channel": CHANNELS[i % len(CHANNELS)],
                     "message_id": f"c_{1000+i*7}"},
            "user": f"Pin c_{1000+i*7} in {CHANNELS[i % len(CHANNELS)]}.",
            "result": '{"pinned":true}',
        })
    for i in range(6):
        emoji = ["thumbsup", "tada", "heart", "rocket", "eyes", "fire"][i]
        samples.append({
            "tool": "react_to_message",
            "args": {"message_id": f"c_{2000+i*7}", "emoji": emoji},
            "user": f"React {emoji} to c_{2000+i*7}.",
            "result": '{"reacted":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "edit_chat_message",
            "args": {"message_id": f"c_e{i:03d}", "text": "(corrected) " + EXTRA_BODIES[i]},
            "user": f"Edit c_e{i:03d} to: (corrected) {EXTRA_BODIES[i]}",
            "result": '{"edited":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "delete_chat_message",
            "args": {"message_id": f"c_d{i:03d}"},
            "user": f"Delete c_d{i:03d}.",
            "result": '{"deleted":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "get_chat_history",
            "args": {"channel": CHANNELS[i % len(CHANNELS)], "limit": 5 + i},
            "user": f"Show last {5+i} messages in {CHANNELS[i % len(CHANNELS)]}.",
            "result": '{"messages":[{"user":"@hana","text":"PR ready"},{"user":"@felix","text":"merging"}]}',
        })
    for i in range(5):
        samples.append({
            "tool": "search_chat_messages",
            "args": {"query": ["deploy", "incident", "outage", "RFC", "kickoff"][i],
                     "channel": CHANNELS[i % len(CHANNELS)]},
            "user": f"Search {CHANNELS[i % len(CHANNELS)]} for '{['deploy', 'incident', 'outage', 'RFC', 'kickoff'][i]}'.",
            "result": '{"results":[{"id":"c_881"}],"total":1}',
        })
    for i in range(4):
        samples.append({
            "tool": "get_unread_chat_count",
            "args": {"channel": CHANNELS[i % len(CHANNELS)]} if i < 3 else {},
            "user": "Unread chat count?" if i == 3 else f"Unread in {CHANNELS[i % len(CHANNELS)]}?",
            "result": f'{{"count":{i*5+2}}}',
        })

    # ---- Forwards & replies ----
    for i in range(8):
        samples.append({
            "tool": "forward_email",
            "args": {"email_id": f"msg_f{i:03d}", "to": EMAILS[i % len(EMAILS)]},
            "user": f"Forward msg_f{i:03d} to {EMAILS[i % len(EMAILS)]}.",
            "result": f'{{"forwarded":true,"new_id":"msg_out_f{i:03d}"}}',
        })
    for i in range(8):
        body = EXTRA_BODIES[i % len(EXTRA_BODIES)]
        samples.append({
            "tool": "reply_email",
            "args": {"email_id": f"msg_r{i:03d}", "body": body},
            "user": f"Reply to msg_r{i:03d}: {body}",
            "result": '{"sent":true}',
        })

    # ---- Block, unsubscribe, OOO, DND, status, signature, folder, group, dl ----
    spam_addrs = ["spam1@junk.biz", "marketing@cheap.co", "promo@ads.net", "noreply@bot.io",
                  "deals@retailer.com", "newsletter@unread.com"]
    for i, a in enumerate(spam_addrs):
        samples.append({
            "tool": "block_sender",
            "args": {"address": a},
            "user": f"Block {a}.",
            "result": '{"blocked":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "unsubscribe_from_list",
            "args": {"list_id": f"list-{i}"},
            "user": f"Unsubscribe me from list-{i}.",
            "result": '{"unsubscribed":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_out_of_office",
            "args": {"start": f"2026-0{6+i}-01", "end": f"2026-0{6+i}-05",
                     "message": f"Out of office {6+i}/1 to {6+i}/5."},
            "user": f"Set OOO {6+i}/1 through {6+i}/5.",
            "result": '{"set":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_dnd_status",
            "args": {"duration_minutes": 30 * (i + 1)},
            "user": f"DND for {30*(i+1)} minutes.",
            "result": f'{{"dnd_until":"2026-05-08T1{i}:00Z"}}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_chat_status",
            "args": {"emoji": [":coffee:", ":palm_tree:", ":computer:", ":sleeping:"][i],
                     "text": ["Coffee", "Vacation", "Heads down", "Sleeping"][i],
                     "expires_in": [600, 86400, 7200, 28800][i]},
            "user": f"Set my status to {['Coffee', 'Vacation', 'Heads down', 'Sleeping'][i]}.",
            "result": '{"set":true}',
        })
    for i in range(4):
        sig = ["— L", "Best,\nLakshman", "Thanks,\nL", "Cheers,\nLakshman"][i]
        samples.append({
            "tool": "set_email_signature",
            "args": {"signature": sig},
            "user": f"Set signature to: {sig}",
            "result": '{"updated":true}',
        })
    for i in range(4):
        n = ["Receipts 2026", "Tax 2025", "Travel", "Job apps"][i]
        samples.append({
            "tool": "create_email_folder",
            "args": {"name": n},
            "user": f"Create email folder {n}.",
            "result": f'{{"created":true,"folder_id":"f_{i+50}"}}',
        })
    for i in range(3):
        samples.append({
            "tool": "list_email_folders",
            "args": {},
            "user": ["List my folders.", "Show all email folders.", "What folders do I have?"][i],
            "result": '{"folders":["Inbox","Sent","Drafts","Archive","Receipts","Promotions"]}',
        })
    for i in range(4):
        samples.append({
            "tool": "create_email_template",
            "args": {"name": ["Polite decline", "Status update", "Meeting recap", "Intro"][i],
                     "subject": "Re: {topic}", "body": "Body for template " + str(i)},
            "user": f"Make a template called {['Polite decline', 'Status update', 'Meeting recap', 'Intro'][i]}.",
            "result": f'{{"id":"tpl_{i+50}"}}',
        })
    for i in range(4):
        n = ["Family", "Investors", "Book club", "Soccer team"][i]
        samples.append({
            "tool": "create_contact_group",
            "args": {"name": n, "members": [f"ctc_{j}" for j in range(i*3, i*3+3)]},
            "user": f"Make a {n} contact group with three people.",
            "result": f'{{"group_id":"grp_{i+30}"}}',
        })
    for i in range(4):
        action = "add" if i % 2 == 0 else "remove"
        em = EMAILS[i % len(EMAILS)]
        samples.append({
            "tool": "update_distribution_list",
            "args": {"list_id": f"dl_{['eng','design','sales','ops'][i]}", "action": action, "email": em},
            "user": f"{action.capitalize()} {em} {'to' if action=='add' else 'from'} dl_{['eng','design','sales','ops'][i]}.",
            "result": f'{{"members":{40 + i}}}',
        })

    # ---- Calendar invite & attachments ----
    for i in range(6):
        samples.append({
            "tool": "send_calendar_invite_email",
            "args": {"to": [EMAILS[i % len(EMAILS)]],
                     "title": ["1:1", "Project sync", "Design review", "Standup", "Retro", "Demo"][i],
                     "when": f"2026-05-{14+i:02d}T15:00:00Z",
                     "duration_minutes": [30, 45, 60, 15, 60, 30][i]},
            "user": f"Send a calendar invite for a {['1:1', 'project sync', 'design review', 'standup', 'retro', 'demo'][i]} on May {14+i}.",
            "result": f'{{"sent":true,"event_id":"ev_{i+700}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "send_email_with_attachment",
            "args": {"to": EMAILS[i % len(EMAILS)],
                     "subject": ["Receipt", "Contract", "Slides", "Photos", "Report", "Invoice"][i],
                     "body": "Attached.",
                     "attachment_path": f"/files/{['receipt','contract','slides','photos','report','invoice'][i]}_{i}.pdf"},
            "user": f"Email {EMAILS[i % len(EMAILS)]} the {['receipt', 'contract', 'slides', 'photos', 'report', 'invoice'][i]} attachment.",
            "result": f'{{"sent":true,"id":"msg_att_{i:03d}"}}',
        })

    # ---- Misc: merge, remove, get_thread, get_email_thread, group_sms, voice_msg, block_phone, typing ----
    for i in range(4):
        samples.append({
            "tool": "merge_contacts",
            "args": {"primary_id": f"ctc_p{i}", "duplicate_id": f"ctc_d{i}"},
            "user": f"Merge ctc_d{i} into ctc_p{i}.",
            "result": f'{{"merged":true,"surviving_id":"ctc_p{i}"}}',
        })
    for i in range(4):
        samples.append({
            "tool": "remove_contact",
            "args": {"contact_id": f"ctc_rm{i:03d}"},
            "user": f"Remove ctc_rm{i:03d}.",
            "result": '{"removed":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "get_email_thread",
            "args": {"thread_id": f"thr_{500+i*7}"},
            "user": f"Pull thread thr_{500+i*7}.",
            "result": f'{{"messages":{3+i},"participants":["me","{EMAILS[i % len(EMAILS)]}"]}}',
        })
    for i in range(3):
        samples.append({
            "tool": "get_thread_participants",
            "args": {"thread_id": f"chat_thr_{i:02d}"},
            "user": f"Who's in chat thread {i}?",
            "result": '{"participants":["@hana","@felix","@me"]}',
        })
    for i in range(5):
        recips = [NAMES[(i*3+j) % len(NAMES)].split()[0] for j in range(3)]
        samples.append({
            "tool": "send_group_sms",
            "args": {"recipients": recips, "body": EXTRA_BODIES[i]},
            "user": f"Group text {', '.join(recips)}: {EXTRA_BODIES[i]}",
            "result": '{"sent":3,"failed":0}',
        })
    for i in range(5):
        contact = NAMES[(i*4) % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_voice_message",
            "args": {"contact": contact, "audio_url": f"rec://vm_x{i:03d}", "duration_sec": 15 + i * 8},
            "user": f"Send {contact} the voice memo I just recorded.",
            "result": '{"sent":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "block_phone_number",
            "args": {"number": PHONES[i % len(PHONES)]},
            "user": f"Block {PHONES[i % len(PHONES)]}.",
            "result": '{"blocked":true}',
        })
    for i in range(3):
        samples.append({
            "tool": "send_typing_indicator",
            "args": {"channel": [CHANNELS[0], "@elena", "@diego"][i]},
            "user": f"Show I'm typing in {[CHANNELS[0], '@elena', '@diego'][i]}.",
            "result": '{"sent":true}',
        })

    return samples


def build_messages(spec, sys_prompt: str, suffix: str) -> dict:
    """Construct ShareGPT messages list from a sample spec."""
    user_msg = {"role": "user", "content": spec["user"]}
    assistant_call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": spec["tool"],
                "arguments": spec["args"],
            },
        }],
    }
    msgs = [{"role": "system", "content": sys_prompt}, user_msg, assistant_call]
    if spec["result"] is not None:
        msgs.append({"role": "tool", "name": spec["tool"], "content": spec["result"]})
        # Compose a short summary that opens with the suffix-pool phrase.
        # Append a tool-specific micro-detail when possible.
        summary = suffix
        msgs.append({"role": "assistant", "content": summary})
    return {"messages": msgs}


def assign_suffixes(specs):
    """Round-robin assign suffix-pool phrases to multi-turn samples."""
    multi_indices = [i for i, s in enumerate(specs) if s["result"] is not None]
    suffixes = [None] * len(specs)
    for k, idx in enumerate(multi_indices):
        suffixes[idx] = SUFFIX_POOL[k % len(SUFFIX_POOL)]
    return suffixes


def trim_to_500(specs):
    """Deterministically pick exactly 500 samples preserving the
    ~75 single-turn / ~425 multi-turn split."""
    rng = random.Random(_stable_int(SEED) % (2**32))
    singles = [s for s in specs if s["result"] is None]
    multis = [s for s in specs if s["result"] is not None]

    rng.shuffle(singles)
    rng.shuffle(multis)

    target_single = 75
    target_multi = 500 - target_single

    # If we don't have enough of either, recompute target proportionally.
    if len(singles) < target_single:
        target_single = len(singles)
        target_multi = 500 - target_single
    if len(multis) < target_multi:
        target_multi = len(multis)
        target_single = 500 - target_multi

    chosen = singles[:target_single] + multis[:target_multi]
    rng.shuffle(chosen)
    return chosen


def main():
    specs = expand_samples()
    chosen = trim_to_500(specs)
    suffixes = assign_suffixes(chosen)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(_stable_int(SEED + "sys") % (2**32))

    written = 0
    with OUT_PATH.open("w") as f:
        for spec, sfx in zip(chosen, suffixes):
            sys_prompt = rng.choice(SYSTEM_PROMPTS)
            sample = build_messages(spec, sys_prompt, sfx if sfx else "")
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            written += 1

    # Stats
    tools = [s["tool"] for s in chosen]
    distinct = len(set(tools))
    single_count = sum(1 for s in chosen if s["result"] is None)
    multi_count = len(chosen) - single_count
    suffix_used = set(sfx for sfx in suffixes if sfx)

    # Per-tool occurrence cap check
    from collections import Counter
    cnt = Counter(tools)
    over_cap = [(t, c) for t, c in cnt.items() if c > 25]

    print(f"wrote {written} samples to {OUT_PATH}")
    print(f"distinct tools: {distinct}")
    print(f"single-turn: {single_count}")
    print(f"multi-turn: {multi_count}")
    print(f"suffix-pool coverage: {len(suffix_used)}/30")
    if over_cap:
        print(f"WARNING tools over 25 occurrences: {over_cap}")
    most_common = cnt.most_common(5)
    print(f"top 5 tools: {most_common}")


if __name__ == "__main__":
    main()
