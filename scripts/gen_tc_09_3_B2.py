#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""gen_tc_09_3_B2.py -- Generate 500 ShareGPT tool-calling samples (TC-B2: messaging, wave 2).

Wave 2 fresh batch in the messaging/email/SMS/chat/contacts domain. Reuses the
same parametric expansion pattern as gen_tc_09_3_B.py but with a different seed
and additional content variations to reduce overlap with batch-07-B.jsonl.
Curation handles any residual duplicates.

Seed: 1009308B (deterministic shuffle).

Output: datasets/tool-calling/raw-09.3/batch-08-B.jsonl (exactly 500 lines).
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from gen_tc_09_3_B_data import (
    NAMES, EMAILS, PHONES, CHANNELS, SUBJECTS,
    SAMPLES as BASE_SAMPLES, SUFFIX_POOL, TOOLS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "datasets" / "tool-calling" / "raw-09.3" / "batch-08-B.jsonl"
SEED = "1009308B"

SYSTEM_PROMPTS = [
    "You are a helpful assistant. Prefer calling tools over guessing.",
    "You are a productivity assistant with access to messaging, email, and contact tools. Call tools when the user's request maps to one.",
    "You are an assistant that helps the user manage communications. Use tools to take actions; respond concisely after tool results.",
    "You are a helpful assistant. When the user requests an action you have a tool for, call the tool with realistic arguments.",
    "You are a communications assistant. Invoke the appropriate tool for any actionable request.",
    "You are a helpful messaging assistant; lean on tools rather than guessing values.",
]


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


# Wave-2 specific content pools (different bodies, subjects, topics).

EXTRA_BODIES = [
    "Heads up — I'll be 10 minutes late to the sync.",
    "Just landed; grabbing a cab now.",
    "Pushed the fix, please re-run the pipeline.",
    "Got the invoice — paid, confirmation attached.",
    "Locked the meeting room from 2-3 today.",
    "Quick favor — can you forward the slide deck?",
    "Tasting menu booked for 7pm Saturday.",
    "Order shipped, tracking number incoming.",
    "Reminder: signoff needed before end of day.",
    "Walked the dog already, you're off the hook.",
    "Server replaced, monitoring overnight.",
    "Dropping the kids off, back online by 10.",
    "Shipping the demo build to QA tonight.",
    "Need an extra pair of eyes on the proposal.",
    "Picked the wine, leaving the cheese to you.",
    "Postmortem doc is open for comments.",
    "Booked the off-site venue, deposit paid.",
    "Finished the brief — ready when you are.",
    "Quick sanity check before I send this off.",
    "Calling it a night, talk tomorrow.",
]

EXTRA_SUBJECTS_BODIES = [
    ("Budget v2 review",        "Updated FY26 budget attached — please flag any push-back by Friday."),
    ("Standup time change",     "Moving daily standup to 9:45am starting Monday."),
    ("New laptop request",      "Filing the request for a 16-inch — let me know if I should add anyone."),
    ("Vendor short-list",       "Three options after the call; my picks ranked in the doc."),
    ("Customer kickoff agenda", "Draft agenda for the Acme kickoff — input welcome before Tuesday."),
    ("Architecture decision",   "Wrote up an ADR for the auth migration; review when you get a sec."),
    ("Sprint retro notes",      "Action items captured at the bottom of the doc."),
    ("Hiring bar discussion",   "Want to align on bar for the staff role before Friday's panel."),
    ("Speaker confirmation",    "Confirmed for the keynote slot. Travel details to follow."),
    ("Pricing update",          "New price sheet effective June 1; field teams notified."),
    ("OKR check-in",            "Q2 OKR check-in is overdue — please update by Wednesday."),
    ("Outage notification",     "Brief incident on payment gateway, mitigated. Postmortem on the way."),
    ("Welcome back",            "Hope you had a good break — quick sync this afternoon?"),
    ("Patent filing",           "Counsel has a draft of the application; review by Thursday EOD."),
    ("Board update draft",      "First pass at the board update — comments appreciated."),
    ("Press inquiry",           "Reporter at TechCrunch reached out — looping in PR."),
    ("Annual offsite",          "Picking dates for the offsite; please vote by Sunday."),
    ("Customer escalation",     "Globex hit a P1 last night; on a war-room call now."),
    ("Conference travel",       "Submitting expenses for last week's conference today."),
    ("Mentorship pairing",      "Matched mentors and mentees — see the assignment table."),
]


def expand_samples():
    samples = list(BASE_SAMPLES)

    # Email expansion with wave-2 subjects/bodies
    for i, (subj, body) in enumerate(EXTRA_SUBJECTS_BODIES):
        recipient_email = EMAILS[(i + 5) % len(EMAILS)]
        recipient_name = NAMES[(i + 5) % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_email",
            "args": {"to": recipient_email, "subject": subj, "body": body},
            "user": f"Send {recipient_name} an email about {subj.lower()}: {body[:60]}",
            "result": f'{{"sent":true,"id":"msg_w2e{i:03d}"}}',
        })

    # SMS expansion
    for i, body in enumerate(EXTRA_BODIES):
        contact = NAMES[(i * 7 + 1) % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_sms",
            "args": {"contact": contact, "body": body},
            "user": f"Shoot {contact} a text: {body}",
            "result": f'{{"sent":true,"sid":"sms_w2_{i:03d}"}}',
        })

    # Chat messages
    chat_msgs = [
        ("Reverted the bad commit, all green now.",     "#engineering"),
        ("New sprint starts Monday — backlog groomed.", "#scrum"),
        ("Pricing review meeting at 4pm today.",        "#finance"),
        ("Reviewer assignments posted.",                "#code-review"),
        ("Friendly reminder: timesheets due Friday.",   "#operations"),
        ("Logs look clean post-deploy.",                "#sre"),
        ("Customer demo went well — three follow-ups.", "#sales"),
        ("Brand assets refreshed in the drive.",        "#marketing"),
        ("New hire starts Monday — please welcome them.", "#general"),
        ("Working group meets Thursday at 11.",         "#platform"),
        ("Migration script ready for review.",          "#data-eng"),
        ("Bug report triaged — assigned out.",          "#qa"),
        ("Office WiFi maintenance tonight.",            "#it-help"),
        ("Pizza Friday at noon.",                       "#kitchen-fridge"),
        ("Status page updated for the recent incident.", "#alerts-prod"),
        ("Roadmap doc shared — comment by EOW.",        "#product"),
        ("Library upgrade landed in main.",             "#frontend"),
        ("New API contract published.",                 "#backend-platform"),
        ("Security scan results posted.",               "#security-incidents"),
        ("Friday demo lineup confirmed.",               "#general"),
    ]
    for i, (text, ch) in enumerate(chat_msgs):
        samples.append({
            "tool": "send_chat_message",
            "args": {"channel": ch, "text": text},
            "user": f"Drop a message in {ch}: {text}",
            "result": f'{{"sent":true,"ts":"2026-05-09T{(i%10):02d}:30:00Z"}}',
        })

    # search_emails varied queries
    queries = [
        ("from:cfo subject:budget",     '{"results":[{"id":"msg_b1"},{"id":"msg_b2"}],"total":2}'),
        ("subject:offsite",             '{"results":[{"id":"msg_o1"}],"total":1}'),
        ("after:2026-05-01 has:attachment", '{"results":[{"id":"msg_a4"},{"id":"msg_a5"}],"total":2}'),
        ("is:unread is:important",      '{"results":[{"id":"msg_imp1"}],"total":1}'),
        ("from:counsel@firm.com",       '{"results":[{"id":"msg_c1"},{"id":"msg_c2"}],"total":2}'),
        ("keyword:renewal",             '{"results":[{"id":"msg_rn1"}],"total":1}'),
        ("subject:offer letter",        '{"results":[{"id":"msg_off1"}],"total":1}'),
        ("from:billing@stripe.com",     '{"results":[{"id":"msg_st1"},{"id":"msg_st2"},{"id":"msg_st3"}],"total":3}'),
        ("from:diego subject:auth",     '{"results":[{"id":"msg_da1"}],"total":1}'),
        ("subject:retro",               '{"results":[{"id":"msg_rt1"},{"id":"msg_rt2"}],"total":2}'),
        ("expense report",              '{"results":[],"total":0}'),
        ("from:partners@bigco.com",     '{"results":[{"id":"msg_p1"}],"total":1}'),
        ("subject:patent",              '{"results":[{"id":"msg_pat1"}],"total":1}'),
        ("subject:welcome back",        '{"results":[{"id":"msg_wb1"}],"total":1}'),
        ("subject:NDA",                 '{"results":[{"id":"msg_nda1"},{"id":"msg_nda2"}],"total":2}'),
    ]
    user_phrasings = [
        "Hunt through email for {q}.",
        "Mail search: {q}.",
        "Comb my inbox for {q}.",
        "Find anything matching {q} in mail.",
        "Look up emails: {q}.",
    ]
    for i, (q, res) in enumerate(queries):
        samples.append({
            "tool": "search_emails",
            "args": {"query": q},
            "user": user_phrasings[i % len(user_phrasings)].format(q=q),
            "result": res,
        })

    # count_unread variants
    folder_phrasings = [
        ("Inbox",     "Give me my Inbox unread count."),
        ("Drafts",    "How many drafts unread?"),
        ("Sent",      "Unread items in Sent?"),
        ("Spam",      "Spam unread total?"),
        ("Archive",   "Archive unread?"),
        ("Important", "How many unread Important emails?"),
        ("Receipts",  "Unread in Receipts folder?"),
        ("Updates",   "Updates folder unread tally?"),
        ("Forums",    "Forums unread total?"),
        ("Promotions","Unread Promotions?"),
    ]
    for i, (folder, user) in enumerate(folder_phrasings):
        samples.append({
            "tool": "count_unread_emails",
            "args": {"folder": folder},
            "user": user,
            "result": f'{{"count":{(i*11+5)%60}}}',
        })

    # Mark/archive/delete (use different id ranges than wave 1)
    for i in range(15):
        eid = f"msg_{4000+i*23}"
        samples.append({
            "tool": "mark_email_read",
            "args": {"email_id": eid},
            "user": f"Flag {eid} as read.",
            "result": '{"marked":true}',
        })
    for i in range(15):
        eid = f"msg_{5000+i*29}"
        samples.append({
            "tool": "archive_email",
            "args": {"email_id": eid},
            "user": f"Send {eid} to archive.",
            "result": '{"archived":true}',
        })
    for i in range(12):
        eid = f"msg_{6000+i*31}"
        samples.append({
            "tool": "delete_email",
            "args": {"email_id": eid},
            "user": f"Trash {eid}.",
            "result": '{"deleted":true}',
        })

    # Voice / call expansions
    for i in range(12):
        contact = NAMES[(i * 11 + 2) % len(NAMES)].split()[0]
        samples.append({
            "tool": "start_voice_call",
            "args": {"contact": contact},
            "user": f"Dial {contact} for me.",
            "result": f'{{"call_id":"call_w2_{i:03d}","status":"ringing"}}',
        })

    for i in range(10):
        vid = f"vm_{7000+i*19}"
        samples.append({
            "tool": "transcribe_voicemail",
            "args": {"voicemail_id": vid},
            "user": f"Transcribe voicemail {vid}.",
            "result": f'{{"transcript":"Hi, this is {NAMES[(i+3)%len(NAMES)].split()[0]} — just checking in, give me a ring when you can."}}',
        })

    for i in range(8):
        cid = f"call_{700+i*23}"
        samples.append({
            "tool": "transcribe_call",
            "args": {"call_id": cid},
            "user": f"Pull a transcript for {cid}.",
            "result": f'{{"transcript":"Reviewed the contract terms, agreed on minor edits, scheduled follow-up.","duration_sec":{200+i*25}}}',
        })

    for i in range(8):
        samples.append({
            "tool": "get_call_log",
            "args": {"limit": 5 + i},
            "user": f"Pull the last {5+i} calls from my log.",
            "result": '{"calls":[{"id":"call_x","with":"Diego","duration_sec":420},{"id":"call_y","with":"+1-555-0177","duration_sec":89}]}',
        })

    for i in range(6):
        vid = f"vm_{9500+i*13}"
        samples.append({
            "tool": "delete_voicemail",
            "args": {"voicemail_id": vid},
            "user": f"Wipe voicemail {vid}.",
            "result": '{"deleted":true}',
        })

    for i in range(6):
        samples.append({
            "tool": "get_voicemail_list",
            "args": {"limit": 4 + i},
            "user": f"Show me the {4+i} most recent voicemails.",
            "result": '{"voicemails":[{"id":"vm_8901","from":"Elena","unread":true},{"id":"vm_8902","from":"Felix","unread":true}]}',
        })

    push_specs = [
        ("Deploy live", "Production deploy v2.4.1 is now live."),
        ("Meeting",     "Sprint planning starts in 10 minutes."),
        ("Order",       "Your grocery order is out for delivery."),
        ("Reminder",    "Don't forget to take out the trash tonight."),
        ("Stocks",      "AAPL down 2% today."),
        ("Workout",     "Yoga class at 6:30 starts in 30 minutes."),
        ("Travel",      "Check-in opens for your flight in 1 hour."),
        ("Backup",      "Cloud backup completed."),
        ("Storm",       "Severe thunderstorm warning in your area."),
        ("Weekly",      "Your weekly screen time report is ready."),
    ]
    for i, (title, body) in enumerate(push_specs):
        samples.append({
            "tool": "send_push_notification",
            "args": {"device": f"device-{(i+1)%3}", "title": title, "body": body},
            "user": f"Push notify: {title} — {body}",
            "result": '{"delivered":true}',
        })

    for i in range(8):
        samples.append({
            "tool": "schedule_email",
            "args": {"to": EMAILS[(i + 4) % len(EMAILS)],
                     "subject": EXTRA_SUBJECTS_BODIES[(i + 2) % len(EXTRA_SUBJECTS_BODIES)][0],
                     "body": EXTRA_SUBJECTS_BODIES[(i + 2) % len(EXTRA_SUBJECTS_BODIES)][1],
                     "send_at": f"2026-05-{18+i:02d}T08:30:00Z"},
            "user": f"Queue this email up for May {18+i} at 8:30am.",
            "result": f'{{"scheduled":true,"id":"sch_w2_{i+300}"}}',
        })

    tags = ["q3-budget", "kickoff-2026", "to-review", "personal-finance", "house", "school", "expenses", "retainer"]
    for i, t in enumerate(tags):
        samples.append({
            "tool": "tag_email",
            "args": {"email_id": f"msg_tw2_{i:03d}", "tag": t},
            "user": f"Tag msg_tw2_{i:03d} with '{t}'.",
            "result": '{"tagged":true}',
        })

    folders_for_move = ["Receipts 2026 Q2", "Travel 2026", "Tax 2026", "Family", "Volunteering", "Side Project", "Archive 2025", "Important Q2"]
    for i, fl in enumerate(folders_for_move):
        samples.append({
            "tool": "move_to_folder",
            "args": {"email_id": f"msg_mw2_{i:03d}", "folder": fl},
            "user": f"Shift msg_mw2_{i:03d} into {fl}.",
            "result": '{"moved":true}',
        })

    for i in range(8):
        samples.append({
            "tool": "snooze_email",
            "args": {"email_id": f"msg_sw2_{i:03d}", "until": f"2026-05-{20+i:02d}T07:30:00Z"},
            "user": f"Snooze msg_sw2_{i:03d} til May {20+i} at 7:30am.",
            "result": f'{{"snoozed_until":"2026-05-{20+i:02d}T07:30:00Z"}}',
        })

    # Contacts
    for i in range(15):
        n = NAMES[(i + 7) % len(NAMES)]
        em = EMAILS[(i + 2) % len(EMAILS)]
        samples.append({
            "tool": "add_contact",
            "args": {"name": n, "email": em},
            "user": f"Save {n} ({em}) into contacts.",
            "result": f'{{"id":"ctc_w2_{i:03d}","saved":true}}',
        })
    for i in range(10):
        cid = f"ctc_uw2_{i:03d}"
        field = ["phone", "email", "title", "company", "notes"][i % 5]
        val = ["+1-555-9001", "fresh@email.com", "VP", "InnoCorp", "Met at conf"][i % 5]
        samples.append({
            "tool": "update_contact",
            "args": {"contact_id": cid, "field": field, "value": val},
            "user": f"Change {field} on {cid} to {val}.",
            "result": '{"updated":true}',
        })
    for i in range(8):
        em = EMAILS[(i + 3) % len(EMAILS)]
        samples.append({
            "tool": "find_contact_by_email",
            "args": {"email": em},
            "user": f"Look up the contact behind {em}.",
            "result": f'{{"id":"ctc_w2f_{i+500:03d}","name":"{NAMES[(i+1) % len(NAMES)]}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "get_contact_details",
            "args": {"contact_id": f"ctc_dw2_{i:03d}"},
            "user": f"Show me ctc_dw2_{i:03d}.",
            "result": f'{{"name":"{NAMES[(i+2) % len(NAMES)]}","email":"{EMAILS[(i+1) % len(EMAILS)]}","phone":"{PHONES[(i+1) % len(PHONES)]}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "list_contacts",
            "args": {"limit": 6 + i * 4},
            "user": f"Give me {6+i*4} contacts.",
            "result": '{"contacts":[{"id":"ctc_3","name":"Diego Alvarez"},{"id":"ctc_4","name":"Hana Park"}],"total":312}',
        })

    # Chat ops
    for i in range(8):
        samples.append({
            "tool": "create_chat_channel",
            "args": {"name": f"#wave2-{i}", "private": i % 3 == 0,
                     "purpose": f"Wave 2 working channel {i}"},
            "user": f"Set up a {'private' if i%3==0 else 'public'} channel called wave2-{i}.",
            "result": f'{{"channel_id":"C2{i:04d}","created":true}}',
        })
    for i in range(8):
        samples.append({
            "tool": "add_user_to_channel",
            "args": {"channel": CHANNELS[(i + 2) % len(CHANNELS)],
                     "user": f"@{NAMES[(i+4) % len(NAMES)].split()[0].lower()}"},
            "user": f"Pull {NAMES[(i+4) % len(NAMES)].split()[0]} into {CHANNELS[(i+2) % len(CHANNELS)]}.",
            "result": '{"added":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "leave_channel",
            "args": {"channel": CHANNELS[(i + 4) % len(CHANNELS)]},
            "user": f"Drop me from {CHANNELS[(i+4) % len(CHANNELS)]}.",
            "result": '{"left":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "mute_channel",
            "args": {"channel": CHANNELS[(i + 1) % len(CHANNELS)], "duration_hours": 2 + i * 3},
            "user": f"Silence {CHANNELS[(i+1) % len(CHANNELS)]} for {2+i*3} hours.",
            "result": '{"muted":true}',
        })
    for i in range(6):
        samples.append({
            "tool": "pin_message",
            "args": {"channel": CHANNELS[(i + 3) % len(CHANNELS)],
                     "message_id": f"c_w2_{2000+i*9}"},
            "user": f"Pin c_w2_{2000+i*9} in {CHANNELS[(i+3) % len(CHANNELS)]}.",
            "result": '{"pinned":true}',
        })
    for i in range(6):
        emoji = ["clap", "white_check_mark", "100", "warning", "sparkles", "saluting_face"][i]
        samples.append({
            "tool": "react_to_message",
            "args": {"message_id": f"c_w2r_{i:03d}", "emoji": emoji},
            "user": f"Drop a {emoji} on c_w2r_{i:03d}.",
            "result": '{"reacted":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "edit_chat_message",
            "args": {"message_id": f"c_w2e_{i:03d}", "text": "(edit) " + EXTRA_BODIES[(i+5) % len(EXTRA_BODIES)]},
            "user": f"Update c_w2e_{i:03d} to: (edit) {EXTRA_BODIES[(i+5) % len(EXTRA_BODIES)]}",
            "result": '{"edited":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "delete_chat_message",
            "args": {"message_id": f"c_w2d_{i:03d}"},
            "user": f"Remove c_w2d_{i:03d}.",
            "result": '{"deleted":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "get_chat_history",
            "args": {"channel": CHANNELS[(i + 2) % len(CHANNELS)], "limit": 6 + i * 2},
            "user": f"Pull the last {6+i*2} chats in {CHANNELS[(i+2) % len(CHANNELS)]}.",
            "result": '{"messages":[{"user":"@elena","text":"shipping in 5"},{"user":"@diego","text":"on it"}]}',
        })
    for i in range(5):
        q = ["release", "rollback", "feature flag", "regression", "approval"][i]
        samples.append({
            "tool": "search_chat_messages",
            "args": {"query": q, "channel": CHANNELS[(i + 1) % len(CHANNELS)]},
            "user": f"Search {CHANNELS[(i+1) % len(CHANNELS)]} for '{q}'.",
            "result": '{"results":[{"id":"c_w2_991"}],"total":1}',
        })
    for i in range(4):
        samples.append({
            "tool": "get_unread_chat_count",
            "args": {"channel": CHANNELS[(i + 2) % len(CHANNELS)]} if i < 3 else {},
            "user": "Total unread chat count?" if i == 3 else f"Unread in {CHANNELS[(i+2) % len(CHANNELS)]}?",
            "result": f'{{"count":{i*4+1}}}',
        })

    for i in range(8):
        samples.append({
            "tool": "forward_email",
            "args": {"email_id": f"msg_fw2_{i:03d}", "to": EMAILS[(i+2) % len(EMAILS)]},
            "user": f"Forward msg_fw2_{i:03d} along to {EMAILS[(i+2) % len(EMAILS)]}.",
            "result": f'{{"forwarded":true,"new_id":"msg_outfw2_{i:03d}"}}',
        })
    for i in range(8):
        body = EXTRA_BODIES[(i + 3) % len(EXTRA_BODIES)]
        samples.append({
            "tool": "reply_email",
            "args": {"email_id": f"msg_rw2_{i:03d}", "body": body},
            "user": f"Send a reply to msg_rw2_{i:03d}: {body}",
            "result": '{"sent":true}',
        })

    spam_addrs = ["sale@spamco.io", "alerts@junkmail.biz", "deals@unwanted.org",
                  "info@coldoutreach.com", "winner@scam.io", "promo@discountland.net"]
    for i, a in enumerate(spam_addrs):
        samples.append({
            "tool": "block_sender",
            "args": {"address": a},
            "user": f"Add {a} to my blocklist.",
            "result": '{"blocked":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "unsubscribe_from_list",
            "args": {"list_id": f"ml-w2-{i}"},
            "user": f"Unsub from ml-w2-{i}.",
            "result": '{"unsubscribed":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_out_of_office",
            "args": {"start": f"2026-0{7+i}-10", "end": f"2026-0{7+i}-15",
                     "message": f"Away {7+i}/10 to {7+i}/15 — back online after."},
            "user": f"Turn on OOO {7+i}/10 through {7+i}/15.",
            "result": '{"set":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_dnd_status",
            "args": {"duration_minutes": 45 * (i + 1)},
            "user": f"Block notifications for {45*(i+1)} minutes.",
            "result": f'{{"dnd_until":"2026-05-09T{i+8:02d}:00Z"}}',
        })
    for i in range(4):
        samples.append({
            "tool": "set_chat_status",
            "args": {"emoji": [":thinking:", ":beach:", ":books:", ":hospital:"][i],
                     "text": ["Deep work", "PTO", "Studying", "Doctor"][i],
                     "expires_in": [3600, 86400, 5400, 7200][i]},
            "user": f"Mark me as {['Deep work', 'PTO', 'Studying', 'Doctor'][i]}.",
            "result": '{"set":true}',
        })
    for i in range(4):
        sig = ["~ Lakshman", "Regards,\nLakshman", "Kind regards,\nL.", "All the best,\nL"][i]
        samples.append({
            "tool": "set_email_signature",
            "args": {"signature": sig},
            "user": f"Update my signature to: {sig}",
            "result": '{"updated":true}',
        })
    for i in range(4):
        n = ["Q2 Receipts", "Tax 2026", "Travel 2026", "Volunteer"][i]
        samples.append({
            "tool": "create_email_folder",
            "args": {"name": n},
            "user": f"Spin up a {n} folder.",
            "result": f'{{"created":true,"folder_id":"f_w2_{i+80}"}}',
        })
    for i in range(3):
        samples.append({
            "tool": "list_email_folders",
            "args": {},
            "user": ["Show all my folders.", "What email folders exist?", "Pull up the folder list."][i],
            "result": '{"folders":["Inbox","Sent","Drafts","Archive","Important","Receipts","Travel"]}',
        })
    for i in range(4):
        samples.append({
            "tool": "create_email_template",
            "args": {"name": ["Follow-up", "Decline meeting", "Schedule chat", "Thank you"][i],
                     "subject": "Re: {topic}", "body": "Wave 2 template body " + str(i)},
            "user": f"Create an email template called {['Follow-up', 'Decline meeting', 'Schedule chat', 'Thank you'][i]}.",
            "result": f'{{"id":"tpl_w2_{i+80}"}}',
        })
    for i in range(4):
        n = ["College friends", "Neighbors", "Hiking group", "Hackathon team"][i]
        samples.append({
            "tool": "create_contact_group",
            "args": {"name": n, "members": [f"ctc_w2_{j}" for j in range(i*3+1, i*3+4)]},
            "user": f"Create a {n} group with three contacts.",
            "result": f'{{"group_id":"grp_w2_{i+50}"}}',
        })
    for i in range(4):
        action = "add" if i % 2 == 1 else "remove"
        em = EMAILS[(i + 1) % len(EMAILS)]
        samples.append({
            "tool": "update_distribution_list",
            "args": {"list_id": f"dl_{['leads','support','marketing','recruit'][i]}", "action": action, "email": em},
            "user": f"{action.capitalize()} {em} {'into' if action=='add' else 'out of'} dl_{['leads','support','marketing','recruit'][i]}.",
            "result": f'{{"members":{55 + i}}}',
        })

    for i in range(6):
        samples.append({
            "tool": "send_calendar_invite_email",
            "args": {"to": [EMAILS[(i + 1) % len(EMAILS)]],
                     "title": ["Coffee chat", "Architecture review", "Quarterly sync", "Lunch", "Postmortem", "Stakeholder review"][i],
                     "when": f"2026-05-{20+i:02d}T16:00:00Z",
                     "duration_minutes": [30, 60, 45, 60, 45, 60][i]},
            "user": f"Send a calendar invite for a {['coffee chat', 'architecture review', 'quarterly sync', 'lunch', 'postmortem', 'stakeholder review'][i]} on May {20+i}.",
            "result": f'{{"sent":true,"event_id":"ev_w2_{i+800}"}}',
        })
    for i in range(6):
        samples.append({
            "tool": "send_email_with_attachment",
            "args": {"to": EMAILS[(i + 2) % len(EMAILS)],
                     "subject": ["Statement", "Spec doc", "Photos", "Pitch deck", "Resume", "W2"][i],
                     "body": "Attaching the file.",
                     "attachment_path": f"/files/w2/{['statement','spec','photos','pitch','resume','w2'][i]}_{i}.pdf"},
            "user": f"Send {EMAILS[(i+2) % len(EMAILS)]} the {['statement', 'spec doc', 'photos', 'pitch deck', 'resume', 'W2'][i]} file.",
            "result": f'{{"sent":true,"id":"msg_attw2_{i:03d}"}}',
        })

    for i in range(4):
        samples.append({
            "tool": "merge_contacts",
            "args": {"primary_id": f"ctc_pw2_{i}", "duplicate_id": f"ctc_dw2_{i}"},
            "user": f"Merge ctc_dw2_{i} into ctc_pw2_{i}.",
            "result": f'{{"merged":true,"surviving_id":"ctc_pw2_{i}"}}',
        })
    for i in range(4):
        samples.append({
            "tool": "remove_contact",
            "args": {"contact_id": f"ctc_rmw2_{i:03d}"},
            "user": f"Delete contact ctc_rmw2_{i:03d}.",
            "result": '{"removed":true}',
        })
    for i in range(5):
        samples.append({
            "tool": "get_email_thread",
            "args": {"thread_id": f"thr_w2_{800+i*9}"},
            "user": f"Get thread thr_w2_{800+i*9}.",
            "result": f'{{"messages":{4+i},"participants":["me","{EMAILS[(i+2) % len(EMAILS)]}"]}}',
        })
    for i in range(3):
        samples.append({
            "tool": "get_thread_participants",
            "args": {"thread_id": f"chat_thr_w2_{i:02d}"},
            "user": f"Who's part of chat thread w2-{i}?",
            "result": '{"participants":["@diego","@elena","@me"]}',
        })
    for i in range(5):
        recips = [NAMES[(i*5+j+2) % len(NAMES)].split()[0] for j in range(3)]
        samples.append({
            "tool": "send_group_sms",
            "args": {"recipients": recips, "body": EXTRA_BODIES[(i + 8) % len(EXTRA_BODIES)]},
            "user": f"Group SMS to {', '.join(recips)}: {EXTRA_BODIES[(i + 8) % len(EXTRA_BODIES)]}",
            "result": '{"sent":3,"failed":0}',
        })
    for i in range(5):
        contact = NAMES[(i*6+1) % len(NAMES)].split()[0]
        samples.append({
            "tool": "send_voice_message",
            "args": {"contact": contact, "audio_url": f"rec://vmw2_{i:03d}", "duration_sec": 20 + i * 9},
            "user": f"Drop {contact} the voice clip I just made.",
            "result": '{"sent":true}',
        })
    for i in range(4):
        samples.append({
            "tool": "block_phone_number",
            "args": {"number": PHONES[(i + 2) % len(PHONES)]},
            "user": f"Blacklist {PHONES[(i+2) % len(PHONES)]}.",
            "result": '{"blocked":true}',
        })
    for i in range(3):
        samples.append({
            "tool": "send_typing_indicator",
            "args": {"channel": [CHANNELS[2], "@hana", "@marcus"][i]},
            "user": f"Send a typing indicator in {[CHANNELS[2], '@hana', '@marcus'][i]}.",
            "result": '{"sent":true}',
        })

    return samples


def build_messages(spec, sys_prompt: str, suffix: str) -> dict:
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
        msgs.append({"role": "assistant", "content": suffix})
    return {"messages": msgs}


def assign_suffixes(specs):
    multi_indices = [i for i, s in enumerate(specs) if s["result"] is not None]
    suffixes = [None] * len(specs)
    # rotate starting offset so wave-2 ordering differs from wave-1
    offset = _stable_int(SEED + "sfx") % len(SUFFIX_POOL)
    for k, idx in enumerate(multi_indices):
        suffixes[idx] = SUFFIX_POOL[(k + offset) % len(SUFFIX_POOL)]
    return suffixes


def trim_to_500(specs):
    rng = random.Random(_stable_int(SEED) % (2**32))
    singles = [s for s in specs if s["result"] is None]
    multis = [s for s in specs if s["result"] is not None]

    rng.shuffle(singles)
    rng.shuffle(multis)

    target_single = 75
    target_multi = 500 - target_single

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

    # Tool-cap enforcement: rebalance if any tool > 25 occurrences (5%).
    cnt = Counter(s["tool"] for s in chosen)
    over = {t: c for t, c in cnt.items() if c > 25}
    if over:
        # Drop overflow specs from chosen, top up from unused multis.
        unused = [s for s in specs if s not in chosen and s["result"] is not None]
        rng = random.Random(_stable_int(SEED + "rebal") % (2**32))
        rng.shuffle(unused)
        # Build per-tool buckets in `chosen`
        for tool, count in over.items():
            excess = count - 25
            # remove `excess` items of this tool, replace with under-cap unused tools
            removed = 0
            new_chosen = []
            for s in chosen:
                if s["tool"] == tool and removed < excess:
                    removed += 1
                    continue
                new_chosen.append(s)
            chosen = new_chosen
            current = Counter(s["tool"] for s in chosen)
            # find replacements whose tool is currently under cap
            i = 0
            while removed > 0 and i < len(unused):
                cand = unused[i]
                i += 1
                if current[cand["tool"]] < 25:
                    chosen.append(cand)
                    current[cand["tool"]] += 1
                    removed -= 1
            unused = unused[i:]
        # re-shuffle with deterministic rng
        rng2 = random.Random(_stable_int(SEED + "post") % (2**32))
        rng2.shuffle(chosen)
        # If we dropped below 500 (shouldn't), pad from any leftover specs.
        leftover = [s for s in specs if s not in chosen]
        rng2.shuffle(leftover)
        while len(chosen) < 500 and leftover:
            cand = leftover.pop()
            current = Counter(s["tool"] for s in chosen)
            if current[cand["tool"]] < 25:
                chosen.append(cand)
        chosen = chosen[:500]

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

    tools = [s["tool"] for s in chosen]
    distinct = len(set(tools))
    single_count = sum(1 for s in chosen if s["result"] is None)
    multi_count = len(chosen) - single_count
    suffix_used = set(sfx for sfx in suffixes if sfx)

    cnt = Counter(tools)
    over_cap = [(t, c) for t, c in cnt.items() if c > 25]
    max_share = max(cnt.values()) / len(tools)

    print(f"wrote {written} samples to {OUT_PATH}")
    print(f"distinct tools: {distinct}")
    print(f"single-turn: {single_count}")
    print(f"multi-turn: {multi_count}")
    print(f"max share: {max_share:.4f} ({max(cnt.values())}/{len(tools)})")
    print(f"suffix-pool coverage: {len(suffix_used)}/30")
    if over_cap:
        print(f"WARNING tools over 25 occurrences: {over_cap}")
    most_common = cnt.most_common(5)
    print(f"top 5 tools: {most_common}")


if __name__ == "__main__":
    main()
