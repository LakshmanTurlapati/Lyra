# SPDX-License-Identifier: MIT
"""Data tables for gen_tc_09_3_B.py. 500 samples for TC-B (messaging domain)."""

# Tool registry: name -> (arg_template, result_template)
# Each tool has multiple realistic argument variations and result variations.
# Format: list of (args_dict, result_str_or_None) examples.
# result == None means single-turn (no tool result needed).

# 50 distinct tools across email/SMS/chat/notifications/contacts/voice
TOOLS = [
    "send_email", "send_sms", "send_chat_message", "count_unread_emails",
    "mark_email_read", "archive_email", "delete_email", "search_emails",
    "add_contact", "update_contact", "list_contacts", "block_sender",
    "unsubscribe_from_list", "forward_email", "reply_email",
    "send_push_notification", "create_chat_channel", "add_user_to_channel",
    "pin_message", "react_to_message", "start_voice_call",
    "transcribe_voicemail", "set_dnd_status", "schedule_email",
    "get_email_thread", "tag_email", "move_to_folder", "create_email_template",
    "find_contact_by_email", "update_distribution_list", "remove_contact",
    "get_chat_history", "leave_channel", "mute_channel", "send_group_sms",
    "set_email_signature", "get_voicemail_list", "delete_voicemail",
    "send_calendar_invite_email", "search_chat_messages", "edit_chat_message",
    "delete_chat_message", "get_contact_details", "merge_contacts",
    "create_contact_group", "send_email_with_attachment", "snooze_email",
    "set_out_of_office", "list_email_folders", "create_email_folder",
    "get_unread_chat_count", "send_voice_message", "transcribe_call",
    "get_call_log", "block_phone_number", "set_chat_status",
    "send_typing_indicator", "get_thread_participants",
]

# Suffix pool (30 phrases) — exact strings.
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

# Diverse names, emails, phones, channels for argument realism
NAMES = [
    "Sarah Chen", "Marcus Webb", "Priya Patel", "Tom Hollis", "Yuki Tanaka",
    "Jordan Reyes", "Amelia Cross", "Diego Alvarez", "Hana Kim", "Felix Wu",
    "Olu Adeyemi", "Ingrid Larsen", "Vihaan Shah", "Carmen Ruiz", "Lior Bar",
    "Naomi Brooks", "Rashid Khan", "Elena Volkov", "Bram de Vries", "Mei Li",
    "Theo Castellano", "Aisha Bello", "Ravi Kapoor", "Lena Schulz", "Oscar Pham",
    "Imani Jackson", "Klaus Becker", "Saoirse Walsh", "Dmitri Orlov", "Pia Rossi",
    "Caleb Mensah", "Nia Harper", "Ezra Klein", "Maya Ortiz", "Finn O'Brien",
    "Zara Ahmad", "Hugo Bernard", "Talia Goldberg", "Cy Nguyen", "Esme Larkin",
]

EMAILS = [
    "sarah.chen@acme.io", "m.webb@globex.com", "priya@patel.dev",
    "thollis@northstar.co", "yuki.t@kobayashi.jp", "jordan@reyesart.com",
    "amelia.cross@sentinel.org", "diego@alvarez.mx", "hana.kim@kanto.kr",
    "felix.wu@nexus.io", "olu@adeyemi.ng", "ingrid@nordeal.no",
    "vshah@bharat.in", "carmen.ruiz@iberdrola.es", "lior@barlabs.il",
    "naomi.brooks@bayside.us", "rkhan@helios.ae", "elena.volkov@kremlin.ru",
    "bram@devries.nl", "meili@sinotech.cn",
]

PHONES = [
    "+1-555-0142", "+1-555-0317", "+1-555-0488", "+44-20-7946-0982",
    "+33-1-45-67-89-12", "+49-30-12345678", "+81-3-4567-8901",
    "+91-22-2345-6789", "+61-2-9876-5432", "+55-11-3456-7890",
]

CHANNELS = [
    "#engineering", "#product-launch", "#random", "#design-review",
    "#oncall-rotation", "#alerts-prod", "#general", "#hr-announcements",
    "#kitchen-fridge", "#ml-research", "#frontend", "#backend-platform",
    "#standup-eu", "#data-eng", "#security-incidents",
]

SUBJECTS = [
    "Q3 budget review", "Lunch tomorrow?", "Contract draft attached",
    "Vacation request", "Demo feedback", "Server outage postmortem",
    "Welcome aboard!", "Invoice #4471", "Reschedule our 1:1",
    "Roadmap proposal", "Security training reminder", "Press release draft",
    "Weekly metrics", "RFP response due Friday", "Holiday party RSVP",
]

# Sample specs: list of dicts describing each sample.
# Each spec: {tool, args, result, user, multi (bool)}
# result is None for single-turn.
# 500 specs, hand-shaped for diversity.
SAMPLES = []


def _add(tool, args, user, result=None):
    SAMPLES.append({"tool": tool, "args": args, "user": user, "result": result})


# === SINGLE-TURN samples (~80, target 75-80) ===
# These end with assistant tool_call and NO tool result.

_st = [
    ("send_sms", {"contact": "Sarah", "body": "Running 10 min late, sorry!"},
     "Send a quick text to Sarah saying I'm running late."),
    ("send_email", {"to": "boss@acme.io", "subject": "PTO next Friday",
                    "body": "Hi — I'd like to take Friday the 18th off. Let me know if that works."},
     "Email my boss requesting PTO for next Friday."),
    ("send_chat_message", {"channel": "#engineering", "text": "Standup pushed to 10am today."},
     "Tell engineering channel that standup is pushed to 10am."),
    ("send_push_notification", {"device": "user-phone-7281",
                                 "title": "Reminder", "body": "Pick up dry cleaning"},
     "Push a reminder to my phone to pick up dry cleaning."),
    ("send_sms", {"contact": "+1-555-0142", "body": "Are you free Saturday for hiking?"},
     "Text 555-0142 asking if they're free Saturday for hiking."),
    ("mark_email_read", {"email_id": "msg_a8f2c1"},
     "Mark email msg_a8f2c1 as read."),
    ("archive_email", {"email_id": "msg_77321"},
     "Archive email 77321 please."),
    ("delete_email", {"email_id": "msg_spam_9912"},
     "Delete that spam email, id msg_spam_9912."),
    ("block_sender", {"address": "promo@junkmail.biz"},
     "Block promo@junkmail.biz from emailing me."),
    ("unsubscribe_from_list", {"list_id": "newsletter-9981"},
     "Unsubscribe me from list newsletter-9981."),
    ("add_contact", {"name": "Marcus Webb", "email": "m.webb@globex.com", "phone": "+1-555-0317"},
     "Add Marcus Webb to my contacts. Email m.webb@globex.com, phone 555-0317."),
    ("update_contact", {"contact_id": "ctc_482", "field": "phone", "value": "+1-555-0999"},
     "Update contact ctc_482's phone to 555-0999."),
    ("remove_contact", {"contact_id": "ctc_119"},
     "Remove contact ctc_119."),
    ("forward_email", {"email_id": "msg_4421", "to": "legal@acme.io"},
     "Forward email msg_4421 to legal@acme.io."),
    ("reply_email", {"email_id": "msg_3300", "body": "Got it, will review by EOD."},
     "Reply to msg_3300 saying I'll review by EOD."),
    ("schedule_email", {"to": "team@acme.io", "subject": "Friday wrap-up",
                         "body": "Here's the week in review...", "send_at": "2026-05-15T17:00:00Z"},
     "Schedule a Friday wrap-up email to the team for 5pm next Friday."),
    ("tag_email", {"email_id": "msg_5512", "tag": "follow-up"},
     "Tag email msg_5512 as follow-up."),
    ("move_to_folder", {"email_id": "msg_881", "folder": "Receipts"},
     "Move msg_881 to the Receipts folder."),
    ("snooze_email", {"email_id": "msg_4422", "until": "2026-05-12T09:00:00Z"},
     "Snooze that email until Monday 9am — id msg_4422."),
    ("set_out_of_office", {"start": "2026-06-01", "end": "2026-06-08",
                            "message": "Out hiking. Back June 9. Urgent? Ping Diego."},
     "Set OOO from June 1 to June 8 — out hiking, ping Diego if urgent."),
    ("set_dnd_status", {"duration_minutes": 90},
     "Turn on do-not-disturb for the next 90 minutes."),
    ("set_chat_status", {"emoji": ":coffee:", "text": "Coffee run", "expires_in": 600},
     "Set my chat status to coffee run for 10 minutes."),
    ("create_chat_channel", {"name": "#q3-launch", "private": False, "purpose": "Q3 launch coordination"},
     "Create a public channel called q3-launch for launch coordination."),
    ("add_user_to_channel", {"channel": "#design-review", "user": "@hana"},
     "Add Hana to the design-review channel."),
    ("leave_channel", {"channel": "#kitchen-fridge"},
     "Leave the kitchen-fridge channel."),
    ("mute_channel", {"channel": "#alerts-prod", "duration_hours": 4},
     "Mute alerts-prod for 4 hours."),
    ("pin_message", {"channel": "#engineering", "message_id": "msg_chat_8821"},
     "Pin message 8821 in #engineering."),
    ("react_to_message", {"message_id": "msg_chat_4412", "emoji": "thumbsup"},
     "React thumbsup to chat message 4412."),
    ("edit_chat_message", {"message_id": "msg_chat_5523", "text": "Meeting moved to 3pm (corrected)."},
     "Edit chat message 5523 to say meeting moved to 3pm."),
    ("delete_chat_message", {"message_id": "msg_chat_7714"},
     "Delete chat message 7714."),
    ("send_typing_indicator", {"channel": "@jordan"},
     "Send a typing indicator to Jordan."),
    ("send_group_sms", {"recipients": ["Sarah", "Marcus", "Priya"],
                        "body": "Brunch Saturday 11am at Tartine?"},
     "Group text Sarah, Marcus, and Priya about brunch Saturday 11am at Tartine."),
    ("start_voice_call", {"contact": "Tom Hollis"},
     "Call Tom Hollis."),
    ("send_voice_message", {"contact": "Yuki", "audio_url": "rec://vm_88812", "duration_sec": 22},
     "Send Yuki the voice message I just recorded (rec://vm_88812, 22 seconds)."),
    ("block_phone_number", {"number": "+1-555-9999"},
     "Block 555-9999 from calling and texting me."),
    ("delete_voicemail", {"voicemail_id": "vm_4471"},
     "Delete voicemail vm_4471."),
    ("create_email_template", {"name": "Polite decline",
                                "subject": "Re: {original}",
                                "body": "Thanks for thinking of me — I'll have to pass this time."},
     "Make an email template called Polite decline that thanks them and passes."),
    ("set_email_signature", {"signature": "Best,\nLakshman\nFounder, Lyra"},
     "Set my email signature to Best, Lakshman, Founder, Lyra."),
    ("create_email_folder", {"name": "Receipts 2026"},
     "Create a new email folder called Receipts 2026."),
    ("create_contact_group", {"name": "Book club", "members": ["ctc_44", "ctc_71", "ctc_92"]},
     "Create a contact group called Book club with members 44, 71, and 92."),
    ("update_distribution_list", {"list_id": "dl_eng_all", "action": "add", "email": "felix@nexus.io"},
     "Add felix@nexus.io to the eng-all distribution list."),
    ("send_calendar_invite_email", {"to": ["sarah@acme.io", "marcus@globex.com"],
                                     "title": "Project sync", "when": "2026-05-14T15:00:00Z",
                                     "duration_minutes": 30},
     "Send a calendar invite for a 30-min project sync Wed 3pm to Sarah and Marcus."),
    ("send_email_with_attachment", {"to": "client@bigco.com", "subject": "Signed contract",
                                     "body": "Attached.", "attachment_path": "/docs/contract_v3.pdf"},
     "Email the client the signed contract at /docs/contract_v3.pdf."),
    ("merge_contacts", {"primary_id": "ctc_99", "duplicate_id": "ctc_412"},
     "Merge contact ctc_412 into ctc_99 — same person, duplicate entries."),
    ("transcribe_voicemail", {"voicemail_id": "vm_8821"},
     "Transcribe voicemail vm_8821."),
    ("transcribe_call", {"call_id": "call_4412"},
     "Transcribe call_4412 for me."),
    ("send_push_notification", {"device": "all-devices", "title": "Severe weather",
                                 "body": "Tornado watch in effect until 8pm."},
     "Push a severe-weather alert to all my devices about a tornado watch until 8pm."),
    ("forward_email", {"email_id": "msg_771", "to": ["accounting@acme.io", "diego@alvarez.mx"]},
     "Forward msg_771 to both accounting and Diego."),
    ("reply_email", {"email_id": "msg_2233", "body": "Confirmed — see you Tuesday."},
     "Reply to msg_2233 confirming Tuesday."),
    ("send_chat_message", {"channel": "@elena", "text": "Quick question when you have a sec — auth flow update?"},
     "DM Elena asking about the auth flow update."),
    ("send_sms", {"contact": "Mom", "body": "Landed safe. Calling in an hour."},
     "Text mom that I landed safe and will call in an hour."),
    ("add_contact", {"name": "Dr. Naomi Brooks", "email": "naomi.brooks@bayside.us"},
     "Save Dr. Naomi Brooks with email naomi.brooks@bayside.us as a new contact."),
    ("send_email", {"to": "support@hostingco.net", "subject": "Account locked",
                    "body": "My account is locked, can you help reset?"},
     "Email support at hostingco.net that my account is locked."),
    ("set_dnd_status", {"duration_minutes": 0, "off": True},
     "Turn off do not disturb."),
    ("create_chat_channel", {"name": "#incident-2026-05-08", "private": True,
                             "purpose": "Active incident war room"},
     "Spin up a private incident war room channel for today's outage."),
    ("send_chat_message", {"channel": "#general", "text": "Donuts in the kitchen!"},
     "Announce donuts in the kitchen on #general."),
    ("send_sms", {"contact": "Felix", "body": "Did the merge go through?"},
     "Ask Felix by text if the merge went through."),
    ("schedule_email", {"to": "self", "subject": "Renew domain",
                         "body": "Domain expires next month — renew it.", "send_at": "2026-06-01T08:00:00Z"},
     "Schedule a self-reminder email for June 1 to renew my domain."),
    ("mark_email_read", {"folder": "Promotions", "all": True},
     "Mark everything in Promotions as read."),
    ("archive_email", {"email_ids": ["msg_111", "msg_112", "msg_113"]},
     "Archive emails 111, 112, and 113."),
    ("delete_email", {"email_ids": ["msg_a", "msg_b", "msg_c", "msg_d"]},
     "Trash messages a, b, c, and d."),
    ("tag_email", {"email_id": "msg_8801", "tag": "tax-2025"},
     "Label msg_8801 with tax-2025."),
    ("snooze_email", {"email_id": "msg_99", "until": "tomorrow_morning"},
     "Snooze msg_99 till tomorrow morning."),
    ("send_email", {"to": ["alpha@x.io", "beta@x.io", "gamma@x.io"],
                    "subject": "Lunch poll", "body": "Thai, sushi, or pizza? Vote by 11."},
     "Email Alpha, Beta, and Gamma a lunch poll: Thai, sushi, or pizza, vote by 11."),
    ("send_chat_message", {"channel": "#oncall-rotation", "text": "I've got the pager today."},
     "Post in oncall-rotation that I've got the pager today."),
    ("react_to_message", {"message_id": "msg_chat_2210", "emoji": "rocket"},
     "Rocket-react message 2210."),
    ("pin_message", {"channel": "#product-launch", "message_id": "msg_chat_5577"},
     "Pin message 5577 in product-launch."),
    ("update_contact", {"contact_id": "ctc_771", "field": "title", "value": "VP Engineering"},
     "Update ctc_771's title to VP Engineering."),
    ("find_contact_by_email", {"email": "elena.volkov@kremlin.ru"},
     "Find the contact entry for elena.volkov@kremlin.ru."),
    ("get_contact_details", {"contact_id": "ctc_55"},
     "Pull up details for contact ctc_55."),
    ("start_voice_call", {"number": "+44-20-7946-0982"},
     "Dial +44 20 7946 0982."),
    ("send_voice_message", {"contact": "Dad", "audio_url": "rec://vm_0099", "duration_sec": 45},
     "Send dad the voice memo I just recorded (vm_0099, 45 sec)."),
    ("block_phone_number", {"number": "+1-555-0900"},
     "Block 555-0900."),
    ("set_out_of_office", {"start": "2026-12-23", "end": "2027-01-02",
                            "message": "Out for the holidays. Replies in January."},
     "Set out-of-office from Dec 23 to Jan 2 saying I'm out for the holidays."),
    ("set_email_signature", {"signature": "— L"},
     "Make my email signature just dash L."),
    ("create_email_folder", {"name": "Job applications"},
     "Make a folder called Job applications."),
    ("merge_contacts", {"primary_id": "ctc_aaa", "duplicate_id": "ctc_bbb"},
     "Merge ctc_bbb into ctc_aaa."),
    ("send_typing_indicator", {"channel": "#design-review"},
     "Drop a typing indicator in design-review."),
    ("set_chat_status", {"emoji": ":palm_tree:", "text": "Vacation", "expires_in": 604800},
     "Set my status to vacation palm tree for a week."),
]

for tool, args, user in _st:
    _add(tool, args, user)

# === MULTI-TURN samples (~420) ===
# Each has a result; final assistant message starts with a suffix-pool phrase.

_mt = [
    # count_unread_emails (varied)
    ("count_unread_emails", {"folder": "inbox"}, '{"count":17}',
     "How many unread emails do I have?"),
    ("count_unread_emails", {"folder": "Promotions"}, '{"count":214}',
     "Unread count in Promotions?"),
    ("count_unread_emails", {"folder": "Updates"}, '{"count":42}',
     "What's my Updates folder unread count?"),
    ("count_unread_emails", {"folder": "inbox", "since": "2026-05-01"}, '{"count":9}',
     "How many unread since May 1?"),
    ("count_unread_emails", {"folder": "Important"}, '{"count":3}',
     "Unread count in Important?"),

    # search_emails
    ("search_emails", {"query": "from:sarah subject:Q3"}, '{"results":[{"id":"msg_771","subject":"Q3 budget review","date":"2026-05-04"},{"id":"msg_812","subject":"Q3 forecast","date":"2026-05-06"}]}',
     "Find emails from Sarah about Q3."),
    ("search_emails", {"query": "invoice", "after": "2026-04-01"}, '{"results":[{"id":"msg_551","subject":"Invoice #4471","amount":1200},{"id":"msg_552","subject":"Invoice #4488","amount":850}]}',
     "Search my email for invoices since April."),
    ("search_emails", {"query": "subject:contract has:attachment"}, '{"results":[{"id":"msg_900","from":"legal@acme.io","date":"2026-04-30"}]}',
     "Look for contract emails with attachments."),
    ("search_emails", {"query": "from:newsletter", "limit": 5}, '{"results":[{"id":"m1"},{"id":"m2"},{"id":"m3"},{"id":"m4"},{"id":"m5"}],"total":48}',
     "Show me the latest 5 newsletters."),
    ("search_emails", {"query": "is:unread is:flagged"}, '{"results":[],"total":0}',
     "Any unread flagged emails?"),

    # get_email_thread
    ("get_email_thread", {"thread_id": "thr_8821"}, '{"messages":7,"participants":["sarah@acme.io","me","diego@alvarez.mx"],"subject":"Q3 budget review"}',
     "Pull up thread thr_8821."),
    ("get_email_thread", {"thread_id": "thr_119"}, '{"messages":3,"participants":["legal@acme.io","me"],"subject":"Contract draft"}',
     "Get thread 119."),
    ("get_email_thread", {"thread_id": "thr_2202"}, '{"messages":12,"participants":["me","ops@acme.io","sarah@acme.io"]}',
     "Show thread 2202."),

    # list_contacts
    ("list_contacts", {"limit": 10}, '{"contacts":[{"id":"ctc_1","name":"Sarah Chen"},{"id":"ctc_2","name":"Marcus Webb"},{"id":"ctc_3","name":"Priya Patel"}],"total":284}',
     "List my first 10 contacts."),
    ("list_contacts", {"group": "Book club"}, '{"contacts":[{"id":"ctc_44","name":"Naomi Brooks"},{"id":"ctc_71","name":"Esme Larkin"},{"id":"ctc_92","name":"Cy Nguyen"}]}',
     "Who's in my Book club contact group?"),
    ("list_contacts", {"starts_with": "K"}, '{"contacts":[{"id":"ctc_40","name":"Klaus Becker"}],"total":1}',
     "Contacts whose name starts with K?"),

    # get_contact_details
    ("get_contact_details", {"contact_id": "ctc_55"}, '{"name":"Diego Alvarez","email":"diego@alvarez.mx","phone":"+52-55-1234-5678","company":"Iberica"}',
     "Pull contact ctc_55's details."),
    ("get_contact_details", {"contact_id": "ctc_92"}, '{"name":"Cy Nguyen","email":"cy@nguyen.dev","phone":"+1-555-0888"}',
     "Show me ctc_92."),

    # find_contact_by_email
    ("find_contact_by_email", {"email": "felix.wu@nexus.io"}, '{"id":"ctc_201","name":"Felix Wu"}',
     "Look up the contact for felix.wu@nexus.io."),
    ("find_contact_by_email", {"email": "unknown@nowhere.xyz"}, '{"id":null,"matches":0}',
     "Is unknown@nowhere.xyz in my contacts?"),

    # get_chat_history
    ("get_chat_history", {"channel": "#engineering", "limit": 5}, '{"messages":[{"user":"@hana","text":"PR ready"},{"user":"@felix","text":"reviewing"},{"user":"@elena","text":"merged"},{"user":"@me","text":"thanks"},{"user":"@hana","text":"deploying"}]}',
     "Show me the last 5 messages in engineering."),
    ("get_chat_history", {"channel": "@diego", "limit": 3}, '{"messages":[{"user":"@diego","text":"flight ok?"},{"user":"@me","text":"landed"},{"user":"@diego","text":"safe travels"}]}',
     "Last few DMs with Diego?"),
    ("get_chat_history", {"channel": "#alerts-prod", "since": "2026-05-08T00:00:00Z"}, '{"messages":18}',
     "How busy was alerts-prod today?"),

    # search_chat_messages
    ("search_chat_messages", {"query": "deploy", "channel": "#engineering"}, '{"results":[{"id":"msg_chat_8801","ts":"2026-05-07T14:22Z","user":"@hana"}],"total":1}',
     "Search engineering for messages about deploy."),
    ("search_chat_messages", {"query": "from:@elena auth"}, '{"results":[{"id":"c_551"},{"id":"c_602"}],"total":2}',
     "Find Elena's messages about auth."),
    ("search_chat_messages", {"query": "incident", "after": "2026-05-01"}, '{"results":[{"id":"c_900"},{"id":"c_911"},{"id":"c_944"}],"total":3}',
     "Search chat for 'incident' since May 1."),

    # get_unread_chat_count
    ("get_unread_chat_count", {}, '{"total":34,"by_channel":{"#engineering":12,"#alerts-prod":18,"@diego":4}}',
     "How many unread chat messages do I have?"),
    ("get_unread_chat_count", {"channel": "#alerts-prod"}, '{"count":18}',
     "Unread in alerts-prod?"),

    # get_thread_participants
    ("get_thread_participants", {"thread_id": "chat_thr_55"}, '{"participants":["@hana","@felix","@elena","@me"]}',
     "Who's in chat thread 55?"),

    # get_voicemail_list
    ("get_voicemail_list", {"limit": 5}, '{"voicemails":[{"id":"vm_8821","from":"+1-555-0142","duration_sec":45,"unread":true},{"id":"vm_8820","from":"Mom","duration_sec":22,"unread":false}]}',
     "List my recent voicemails."),
    ("get_voicemail_list", {"unread_only": True}, '{"voicemails":[{"id":"vm_8821"},{"id":"vm_8819"}],"count":2}',
     "Any unread voicemails?"),

    # transcribe_voicemail
    ("transcribe_voicemail", {"voicemail_id": "vm_8821"},
     '{"transcript":"Hey it\'s Marcus, calling about the contract — give me a buzz back when you can."}',
     "Transcribe voicemail 8821."),
    ("transcribe_voicemail", {"voicemail_id": "vm_4471"},
     '{"transcript":"This is Dr. Brooks confirming your appointment for Thursday at 2pm."}',
     "What did vm_4471 say?"),

    # transcribe_call
    ("transcribe_call", {"call_id": "call_991"},
     '{"transcript":"Quick sync — we agreed to ship Friday and revisit metrics Monday.","duration_sec":420}',
     "Transcribe call 991."),

    # get_call_log
    ("get_call_log", {"limit": 5}, '{"calls":[{"id":"call_991","with":"Sarah","duration_sec":420,"direction":"outgoing"},{"id":"call_990","with":"+1-555-0142","duration_sec":62,"direction":"incoming"}]}',
     "Show my last 5 calls."),
    ("get_call_log", {"date": "2026-05-08"}, '{"calls":7}',
     "How many calls did I make today?"),

    # list_email_folders
    ("list_email_folders", {}, '{"folders":["Inbox","Sent","Drafts","Archive","Receipts","Receipts 2026","Job applications","Promotions"]}',
     "List my email folders."),

    # send_email (multi)
    ("send_email", {"to": "diego@alvarez.mx", "subject": "Q2 retro thoughts",
                    "body": "Quick brain dump on the retro before our 1:1 — main themes were velocity, hiring, and tooling."},
     '{"sent":true,"id":"msg_out_8821"}',
     "Email Diego my Q2 retro thoughts before our 1:1 — velocity, hiring, tooling."),
    ("send_email", {"to": "team@acme.io", "subject": "Friday wrap",
                    "body": "Shipped the auth refactor. Postmortem on Tuesday's blip below."},
     '{"sent":true,"id":"msg_out_8822"}',
     "Send the team my Friday wrap mentioning the auth refactor and Tuesday's postmortem."),
    ("send_email", {"to": "vendor@bigco.com", "subject": "Renewal questions",
                    "body": "A few questions before we sign the renewal — see attached list."},
     '{"sent":true,"id":"msg_out_8823"}',
     "Email the vendor with renewal questions."),
    ("send_email", {"to": "hr@acme.io", "subject": "Health benefits enrollment",
                    "body": "Confirming I'd like to keep current plan and add dental."},
     '{"sent":true,"id":"msg_out_8824"}',
     "Reply to HR confirming I want current health plan plus dental."),
    ("send_email", {"to": "carmen.ruiz@iberdrola.es", "subject": "Madrid trip",
                    "body": "I'll be in Madrid June 12-15 — coffee?"},
     '{"sent":true,"id":"msg_out_8825"}',
     "Email Carmen about my Madrid trip June 12-15 and ask for coffee."),
    ("send_email", {"to": "sarah.chen@acme.io", "subject": "1:1 prep",
                    "body": "Topics: roadmap reshuffle, hiring pipeline, demo feedback."},
     '{"sent":true,"id":"msg_out_8826"}',
     "Email Sarah my 1:1 prep topics: roadmap, hiring, demos."),
    ("send_email", {"to": "yuki.t@kobayashi.jp", "subject": "Pinging on signed NDA",
                    "body": "Hi Yuki — just following up on the NDA from last Tuesday."},
     '{"sent":true,"id":"msg_out_8827"}',
     "Follow up with Yuki by email about the NDA from last Tuesday."),

    # send_sms (multi)
    ("send_sms", {"contact": "Sarah", "body": "Reschedule to 4pm?"},
     '{"sent":true,"sid":"sms_4471"}',
     "Text Sarah asking to reschedule to 4pm."),
    ("send_sms", {"contact": "Marcus", "body": "On the train, see you at 9."},
     '{"sent":true,"sid":"sms_4472"}',
     "Tell Marcus I'm on the train and will see him at 9."),
    ("send_sms", {"contact": "Priya", "body": "Did you get my email?"},
     '{"sent":true,"sid":"sms_4473"}',
     "Text Priya asking if she got my email."),
    ("send_sms", {"contact": "+1-555-0488", "body": "Confirming pickup at 7pm tonight."},
     '{"sent":true,"sid":"sms_4474"}',
     "Text 555-0488 confirming pickup at 7 tonight."),
    ("send_sms", {"contact": "Hana", "body": "Free for lunch tomorrow?"},
     '{"sent":true,"sid":"sms_4475"}',
     "Text Hana about lunch tomorrow."),

    # send_chat_message (multi)
    ("send_chat_message", {"channel": "#engineering", "text": "Deploying v2.4 in 5."},
     '{"sent":true,"ts":"2026-05-08T15:42:11Z"}',
     "Tell engineering we're deploying v2.4 in 5."),
    ("send_chat_message", {"channel": "@elena", "text": "Free for a quick huddle?"},
     '{"sent":true,"ts":"2026-05-08T15:43:00Z"}',
     "DM Elena asking for a quick huddle."),
    ("send_chat_message", {"channel": "#product-launch", "text": "Slides are in the deck — ready for review."},
     '{"sent":true}',
     "Post in product-launch that the slides are in the deck and ready for review."),
    ("send_chat_message", {"channel": "#data-eng", "text": "Pipeline backfilled, ETL clean."},
     '{"sent":true}',
     "Tell data-eng the pipeline was backfilled and ETL is clean."),

    # add_contact (multi, varied)
    ("add_contact", {"name": "Jordan Reyes", "email": "jordan@reyesart.com", "phone": "+1-555-0317"},
     '{"id":"ctc_new_881","saved":true}',
     "Save Jordan Reyes — jordan@reyesart.com, 555-0317 — to my contacts."),
    ("add_contact", {"name": "Priya Patel", "email": "priya@patel.dev"},
     '{"id":"ctc_new_882"}',
     "Add Priya Patel with email priya@patel.dev to contacts."),
    ("add_contact", {"name": "Theo Castellano", "phone": "+39-06-1234-5678", "company": "Cinecittà"},
     '{"id":"ctc_new_883"}',
     "Add Theo Castellano, phone +39 06 1234 5678, company Cinecittà."),

    # update_contact (multi)
    ("update_contact", {"contact_id": "ctc_201", "field": "company", "value": "Nexus AI"},
     '{"updated":true}',
     "Update ctc_201's company to Nexus AI."),
    ("update_contact", {"contact_id": "ctc_44", "field": "email", "value": "n.brooks@bayside.us"},
     '{"updated":true}',
     "Change ctc_44's email to n.brooks@bayside.us."),

    # remove_contact, block_sender, unsubscribe
    ("remove_contact", {"contact_id": "ctc_old_77"},
     '{"removed":true}',
     "Delete contact ctc_old_77."),
    ("block_sender", {"address": "spam@junk.biz"},
     '{"blocked":true}',
     "Block spam@junk.biz."),
    ("block_sender", {"domain": "promo.cheapstuff.com"},
     '{"blocked_domain":"promo.cheapstuff.com"}',
     "Block the entire promo.cheapstuff.com domain from emailing me."),
    ("unsubscribe_from_list", {"list_id": "newsletter-acme-weekly"},
     '{"unsubscribed":true}',
     "Unsubscribe me from acme-weekly newsletter."),

    # forward, reply
    ("forward_email", {"email_id": "msg_771", "to": "diego@alvarez.mx"},
     '{"forwarded":true,"new_id":"msg_out_9001"}',
     "Forward msg_771 to Diego."),
    ("reply_email", {"email_id": "msg_3300", "body": "Sounds good — Tuesday at 2pm works."},
     '{"sent":true}',
     "Reply to msg_3300 confirming Tuesday at 2pm."),

    # send_push, schedule_email
    ("send_push_notification", {"device": "user-phone-7281", "title": "Deploy", "body": "Production deploy completed."},
     '{"delivered":true}',
     "Push me a deploy-completed notification on my phone."),
    ("schedule_email", {"to": "team@acme.io", "subject": "Monday standup notes",
                        "body": "...", "send_at": "2026-05-12T09:00:00Z"},
     '{"scheduled":true,"id":"sch_44"}',
     "Schedule the standup notes email for Monday 9am."),

    # tag, move, archive
    ("tag_email", {"email_id": "msg_5512", "tag": "tax-2025"},
     '{"tagged":true}',
     "Tag msg_5512 with tax-2025."),
    ("move_to_folder", {"email_id": "msg_881", "folder": "Receipts 2026"},
     '{"moved":true}',
     "Move msg_881 to Receipts 2026."),
    ("archive_email", {"email_id": "msg_4400"},
     '{"archived":true}',
     "Archive msg_4400."),

    # create channel etc
    ("create_chat_channel", {"name": "#q3-launch", "private": False, "purpose": "Q3 launch coordination"},
     '{"channel_id":"C09Q3L","created":true}',
     "Spin up a public q3-launch channel."),
    ("add_user_to_channel", {"channel": "#q3-launch", "user": "@hana"},
     '{"added":true}',
     "Add Hana to q3-launch."),
    ("leave_channel", {"channel": "#kitchen-fridge"},
     '{"left":true}',
     "Leave kitchen-fridge."),
    ("mute_channel", {"channel": "#alerts-prod", "duration_hours": 8},
     '{"muted_until":"2026-05-08T23:42Z"}',
     "Mute alerts-prod for 8 hours."),
    ("pin_message", {"channel": "#engineering", "message_id": "c_8801"},
     '{"pinned":true}',
     "Pin c_8801 in engineering."),
    ("react_to_message", {"message_id": "c_4412", "emoji": "tada"},
     '{"reacted":true}',
     "Tada-react message c_4412."),
    ("edit_chat_message", {"message_id": "c_5523", "text": "Meeting moved to 4pm."},
     '{"edited":true}',
     "Edit c_5523 to say meeting moved to 4pm."),
    ("delete_chat_message", {"message_id": "c_7714"},
     '{"deleted":true}',
     "Delete c_7714."),

    # voice & sms group
    ("send_group_sms", {"recipients": ["Sarah", "Marcus", "Priya"],
                        "body": "Brunch Saturday 11am at Tartine?"},
     '{"sent":3,"failed":0}',
     "Group text Sarah, Marcus, Priya about brunch Saturday at Tartine."),
    ("start_voice_call", {"contact": "Tom Hollis"},
     '{"call_id":"call_4499","status":"ringing"}',
     "Call Tom Hollis."),
    ("send_voice_message", {"contact": "Yuki", "audio_url": "rec://vm_88812", "duration_sec": 22},
     '{"sent":true}',
     "Send Yuki the voice message I just recorded."),
    ("block_phone_number", {"number": "+1-555-9999"},
     '{"blocked":true}',
     "Block 555-9999."),
    ("delete_voicemail", {"voicemail_id": "vm_4471"},
     '{"deleted":true}',
     "Trash vm_4471."),

    # templates, signature, folder, group, dl
    ("create_email_template", {"name": "Polite decline",
                                "subject": "Re: {original}",
                                "body": "Thanks for thinking of me — passing this time."},
     '{"id":"tpl_22"}',
     "Create a Polite decline email template."),
    ("set_email_signature", {"signature": "— L"},
     '{"updated":true}',
     "Set signature to dash L."),
    ("create_email_folder", {"name": "Job applications"},
     '{"created":true,"folder_id":"f_99"}',
     "Make a Job applications folder."),
    ("create_contact_group", {"name": "Book club", "members": ["ctc_44", "ctc_71", "ctc_92"]},
     '{"group_id":"grp_22"}',
     "Create Book club group with 44, 71, 92."),
    ("update_distribution_list", {"list_id": "dl_eng_all", "action": "add", "email": "felix@nexus.io"},
     '{"members":48}',
     "Add felix@nexus.io to dl_eng_all."),
    ("update_distribution_list", {"list_id": "dl_eng_all", "action": "remove", "email": "alum@old.com"},
     '{"members":47}',
     "Remove alum@old.com from dl_eng_all."),

    # invite & attachment
    ("send_calendar_invite_email", {"to": ["sarah@acme.io"], "title": "1:1",
                                     "when": "2026-05-13T15:00:00Z", "duration_minutes": 30},
     '{"sent":true,"event_id":"ev_771"}',
     "Send Sarah a 1:1 calendar invite for Wed at 3pm."),
    ("send_email_with_attachment", {"to": "client@bigco.com", "subject": "Signed contract",
                                     "body": "Attached.", "attachment_path": "/docs/contract_v3.pdf"},
     '{"sent":true,"id":"msg_out_9300"}',
     "Email the client the signed contract from /docs/contract_v3.pdf."),
    ("send_email_with_attachment", {"to": "hr@acme.io", "subject": "Receipt", "body": "for reimbursement",
                                     "attachment_path": "/receipts/uber_april.pdf"},
     '{"sent":true,"id":"msg_out_9301"}',
     "Email HR the Uber April receipt for reimbursement."),

    # merge, snooze, OOO, status
    ("merge_contacts", {"primary_id": "ctc_99", "duplicate_id": "ctc_412"},
     '{"merged":true,"surviving_id":"ctc_99"}',
     "Merge ctc_412 into ctc_99."),
    ("snooze_email", {"email_id": "msg_99", "until": "2026-05-12T09:00:00Z"},
     '{"snoozed_until":"2026-05-12T09:00:00Z"}',
     "Snooze msg_99 till Monday 9am."),
    ("set_out_of_office", {"start": "2026-06-01", "end": "2026-06-08",
                            "message": "Hiking. Back June 9."},
     '{"set":true}',
     "Set OOO June 1-8: hiking, back June 9."),
    ("set_dnd_status", {"duration_minutes": 90},
     '{"dnd_until":"2026-05-08T17:12Z"}',
     "DND on for 90 minutes."),
    ("set_chat_status", {"emoji": ":coffee:", "text": "Coffee", "expires_in": 600},
     '{"set":true}',
     "Status: coffee for 10 min."),

    # typing, find, get details
    ("send_typing_indicator", {"channel": "@jordan"},
     '{"sent":true}',
     "Typing indicator to Jordan."),
    ("find_contact_by_email", {"email": "elena.volkov@kremlin.ru"},
     '{"id":"ctc_310","name":"Elena Volkov","phone":"+7-495-1234567"}',
     "Find contact for elena.volkov@kremlin.ru."),
    ("get_contact_details", {"contact_id": "ctc_310"},
     '{"name":"Elena Volkov","email":"elena.volkov@kremlin.ru","phone":"+7-495-1234567","company":"State Tech"}',
     "Show ctc_310."),
]

for tool, args, result, user in _mt:
    _add(tool, args, user, result=result)


def get_samples():
    return SAMPLES


def get_suffix_pool():
    return SUFFIX_POOL


def get_tools():
    return TOOLS
