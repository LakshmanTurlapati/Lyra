"""Fixture data for TC-F2 (fresh batch, distinct from TC-F)."""

PROJECT_KEYS = [
    "PAYMT", "BILL", "AUTH", "ONB", "DASH", "RPT", "INSI", "RECO", "CHECKO",
    "FRAUD", "RISK", "TRUST", "BRAND", "VOICE", "TONE", "SHIP", "WARE", "PICK",
    "INV", "STOCK", "CRED", "DEBIT", "AUDIT", "TAXES", "DOCSV2", "ARCAD",
    "LUME", "VERSE", "ECHO2", "PHX", "ZENI",
]
LIN_PREFIXES = ["JIRA", "ISS", "BUG2", "STORY", "EPIC", "TASKR"]

FIRST_NAMES = [
    "anya", "boris", "celine", "darius", "esme", "fiona", "gunnar", "halle",
    "iggy", "jolie", "krish", "leland", "magda", "nikolai", "ophelia", "pavel",
    "quill", "renata", "sergio", "thalia", "ulysses", "vera", "willem", "ximena",
    "yusuf", "zora", "asha", "bram", "cyril", "dilan", "esteban", "freya",
    "gerwin", "hanako", "ilse", "junko", "kostas", "lior",
]
LAST_NAMES = [
    "blackwood", "carrasco", "duarte", "elsayed", "freeman", "gallagher",
    "hansen", "ito", "jameson", "kostov", "lindqvist", "mendoza", "nikolic",
    "ostrowski", "pereira", "quintero", "ramos", "saito", "thompson", "uchida",
    "vasquez", "wexler", "yilmaz", "zambrano",
]
COMPANIES = [
    "lumencorp", "northbeam", "oakridge", "petalsoft", "quaildata", "ridgemont",
    "silverleaf", "tidewater", "umbra", "vellum", "winterfell-co", "xerolab",
    "yardline", "zephyr-io",
]
SPRINT_NAMES = [
    "Andromeda", "Boreas", "Cygnus", "Draco", "Electra", "Fornax", "Gemini",
    "Hydra", "Indus", "Juno", "Kepler", "Leo", "Mensa", "Norma", "Octans",
    "Phoenix", "Quill", "Reticulum", "Sextans", "Triangulum",
]
TASK_TITLES = [
    "Document the new auth handshake", "Reduce checkout latency",
    "Wire up dunning email retries", "Build pricing experiment harness",
    "Decom the v1 webhook receiver", "Stand up read replica for analytics",
    "Replace cron jobs with workflow engine", "Write runbook for region failover",
    "Backfill missing customer tax IDs", "Triage holiday-season bug spike",
    "Refresh design tokens for marketing", "Stand up branch deploy previews",
    "Tidy up flaky e2e suite", "Write FAQ for new refund policy",
    "Cap memory on the embeddings worker", "Move secrets to KMS",
    "Add SLO dashboards for ingest", "Move cold storage to glacier tier",
    "Sketch out post-mortem template v3", "Cut over DNS to new CDN",
    "Audit unused feature flags", "Stand up a status page",
    "Tighten OAuth scopes for partners", "Add idempotency keys to refund flow",
    "Profile slow report generation",
]
LEAD_SOURCES = [
    "podcast", "newsletter", "search_ad", "review_site", "user_referral",
    "field_event", "partner_marketplace", "warm_intro", "youtube",
]
DEAL_STAGES = [
    "intro_call", "discovery_done", "champion_secured", "procurement",
    "legal_review", "verbal_yes", "signed", "no_decision",
]
EXPENSE_CATS = [
    "ground_transport", "airfare", "client_gift", "tools", "team_lunch",
    "shipping", "co_working", "professional_services",
]
EMP_DEPTS = [
    "platform", "growth", "biz_ops", "people_ops", "finance_ops",
    "customer_success", "research", "ml", "compliance",
]
