#!/usr/bin/env python3
"""TC-G generator: 500 ShareGPT tool-calling samples for finance/banking/crypto/accounting."""
import json
import random
import os

SEED = "1009313G"
OUT = "/Users/lakshman/Documents/Lyra/datasets/tool-calling/raw-09.3/batch-13-G.jsonl"
SYSTEM = "You are a helpful assistant. Prefer calling tools over guessing."

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

TICKERS = ["AAPL","MSFT","NVDA","GOOGL","AMZN","TSLA","META","NFLX","AMD","INTC","CRM","ORCL","ADBE","CSCO","PEP","KO","JPM","BAC","WFC","GS","MS","C","V","MA","DIS","NKE","WMT","TGT","HD","LOW","COST","MCD","SBUX","UBER","LYFT","ABNB","SHOP","SQ","PYPL","COIN","HOOD","PLTR","SNOW","DDOG","CRWD","ZS","NET","PANW","FTNT","ROKU","SPOT","PINS","SNAP","TWLO","DOCU","ZM","OKTA","WDAY","NOW","TEAM","BA","CAT","DE","GE","F","GM","XOM","CVX","COP","SLB","UNH","PFE","MRK","JNJ","ABBV","LLY","TMO","DHR","BMY","T","VZ","TMUS","CMCSA"]
CRYPTOS = ["BTC","ETH","SOL","ADA","DOGE","XRP","DOT","AVAX","MATIC","LINK","UNI","ATOM","LTC","BCH","NEAR","ARB","OP","SUI","APT","FIL","ICP","HBAR","ALGO","SHIB","PEPE","WIF","BNB","TRX","TON","ETC"]
QUOTES = ["USD","USDT","USDC","EUR","GBP","JPY"]
CURRENCIES = ["USD","EUR","GBP","JPY","CAD","AUD","CHF","CNY","INR","MXN","BRL","SGD","HKD","NZD","SEK","NOK"]

def acct(rng): return f"acct_{rng.randint(10000000,99999999)}"
def order_id(rng): return f"ord_{rng.randint(100000,999999)}"
def txn_id(rng): return f"txn_{rng.randint(10000000,99999999)}"
def card_id(rng): return f"card_{rng.randint(1000,9999)}"
def loan_id(rng): return f"loan_{rng.randint(100000,999999)}"
def wallet(rng): return "0x" + "".join(rng.choice("0123456789abcdef") for _ in range(40))
def invoice_id(rng): return f"INV-{rng.randint(1000,9999)}"
def amt(rng, lo=10, hi=10000): return round(rng.uniform(lo, hi), 2)

# 50 distinct tools
TOOLS = [
    "get_stock_price","get_stock_quote","get_options_chain","place_market_order","place_limit_order",
    "place_stop_order","cancel_order","get_order_status","get_portfolio","get_positions",
    "get_buying_power","get_dividend_history","get_earnings_date","get_company_fundamentals","screener_run",
    "get_crypto_price","place_crypto_order","transfer_crypto","get_wallet_balance","swap_crypto",
    "get_gas_price","get_account_balance","list_recent_transactions","transfer_funds","pay_bill",
    "schedule_payment","cancel_payment","send_p2p","request_p2p","dispute_charge",
    "freeze_card","unfreeze_card","report_card_lost","apply_for_loan","check_credit_score",
    "book_journal_entry","record_invoice","record_expense","reconcile_account","generate_balance_sheet",
    "generate_pnl","convert_currency","get_fx_rate","get_loan_payoff","get_interest_rate",
    "get_mortgage_quote","place_futures_order","get_options_quote","stake_crypto","unstake_crypto",
    "get_staking_rewards","get_tax_lot","list_open_orders","get_account_statement",
]

def gen_args(rng, tool):
    if tool == "get_stock_price": return {"symbol": rng.choice(TICKERS)}
    if tool == "get_stock_quote": return {"symbol": rng.choice(TICKERS), "extended_hours": rng.choice([True, False])}
    if tool == "get_options_chain": return {"symbol": rng.choice(TICKERS), "expiry": f"2026-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}"}
    if tool == "place_market_order": return {"symbol": rng.choice(TICKERS), "side": rng.choice(["buy","sell"]), "quantity": rng.randint(1,500)}
    if tool == "place_limit_order": return {"symbol": rng.choice(TICKERS), "side": rng.choice(["buy","sell"]), "quantity": rng.randint(1,500), "limit_price": amt(rng,5,800)}
    if tool == "place_stop_order": return {"symbol": rng.choice(TICKERS), "side": rng.choice(["buy","sell"]), "quantity": rng.randint(1,500), "stop_price": amt(rng,5,800)}
    if tool == "cancel_order": return {"order_id": order_id(rng)}
    if tool == "get_order_status": return {"order_id": order_id(rng)}
    if tool == "get_portfolio": return {"account_id": acct(rng)}
    if tool == "get_positions": return {"account_id": acct(rng)}
    if tool == "get_buying_power": return {"account_id": acct(rng)}
    if tool == "get_dividend_history": return {"symbol": rng.choice(TICKERS), "years": rng.randint(1,10)}
    if tool == "get_earnings_date": return {"symbol": rng.choice(TICKERS)}
    if tool == "get_company_fundamentals": return {"symbol": rng.choice(TICKERS)}
    if tool == "screener_run": return {"min_market_cap": rng.choice([1e9,5e9,10e9,50e9]), "sector": rng.choice(["tech","finance","healthcare","energy","consumer"])}
    if tool == "get_crypto_price": return {"symbol": rng.choice(CRYPTOS), "quote": rng.choice(QUOTES)}
    if tool == "place_crypto_order": return {"symbol": rng.choice(CRYPTOS), "side": rng.choice(["buy","sell"]), "quantity": round(rng.uniform(0.001,5),4), "quote": rng.choice(QUOTES)}
    if tool == "transfer_crypto": return {"symbol": rng.choice(CRYPTOS), "amount": round(rng.uniform(0.01,2),4), "to_address": wallet(rng)}
    if tool == "get_wallet_balance": return {"address": wallet(rng)}
    if tool == "swap_crypto": return {"from_symbol": rng.choice(CRYPTOS), "to_symbol": rng.choice(CRYPTOS), "amount": round(rng.uniform(0.1,10),3)}
    if tool == "get_gas_price": return {"network": rng.choice(["ethereum","polygon","arbitrum","optimism","base"])}
    if tool == "get_account_balance": return {"account_id": acct(rng)}
    if tool == "list_recent_transactions": return {"account_id": acct(rng), "limit": rng.randint(5,50)}
    if tool == "transfer_funds": return {"from_account": acct(rng), "to_account": acct(rng), "amount": amt(rng,50,5000)}
    if tool == "pay_bill": return {"biller": rng.choice(["ConEd","Verizon","Comcast","PG&E","ATT","Spectrum"]), "amount": amt(rng,20,500), "account_id": acct(rng)}
    if tool == "schedule_payment": return {"payee": rng.choice(["landlord","ConEd","Verizon","mortgage"]), "amount": amt(rng,100,3000), "date": f"2026-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}"}
    if tool == "cancel_payment": return {"payment_id": txn_id(rng)}
    if tool == "send_p2p": return {"recipient": rng.choice(["@alice","@bob","@carol","@dave","@erin"]), "amount": amt(rng,5,500), "note": rng.choice(["dinner","rent share","coffee","groceries","tickets"])}
    if tool == "request_p2p": return {"from_user": rng.choice(["@alice","@bob","@carol"]), "amount": amt(rng,5,300), "note": rng.choice(["lunch","split","cab"])}
    if tool == "dispute_charge": return {"transaction_id": txn_id(rng), "reason": rng.choice(["unauthorized","duplicate","not_received","wrong_amount"])}
    if tool == "freeze_card": return {"card_id": card_id(rng)}
    if tool == "unfreeze_card": return {"card_id": card_id(rng)}
    if tool == "report_card_lost": return {"card_id": card_id(rng)}
    if tool == "apply_for_loan": return {"amount": amt(rng,5000,100000), "term_months": rng.choice([12,24,36,48,60]), "purpose": rng.choice(["auto","home_improvement","debt_consolidation","business"])}
    if tool == "check_credit_score": return {"ssn_last4": f"{rng.randint(0,9999):04d}"}
    if tool == "book_journal_entry": return {"debit_account": rng.choice(["Cash","AR","Inventory","Equipment"]), "credit_account": rng.choice(["AP","Revenue","Equity","Loans"]), "amount": amt(rng,100,50000), "memo": "monthly entry"}
    if tool == "record_invoice": return {"customer": rng.choice(["Acme Co","Globex","Initech","Umbrella"]), "amount": amt(rng,500,25000), "due_date": f"2026-{rng.randint(6,12):02d}-15"}
    if tool == "record_expense": return {"category": rng.choice(["office","travel","meals","software","utilities"]), "amount": amt(rng,10,2000), "vendor": rng.choice(["Amazon","Uber","Slack","AWS"])}
    if tool == "reconcile_account": return {"account_id": acct(rng), "period": f"2026-{rng.randint(1,5):02d}"}
    if tool == "generate_balance_sheet": return {"as_of": f"2026-{rng.randint(1,5):02d}-30"}
    if tool == "generate_pnl": return {"start": "2026-01-01", "end": f"2026-{rng.randint(2,5):02d}-30"}
    if tool == "convert_currency": return {"amount": amt(rng,10,10000), "from_currency": rng.choice(CURRENCIES), "to_currency": rng.choice(CURRENCIES)}
    if tool == "get_fx_rate": return {"base": rng.choice(CURRENCIES), "quote": rng.choice(CURRENCIES)}
    if tool == "get_loan_payoff": return {"loan_id": loan_id(rng)}
    if tool == "get_interest_rate": return {"product": rng.choice(["savings","cd_12m","cd_24m","money_market"])}
    if tool == "get_mortgage_quote": return {"loan_amount": amt(rng,100000,800000), "term_years": rng.choice([15,20,30]), "credit_score": rng.randint(620,820)}
    if tool == "place_futures_order": return {"symbol": rng.choice(["ES","NQ","CL","GC","SI","ZB"]), "side": rng.choice(["buy","sell"]), "contracts": rng.randint(1,10)}
    if tool == "get_options_quote": return {"symbol": rng.choice(TICKERS), "strike": amt(rng,50,500), "expiry": f"2026-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}", "type": rng.choice(["call","put"])}
    if tool == "stake_crypto": return {"symbol": rng.choice(["ETH","SOL","ADA","DOT","ATOM"]), "amount": round(rng.uniform(0.5,100),3)}
    if tool == "unstake_crypto": return {"symbol": rng.choice(["ETH","SOL","ADA","DOT","ATOM"]), "amount": round(rng.uniform(0.5,100),3)}
    if tool == "get_staking_rewards": return {"address": wallet(rng)}
    if tool == "get_tax_lot": return {"symbol": rng.choice(TICKERS), "account_id": acct(rng)}
    if tool == "list_open_orders": return {"account_id": acct(rng)}
    if tool == "get_account_statement": return {"account_id": acct(rng), "month": f"2026-{rng.randint(1,5):02d}"}
    return {}

def gen_user(rng, tool, args):
    """User prompt that motivates calling this tool."""
    t = tool
    if t == "get_stock_price": return rng.choice([f"What's {args['symbol']} at right now?", f"Quick check on {args['symbol']} price?", f"How much is {args['symbol']} trading for?", f"Price on {args['symbol']}?"])
    if t == "get_stock_quote": return f"Pull a full quote on {args['symbol']}{' including pre-market' if args.get('extended_hours') else ''}."
    if t == "get_options_chain": return f"Show me the options chain for {args['symbol']} expiring {args['expiry']}."
    if t == "place_market_order": return rng.choice([f"{args['side'].title()} {args['quantity']} shares of {args['symbol']} at market.", f"Market {args['side']} {args['quantity']} {args['symbol']} please.", f"Go ahead and {args['side']} {args['quantity']} {args['symbol']} now."])
    if t == "place_limit_order": return f"Put in a limit {args['side']} for {args['quantity']} {args['symbol']} at ${args['limit_price']}."
    if t == "place_stop_order": return f"Set a stop {args['side']} on {args['quantity']} {args['symbol']} at ${args['stop_price']}."
    if t == "cancel_order": return f"Cancel order {args['order_id']}."
    if t == "get_order_status": return f"What's the status on {args['order_id']}?"
    if t == "get_portfolio": return f"Pull up my portfolio for account {args['account_id']}."
    if t == "get_positions": return f"What positions do I hold in {args['account_id']}?"
    if t == "get_buying_power": return f"How much buying power on {args['account_id']}?"
    if t == "get_dividend_history": return f"Get me {args['years']} years of dividend history for {args['symbol']}."
    if t == "get_earnings_date": return f"When does {args['symbol']} report earnings?"
    if t == "get_company_fundamentals": return f"Pull fundamentals on {args['symbol']}."
    if t == "screener_run": return f"Run a screener for {args['sector']} stocks above ${int(args['min_market_cap']/1e9)}B market cap."
    if t == "get_crypto_price": return rng.choice([f"What's {args['symbol']} trading at?", f"Price check on {args['symbol']} in {args['quote']}?", f"How's {args['symbol']} doing?"])
    if t == "place_crypto_order": return f"{args['side'].title()} {args['quantity']} {args['symbol']} against {args['quote']}."
    if t == "transfer_crypto": return f"Send {args['amount']} {args['symbol']} to {args['to_address'][:10]}..."
    if t == "get_wallet_balance": return f"Check the balance on wallet {args['address'][:10]}..."
    if t == "swap_crypto": return f"Swap {args['amount']} {args['from_symbol']} for {args['to_symbol']}."
    if t == "get_gas_price": return f"What's gas looking like on {args['network']}?"
    if t == "get_account_balance": return f"Balance on {args['account_id']}?"
    if t == "list_recent_transactions": return f"Show last {args['limit']} transactions on {args['account_id']}."
    if t == "transfer_funds": return f"Move ${args['amount']} from {args['from_account']} to {args['to_account']}."
    if t == "pay_bill": return f"Pay my {args['biller']} bill of ${args['amount']} from {args['account_id']}."
    if t == "schedule_payment": return f"Schedule ${args['amount']} to {args['payee']} on {args['date']}."
    if t == "cancel_payment": return f"Cancel scheduled payment {args['payment_id']}."
    if t == "send_p2p": return f"Send ${args['amount']} to {args['recipient']} for {args['note']}."
    if t == "request_p2p": return f"Request ${args['amount']} from {args['from_user']} for {args['note']}."
    if t == "dispute_charge": return f"Dispute transaction {args['transaction_id']} — {args['reason'].replace('_',' ')}."
    if t == "freeze_card": return f"Freeze card {args['card_id']} immediately."
    if t == "unfreeze_card": return f"Unfreeze {args['card_id']}, I found it."
    if t == "report_card_lost": return f"My card {args['card_id']} is lost — report it."
    if t == "apply_for_loan": return f"Apply for a ${int(args['amount'])} loan over {args['term_months']} months for {args['purpose'].replace('_',' ')}."
    if t == "check_credit_score": return f"Run a credit pull, last 4 SSN {args['ssn_last4']}."
    if t == "book_journal_entry": return f"Book a journal entry: debit {args['debit_account']} credit {args['credit_account']} for ${args['amount']}."
    if t == "record_invoice": return f"Create an invoice for {args['customer']} — ${args['amount']} due {args['due_date']}."
    if t == "record_expense": return f"Log a {args['category']} expense of ${args['amount']} from {args['vendor']}."
    if t == "reconcile_account": return f"Reconcile {args['account_id']} for {args['period']}."
    if t == "generate_balance_sheet": return f"Generate the balance sheet as of {args['as_of']}."
    if t == "generate_pnl": return f"Run a P&L from {args['start']} to {args['end']}."
    if t == "convert_currency": return f"Convert {args['amount']} {args['from_currency']} to {args['to_currency']}."
    if t == "get_fx_rate": return f"What's {args['base']}/{args['quote']} right now?"
    if t == "get_loan_payoff": return f"What's the payoff on {args['loan_id']}?"
    if t == "get_interest_rate": return f"Current rate on {args['product']}?"
    if t == "get_mortgage_quote": return f"Mortgage quote: ${int(args['loan_amount'])} over {args['term_years']} years, credit {args['credit_score']}."
    if t == "place_futures_order": return f"{args['side'].title()} {args['contracts']} {args['symbol']} futures."
    if t == "get_options_quote": return f"Quote on {args['symbol']} ${args['strike']} {args['type']} {args['expiry']}."
    if t == "stake_crypto": return f"Stake {args['amount']} {args['symbol']}."
    if t == "unstake_crypto": return f"Unstake {args['amount']} {args['symbol']}."
    if t == "get_staking_rewards": return f"What rewards have I earned on {args['address'][:10]}...?"
    if t == "get_tax_lot": return f"Pull tax lots for {args['symbol']} on {args['account_id']}."
    if t == "list_open_orders": return f"What open orders on {args['account_id']}?"
    if t == "get_account_statement": return f"Pull statement for {args['account_id']}, {args['month']}."
    return "Help me with this."

def gen_tool_result(rng, tool, args):
    if tool in ("get_stock_price","get_stock_quote"):
        return {"symbol": args['symbol'], "price": amt(rng,5,800), "change_pct": round(rng.uniform(-4,4),2)}
    if tool == "get_options_chain":
        return {"symbol": args['symbol'], "expiry": args['expiry'], "calls": rng.randint(20,80), "puts": rng.randint(20,80)}
    if tool in ("place_market_order","place_limit_order","place_stop_order","place_crypto_order","place_futures_order"):
        return {"order_id": order_id(rng), "status": "filled" if rng.random()>0.3 else "open"}
    if tool == "cancel_order": return {"order_id": args['order_id'], "status": "cancelled"}
    if tool == "get_order_status": return {"order_id": args['order_id'], "status": rng.choice(["filled","open","cancelled","partial"])}
    if tool == "get_portfolio": return {"value": amt(rng,10000,500000), "cash": amt(rng,1000,50000)}
    if tool == "get_positions": return {"positions": rng.randint(3,25)}
    if tool == "get_buying_power": return {"buying_power": amt(rng,1000,100000)}
    if tool == "get_dividend_history": return {"symbol": args['symbol'], "total_paid": amt(rng,100,5000)}
    if tool == "get_earnings_date": return {"symbol": args['symbol'], "next_earnings": f"2026-{rng.randint(6,12):02d}-{rng.randint(1,28):02d}"}
    if tool == "get_company_fundamentals": return {"symbol": args['symbol'], "pe": round(rng.uniform(8,60),1), "eps": round(rng.uniform(0.5,15),2)}
    if tool == "screener_run": return {"matches": rng.randint(5,80)}
    if tool == "get_crypto_price": return {"price": amt(rng,0.1,80000), "change_24h": round(rng.uniform(-8,8),2)}
    if tool == "transfer_crypto": return {"tx_hash": "0x"+"".join(rng.choice("0123456789abcdef") for _ in range(20)), "status": "pending"}
    if tool == "get_wallet_balance": return {"balance_eth": round(rng.uniform(0.01,50),4)}
    if tool == "swap_crypto": return {"received": round(rng.uniform(0.1,1000),3), "fee_pct": 0.3}
    if tool == "get_gas_price": return {"gwei": round(rng.uniform(8,120),1)}
    if tool == "get_account_balance": return {"balance": amt(rng,100,50000), "available": amt(rng,100,50000)}
    if tool == "list_recent_transactions": return {"count": args['limit'], "total_out": amt(rng,500,8000)}
    if tool in ("transfer_funds","pay_bill","send_p2p","schedule_payment"):
        return {"txn_id": txn_id(rng), "status": "success"}
    if tool == "request_p2p": return {"request_id": txn_id(rng), "status": "sent"}
    if tool == "cancel_payment": return {"payment_id": args['payment_id'], "status": "cancelled"}
    if tool == "dispute_charge": return {"case_id": f"DSP-{rng.randint(10000,99999)}", "status": "opened"}
    if tool in ("freeze_card","unfreeze_card","report_card_lost"):
        st = {"freeze_card":"frozen","unfreeze_card":"active","report_card_lost":"replacement_ordered"}[tool]
        return {"card_id": args['card_id'], "status": st}
    if tool == "apply_for_loan": return {"app_id": f"LA-{rng.randint(10000,99999)}", "status": "under_review"}
    if tool == "check_credit_score": return {"score": rng.randint(580,820), "bureau": rng.choice(["Equifax","Experian","TransUnion"])}
    if tool == "book_journal_entry": return {"je_id": f"JE-{rng.randint(1000,9999)}", "posted": True}
    if tool == "record_invoice": return {"invoice_id": invoice_id(rng), "status": "draft"}
    if tool == "record_expense": return {"expense_id": f"EXP-{rng.randint(1000,9999)}", "status": "recorded"}
    if tool == "reconcile_account": return {"matched": rng.randint(20,200), "unmatched": rng.randint(0,5)}
    if tool == "generate_balance_sheet": return {"assets": amt(rng,100000,5000000), "liabilities": amt(rng,50000,3000000)}
    if tool == "generate_pnl": return {"revenue": amt(rng,50000,2000000), "net_income": amt(rng,5000,500000)}
    if tool == "convert_currency": return {"converted": round(args['amount']*rng.uniform(0.5,2),2), "rate": round(rng.uniform(0.5,2),4)}
    if tool == "get_fx_rate": return {"rate": round(rng.uniform(0.5,2),4)}
    if tool == "get_loan_payoff": return {"payoff": amt(rng,1000,80000), "good_through": f"2026-{rng.randint(6,12):02d}-30"}
    if tool == "get_interest_rate": return {"apy": round(rng.uniform(0.5,5.5),2)}
    if tool == "get_mortgage_quote": return {"rate": round(rng.uniform(5.5,7.8),3), "monthly": amt(rng,800,5000)}
    if tool == "get_options_quote": return {"bid": amt(rng,0.5,30), "ask": amt(rng,0.5,30), "iv": round(rng.uniform(0.15,0.9),3)}
    if tool in ("stake_crypto","unstake_crypto"): return {"status": "pending", "epoch": rng.randint(100,500)}
    if tool == "get_staking_rewards": return {"rewards": round(rng.uniform(0.001,5),4), "apr": round(rng.uniform(2,15),2)}
    if tool == "get_tax_lot": return {"lots": rng.randint(1,12), "unrealized": amt(rng,-5000,15000)}
    if tool == "list_open_orders": return {"open_count": rng.randint(0,8)}
    if tool == "get_account_statement": return {"opening": amt(rng,1000,30000), "closing": amt(rng,1000,30000)}
    return {"ok": True}

def make_single_turn(rng, tool):
    args = gen_args(rng, tool)
    user = gen_user(rng, tool, args)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
    ]}

def make_multi_turn(rng, tool, suffix):
    args = gen_args(rng, tool)
    user = gen_user(rng, tool, args)
    result = gen_tool_result(rng, tool, args)
    result_str = json.dumps(result)
    # construct followup
    final = f"{suffix} " + render_followup(rng, tool, args, result)
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": args}}]},
        {"role": "tool", "name": tool, "content": result_str},
        {"role": "assistant", "content": final},
    ]}

def render_followup(rng, tool, args, result):
    if "price" in result and "change_pct" in result:
        return f"{args['symbol']} is at ${result['price']}, {result['change_pct']}% on the day."
    if "change_24h" in result:
        return f"{args['symbol']} sits at ${result['price']}, {result['change_24h']}% over 24h."
    if "order_id" in result and "status" in result:
        return f"Order {result['order_id']} came back {result['status']}."
    if "tx_hash" in result:
        return f"Transfer initiated, tx {result['tx_hash'][:14]}..., status {result['status']}."
    if "balance" in result:
        return f"Balance is ${result['balance']}, ${result.get('available', result['balance'])} available."
    if "txn_id" in result:
        return f"Transaction {result['txn_id']} {result['status']}."
    if "score" in result:
        return f"Credit score {result['score']} from {result['bureau']}."
    if "rate" in result and "monthly" in result:
        return f"Quoted {result['rate']}% with ${result['monthly']}/month."
    if "apy" in result:
        return f"That product is paying {result['apy']}% APY."
    if "rate" in result and "converted" in result:
        return f"Got {result['converted']} at a rate of {result['rate']}."
    if "rate" in result:
        return f"Rate is {result['rate']}."
    if "case_id" in result:
        return f"Dispute opened, case {result['case_id']}."
    if "app_id" in result:
        return f"Loan application {result['app_id']} is {result['status'].replace('_',' ')}."
    if "card_id" in result:
        return f"Card {result['card_id']} is now {result['status'].replace('_',' ')}."
    if "invoice_id" in result:
        return f"Invoice {result['invoice_id']} saved as {result['status']}."
    if "je_id" in result:
        return f"Journal entry {result['je_id']} posted."
    if "expense_id" in result:
        return f"Expense {result['expense_id']} on the books."
    if "assets" in result:
        return f"Assets ${result['assets']}, liabilities ${result['liabilities']}."
    if "revenue" in result:
        return f"Revenue ${result['revenue']}, net income ${result['net_income']}."
    if "matched" in result:
        return f"{result['matched']} matched, {result['unmatched']} unmatched."
    if "value" in result:
        return f"Portfolio value ${result['value']}, cash ${result['cash']}."
    if "buying_power" in result:
        return f"Buying power ${result['buying_power']}."
    if "balance_eth" in result:
        return f"Wallet holds {result['balance_eth']} ETH."
    if "gwei" in result:
        return f"Gas is {result['gwei']} gwei."
    if "rewards" in result:
        return f"Earned {result['rewards']} so far at {result['apr']}% APR."
    if "lots" in result:
        return f"{result['lots']} lots, ${result['unrealized']} unrealized."
    if "open_count" in result:
        return f"{result['open_count']} open orders."
    if "matches" in result:
        return f"Screener found {result['matches']} matches."
    if "next_earnings" in result:
        return f"Next earnings on {result['next_earnings']}."
    if "pe" in result:
        return f"P/E {result['pe']}, EPS ${result['eps']}."
    if "total_paid" in result:
        return f"Total dividends paid ${result['total_paid']}."
    if "calls" in result:
        return f"{result['calls']} calls and {result['puts']} puts available."
    if "received" in result:
        return f"Got {result['received']} after a {result['fee_pct']}% fee."
    if "payoff" in result:
        return f"Payoff is ${result['payoff']} good through {result['good_through']}."
    if "request_id" in result:
        return f"Request {result['request_id']} {result['status']}."
    if "opening" in result:
        return f"Opening ${result['opening']}, closing ${result['closing']}."
    if "positions" in result:
        return f"{result['positions']} positions on the books."
    if "count" in result:
        return f"{result['count']} txns, ${result['total_out']} out."
    if "epoch" in result:
        return f"Status {result['status']} as of epoch {result['epoch']}."
    if "bid" in result:
        return f"Bid ${result['bid']} ask ${result['ask']} IV {result['iv']}."
    return "Result is above."

def main():
    rng = random.Random(SEED)
    samples = []

    # 15% single-turn = 75
    n_single = 75
    n_multi = 425

    # Single-turn: cycle through tools to keep distribution balanced
    tool_cycle = TOOLS[:]
    rng.shuffle(tool_cycle)
    for i in range(n_single):
        tool = tool_cycle[i % len(tool_cycle)]
        samples.append(make_single_turn(rng, tool))

    # Multi-turn: 425. Suffix coverage: 30 phrases. Aim ~14 each = 420, +5 extras.
    suffix_assignments = []
    for s in SUFFIX_POOL:
        suffix_assignments.extend([s]*14)  # 30*14=420
    # add 5 more to round to 425
    extras = rng.sample(SUFFIX_POOL, 5)
    suffix_assignments.extend(extras)
    rng.shuffle(suffix_assignments)

    # Tool distribution multi-turn: cycle for balance
    tool_cycle2 = TOOLS[:]
    rng.shuffle(tool_cycle2)
    for i in range(n_multi):
        tool = tool_cycle2[i % len(tool_cycle2)]
        suffix = suffix_assignments[i]
        samples.append(make_multi_turn(rng, tool, suffix))

    rng.shuffle(samples)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    tool_counter = Counter()
    single_turns = 0
    suffix_counter = Counter()
    BLACKLIST = ["I've gathered all the information","I've completed the task","Here's what I found:","Based on the results,","The results show that"]
    blacklist_hits = 0
    for s in samples:
        msgs = s["messages"]
        last = msgs[-1]
        if last["role"] == "assistant" and last.get("tool_calls"):
            single_turns += 1
            for tc in last["tool_calls"]:
                tool_counter[tc["function"]["name"]] += 1
        else:
            # multi-turn final assistant
            for m in msgs:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        tool_counter[tc["function"]["name"]] += 1
            content = last["content"]
            for sfx in SUFFIX_POOL:
                if content.startswith(sfx):
                    suffix_counter[sfx] += 1
                    break
            for bl in BLACKLIST:
                if bl in content:
                    blacklist_hits += 1
    total = len(samples)
    print(f"Total: {total}")
    print(f"Distinct tools: {len(tool_counter)}")
    print(f"Single-turn: {single_turns} ({100*single_turns/total:.1f}%)")
    print(f"Suffix coverage: {len(suffix_counter)}/30")
    print(f"Suffix min/max: {min(suffix_counter.values())}/{max(suffix_counter.values())}")
    print(f"Blacklist hits: {blacklist_hits}")
    top = tool_counter.most_common(3)
    print(f"Top tools: {top}")
    max_pct = 100*top[0][1]/total
    print(f"Max tool pct: {max_pct:.2f}%")

if __name__ == "__main__":
    main()
