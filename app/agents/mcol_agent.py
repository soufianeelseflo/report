import asyncio
import json
import datetime
import subprocess
import os
import re
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from lemonsqueezy import LemonSqueezy # Import Lemon Squeezy

# Corrected relative imports for package structure
from autonomous_agency.app.core.config import settings
from autonomous_agency.app.db import crud, models
from autonomous_agency.app.agents.agent_utils import get_httpx_client, call_llm_api # Use updated call_llm_api
# Ensure call_llm_api is robustly defined, assuming it's in agent_utils now
# try:
#     from autonomous_agency.app.agents.agent_utils import call_llm_api
# except ImportError:
#     print("Warning: Could not import call_llm_api from agent_utils, using fallback.")
#     async def call_llm_api(client: httpx.AsyncClient, prompt: str, model: str = "google/gemini-1.5-pro-latest") -> Optional[Dict[str, Any]]: return None


# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS
MCOL_IMPLEMENTATION_MODE = settings.MCOL_IMPLEMENTATION_MODE # "SUGGEST" or "ATTEMPT_EXECUTE"
WEBSITE_OUTPUT_DIR = "/app/static_website" # Matches Dockerfile and main.py

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    if not snapshot: return "No KPI data available."
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    lines.append(f"- Reports: AwaitingPayment={snapshot.awaiting_payment_reports}, Pending={snapshot.pending_reports}, Processing={snapshot.processing_reports}, Completed(24h)={snapshot.completed_reports_24h}, Failed(24h)={snapshot.failed_reports_24h}, AvgTime={snapshot.avg_report_time_seconds or 0:.2f}s")
    lines.append(f"- Prospecting: New(24h)={snapshot.new_prospects_24h}")
    lines.append(f"- Email: Sent(24h)={snapshot.emails_sent_24h}, ActiveAccounts={snapshot.active_email_accounts}, Deactivated(24h)={snapshot.deactivated_accounts_24h}, BounceRate(24h)={snapshot.bounce_rate_24h or 0:.2f}%")
    lines.append(f"- Revenue: Orders(24h)={snapshot.orders_created_24h}, Revenue(24h)=${snapshot.revenue_24h:.2f}")
    return "\n".join(lines)

async def analyze_performance_and_prioritize(client: httpx.AsyncClient, kpi_data_str: str) -> Optional[Dict[str, str]]:
    """Uses LLM to analyze KPIs, identify the biggest problem, and explain why."""
    primary_goal = "Achieve $10,000 revenue within 72 hours, then sustain growth via AI report sales ($499/$999) and autonomous client acquisition."
    system_context = f"""
    System Overview: Autonomous agency using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses 'open-deep-research', delivers via email), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation), MCOL (self-improvement). Deployed via Docker at {settings.AGENCY_BASE_URL}. Payment via Lemon Squeezy (checkout links + webhook). Website serving via FastAPI static files.
    """
    prompt = f"""
    Analyze the following system performance data for an autonomous AI reporting agency aiming for rapid revenue generation ($10k/72h).
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL bottleneck currently preventing the achievement of the primary goal ($10k in 72h). Consider the entire funnel: Prospecting -> Email Outreach -> Website Visit -> Order Attempt -> Payment Success -> Report Generation -> Delivery. Look at KPIs like Orders(24h), Revenue(24h), New Prospects, Email Sent/Bounce Rate, Pending Reports.
    2. Briefly explain the reasoning for selecting this problem (impact on revenue/growth).
    3. Respond ONLY with a JSON object containing "problem" and "reasoning".
    Example Problems: "Zero Orders Created (Website/Payment Issue?)", "Low Prospect Acquisition Rate", "High Email Bounce Rate (>20%)", "Payment Webhook Not Triggering Report Creation", "Report Generation Failing Frequently", "No Website Generated Yet".
    If Revenue(24h) is significantly positive and growing, identify the next major bottleneck to scaling. If all looks optimal, respond: {{"problem": "None", "reasoning": "Current KPIs indicate strong progress towards goal."}}
    """
    print("[MCOL] Analyzing KPIs with LLM...")
    llm_response = await call_llm_api(client, prompt)

    if llm_response and isinstance(llm_response, dict) and "problem" in llm_response and "reasoning" in llm_response:
        problem = llm_response["problem"].strip()
        reasoning = llm_response["reasoning"].strip()
        if problem != "None":
            print(f"[MCOL] Identified Priority Problem: {problem} (Reason: {reasoning})")
            return {"problem": problem, "reasoning": reasoning}
        else:
            print("[MCOL] LLM analysis indicates no critical problems currently, or goal is being met.")
            return None
    else:
        print("[MCOL] Failed to get valid analysis from LLM.")
        return None

async def generate_solution_strategies(client: httpx.AsyncClient, problem: str, reasoning: str, kpi_data_str: str) -> Optional[List[Dict[str, str]]]:
    """Uses LLM to generate potential solution strategies for the identified problem."""
    system_context = f"""
    System: FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (API Key, Variant IDs, Webhook Secret configured). Website: Served via FastAPI static files (target dir: /app/static_website) at {settings.AGENCY_BASE_URL}. Budget: $10/month (proxies). Uses free tiers aggressively.
    Code Structure: /app/autonomous_agency/app/ contains main.py, agents/, db/, core/, workers/, api/endpoints/. /app/open-deep-research/ contains the external tool. Migrations via Alembic. Static website files go in /app/static_website.
    """
    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate 2-3 diverse, actionable, and creative strategies to solve the identified problem, suitable for an autonomous AI agent to potentially implement or suggest. Prioritize strategies leveraging AI/automation, Lemon Squeezy integration, and adhering to extreme budget constraints. Focus on achieving the $10k/72h goal. Incorporate rule-bending, algorithm-aware tactics where applicable (e.g., for SEO, outreach).

    Instructions:
    1. For each strategy, provide a concise 'name' and a detailed 'description' outlining the steps involved.
    2. If code generation/modification is involved, specify the target file(s) and the nature of the change.
    3. If external interaction is needed (e.g., "Manual Lemon Squeezy product setup"), state it clearly.
    4. Output ONLY a JSON list of strategy objects. Each object must have "name" and "description" keys.

    Example Strategies (Refined & Aggressive):
    - Problem: "No Website Generated Yet" -> Strategy: {{"name": "Generate SEO & Conversion Optimized Website v1", "description": "Use LLM (Gemini 1.5 Pro) to generate `index.html` optimized for SEO keywords related to 'AI research reports', 'market analysis', 'competitive intelligence'. Include compelling copy, clear value prop, social proof placeholders, $499/$999 pricing + anchors, and order form JS posting to `/api/v1/payments/create-checkout` and redirecting. Save to `/app/static_website/index.html`."}}
    - Problem: "Zero Orders Created" -> Strategy: {{"name": "Aggressive Conversion Rate Optimization (CRO) & Debug", "description": "1. Verify website JS posts correctly to payment endpoint. 2. Check payment endpoint logs for errors. 3. Use LLM to analyze generated website HTML/JS for conversion bottlenecks (e.g., unclear CTA, slow load). 4. Suggest specific A/B tests for headlines/pricing presentation (MCOL logs suggestion). 5. Ensure Lemon Squeezy variants are correct. 6. Suggest manual test purchase."}}
    - Problem: "Payment Webhook Not Triggering Report Creation" -> Strategy: {{"name": "Debug Payment Webhook & DB Insertion", "description": "1. Verify LS webhook points to `{settings.AGENCY_BASE_URL}/api/v1/payments/webhook`. 2. Check `payments.py -> lemon_squeezy_webhook` logs for signature errors or processing failures. 3. Ensure webhook secret matches. 4. Verify `crud.py -> create_report_request_from_webhook` correctly parses payload/custom data and inserts into DB with 'PENDING' status. Check DB logs."}}
    - Problem: "Low Prospect Acquisition Rate" -> Strategy: {{"name": "Hyper-Targeted LinkedIn Outreach Prep", "description": "Modify 'ProspectResearcher': Use LLM to identify 2-3 specific, high-ranking executives (CEO, VP Marketing, Head of Strategy) at companies identified via signals. Store names/titles/LinkedIn URLs (if findable via search simulation) in `Prospect` table. MCOL Action: Log detailed suggestions for manual operator: 'Connect with [Name, Title] at [Company] on LinkedIn. Reference [Specific Signal/Pain Point]. Suggest discussing how our rapid AI reports address [Benefit].'"}}
    - Problem: "High Email Bounce Rate (>15%)" -> Strategy: {{"name": "Integrate Free Email Validation & Aggressive List Pruning", "description": "Research free tier email validation APIs (e.g., debounce.io, zerobounce - check current free limits). Use LLM to generate code modification for 'EmailMarketer -> process_email_batch' to call the chosen API before `generate_personalized_email`. If API flags email as invalid/risky, update prospect status to 'INVALID_EMAIL' and skip. Requires adding API key to .env."}}
    - Problem: "Low Website Traffic/SEO Ranking" -> Strategy: {{"name": "AI-Driven SEO Content Generation & Backlink Simulation", "description": "Use LLM (Gemini 1.5 Pro) to generate 2-3 high-quality blog posts relevant to 'AI market research', 'competitive analysis tools'. Save as HTML in `/app/static_website/blog/`. Modify `index.html` to link to them. MCOL Action: Simulate social sharing/backlinks by prompting LLM to generate realistic-sounding forum posts/comments mentioning the agency/blog posts (log these for potential manual posting)."}}
    """
    print(f"[MCOL] Generating strategies for problem: {problem}")
    # Use a powerful model capable of complex generation/analysis
    llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest") # Or your preferred powerful model

    strategies = None
    if llm_response:
        if isinstance(llm_response, list): strategies = llm_response
        elif isinstance(llm_response.get("strategies"), list): strategies = llm_response["strategies"]
        elif isinstance(llm_response.get("raw_inference"), str):
            try:
                cleaned = llm_response["raw_inference"].strip().replace("```json", "").replace("```", "").strip()
                parsed_raw = json.loads(cleaned)
                if isinstance(parsed_raw, list): strategies = parsed_raw
            except json.JSONDecodeError: print("[MCOL] LLM raw inference is not valid JSON list.")

    if strategies and all(isinstance(s, dict) and "name" in s and "description" in s for s in strategies):
        print(f"[MCOL] LLM generated {len(strategies)} strategies.")
        return strategies
    else:
        print(f"[MCOL] Failed to get valid strategies from LLM. Response: {llm_response}")
        return None

def choose_strategy(strategies: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Selects the best strategy (simple: pick the first one)."""
    if not strategies: return None
    print(f"[MCOL] Choosing strategy: {strategies[0]['name']}")
    return strategies[0]

async def implement_strategy(client: httpx.AsyncClient, strategy: Dict[str, str]) -> Dict[str, Any]:
    """Attempts to implement the chosen strategy by generating/modifying code or suggesting actions."""
    action_details = {"name": strategy["name"], "description": strategy["description"]}
    print(f"[MCOL] Processing strategy: {strategy['name']}")
    strategy_name = strategy.get("name", "").lower()

    # --- Define Implementation Logic per Strategy ---
    async def generate_website_files():
        print("[MCOL] Attempting to generate website files (index.html)...")
        website_prompt = f"""
        Generate a complete, single-file HTML document (`index.html`) for a professional agency website: "{settings.PROJECT_NAME}".
        Offer AI-powered research reports. Target audience: Businesses needing market/competitor insights fast.
        Sections:
        1. Hero: Headline: "Instant AI-Powered Research Reports: Actionable Insights in Hours, Not Weeks." Sub-headline: "Leverage cutting-edge AI for deep market analysis, competitor intelligence, and strategic foresight. Get started now." CTA Button: "Order Your Report".
        2. Features: Highlight Speed (Hours vs Weeks), Accuracy (AI precision + sources), Depth (Comprehensive analysis), Cost-Effectiveness (vs traditional consulting). Use icons (unicode emojis ok).
        3. Pricing: Clear side-by-side comparison:
           - Standard Report: $499 (Anchor: ~~$1497~~). Features: Core market/competitor overview, Key trends, Standard turnaround (~12h). Button: "Order Standard Report".
           - Premium Deep Dive: $999 (Anchor: ~~$2997~~). Features: Max depth/breadth, Extended analysis & recommendations, Priority turnaround (<6h), Raw data export option. Button: "Order Premium Report".
        4. Order Form: (id="report-order-form") Fields: Name (text), Email (email), Company Name (text, optional, id="company_name"), Report Type (select, id="report_type", options: value="standard_499" text="Standard Report ($499)", value="premium_999" text="Premium Deep Dive ($999)"), Research Topic/Details (textarea, required, id="request_details"). Submit Button (id="submit-order-btn", text="Proceed to Payment"). Message Div (id="form-message").
        5. Footer: Copyright {datetime.datetime.now().year} {settings.PROJECT_NAME}.
        CSS: Use a clean, modern style block. Professional fonts (sans-serif), blue/gray/white color scheme. Ensure responsiveness.
        JavaScript: In <script> block:
           - On form submit: Prevent default. Show "Processing..." message, disable button. Get form values (name, email, company, report_type, request_details).
           - Use `fetch` to POST JSON {{"report_type": ..., "client_email": ..., "client_name": ..., "company_name": ..., "request_details": ...}} to `/api/v1/payments/create-checkout`.
           - On success (status 201): Parse JSON response, get `checkout_url`, redirect (`window.location.href = checkout_url`).
           - On error: Show error in message div, re-enable button.
        Output ONLY the raw, complete HTML code for `index.html`. No explanations.
        """
        html_code_response = await call_llm_api(client, website_prompt, model="google/gemini-1.5-pro-latest") # Use powerful model
        if html_code_response and isinstance(html_code_response.get("raw_inference"), str):
            code = html_code_response["raw_inference"].strip()
            # More robust check
            if code.startswith("<!DOCTYPE html>") and code.endswith("</html>") and "report-order-form" in code and "/api/v1/payments/create-checkout" in code:
                try:
                    os.makedirs(WEBSITE_OUTPUT_DIR, exist_ok=True)
                    filepath = os.path.join(WEBSITE_OUTPUT_DIR, "index.html")
                    with open(filepath, "w", encoding="utf-8") as f: f.write(code)
                    print(f"[MCOL] Successfully generated and saved website file: {filepath}")
                    return {"status": "COMPLETED", "result": f"Generated website file: {filepath}"}
                except Exception as e:
                    print(f"[MCOL] Failed to write website file: {e}")
                    return {"status": "FAILED", "result": f"Failed to write website file: {e}"}
            else:
                print("[MCOL] LLM generated invalid HTML structure or content.")
                return {"status": "FAILED", "result": f"LLM generated invalid HTML. Output preview: {code[:500]}"}
        else:
            print("[MCOL] Failed to generate website HTML from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate website HTML."}

    async def generate_payment_endpoint():
        print("[MCOL] Generating Lemon Squeezy payment endpoint code suggestion...")
        payment_prompt = f"""
        Generate the complete Python code for the file `autonomous_agency/app/api/endpoints/payments.py`.
        This file should define a FastAPI APIRouter.
        Include an endpoint `POST /create-checkout`:
        - Accepts JSON body: `report_type` (str: 'standard_499' or 'premium_999'), `client_email` (str), `client_name` (str, optional), `company_name` (str, optional), `request_details` (str).
        - Uses the `lemonsqueezy.py` library. Initialize client `ls = LemonSqueezy(api_key=settings.LEMONSQUEEZY_API_KEY)`.
        - Maps `report_type` to `settings.LEMONSQUEEZY_VARIANT_STANDARD` or `settings.LEMONSQUEEZY_VARIANT_PREMIUM`. Raise 400 if invalid.
        - Creates a payload for Lemon Squeezy `POST /v1/checkouts` API.
        - Include `checkout_data` with `email`, `name`.
        - Include `custom_data` containing `research_topic` (from `request_details`), `company_name`, `client_name`, `report_type`.
        - Set `redirect_url` using `settings.AGENCY_BASE_URL + '/order-success?session_id={{CHECKOUT_SESSION_ID}}'`.
        - Use `httpx.AsyncClient` (via dependency `get_ls_client`) to call the Lemon Squeezy API.
        - Return `{"checkout_url": url}` on success (status 201). Handle LS API errors gracefully (status 502).
        Include another endpoint `POST /webhook`:
        - Verifies the `X-Signature` header using `settings.LEMONSQUEEZY_WEBHOOK_SECRET`, `hmac`, and `hashlib.sha256`. Raise 400 on failure.
        - Parses the JSON request body.
        - If `meta.event_name` is `order_created`:
            - Call `crud.create_report_request_from_webhook(db, order_data)` using the `data` part of the webhook payload.
            - Commit the DB session on success. Rollback on error.
        - Return 200 OK, even on processing errors (log them) to prevent excessive retries.
        Include all necessary imports: FastAPI, Depends, HTTPException, Request, Header, AsyncSession, httpx, hmac, hashlib, json, Optional, BaseModel, Field, lemonsqueezy, settings, crud, models, get_db_session.
        Output ONLY the raw, complete Python code for the `payments.py` file. No explanations.
        """
        code_response = await call_llm_api(client, payment_prompt, model="google/gemini-1.5-pro-latest")
        if code_response and isinstance(code_response.get("raw_inference"), str):
            code_snippet = code_response["raw_inference"].strip()
            if "APIRouter()" in code_snippet and "create_lemon_squeezy_checkout" in code_snippet and "lemon_squeezy_webhook" in code_snippet:
                 # Suggest creation/replacement
                 target_file = "autonomous_agency/app/api/endpoints/payments.py"
                 print(f"[MCOL] SUGGESTION: Create/Replace '{target_file}' with the following code. Also ensure it's included in 'main.py' router.\n```python\n{code_snippet}\n```")
                 return {"status": "SUGGESTED", "result": f"Code generated for {target_file}. Requires manual review/placement and adding router to main.py."}
            else:
                 print("[MCOL] LLM generated incomplete/invalid code for payments endpoint.")
                 return {"status": "FAILED", "result": f"LLM generated invalid code. Preview: {code_snippet[:500]}"}
        else:
            print("[MCOL] Failed to generate payment endpoint code from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate payment endpoint code."}

    async def generate_report_delivery_code():
        print("[MCOL] Generating report delivery code modification suggestion...")
        delivery_prompt = """
        Generate the Python code snippet to add email delivery with attachment to the `process_single_report_request` function in `autonomous_agency/app/agents/report_generator.py`.
        Goal: After the report is confirmed generated (`final_status == "COMPLETED"`), send an email to `request.client_email` with the report file attached.
        Instructions:
        1. Define an internal async helper function `_send_delivery_email(db: AsyncSession, request: models.ReportRequest, report_path: str)` within `report_generator.py`.
        2. Inside `_send_delivery_email`:
           - Get an active sending account via `crud.get_active_email_account_for_sending(db)`. Handle None case.
           - Decrypt password using `decrypt_data`. Handle failure.
           - Create `EmailMessage`. Set Subject, From, To, Date, Message-ID.
           - Set plain text body.
           - Use `mimetypes.guess_type` and `msg.add_attachment` to attach the file at `report_path`. Handle file not found or attachment errors gracefully (e.g., send email without attachment but mention error).
           - Use `aiosmtplib.SMTP` to connect, login, send. Handle SMTP exceptions.
           - If send succeeds, call `crud.increment_email_sent_count` and commit. Return True/False.
        3. In `process_single_report_request`, after the final DB update and commit:
           - Check if `process_success` is True and `final_status == "COMPLETED"`.
           - If so, call `asyncio.create_task(_send_delivery_email(db, updated_request, output_path))` to run delivery in the background.
        4. Add necessary imports at the top: `asyncio`, `mimetypes`, `EmailMessage`, `formatdate`, `make_msgid`, `aiosmtplib`, `crud`, `decrypt_data`, `models`.
        Output ONLY the raw Python code for the `_send_delivery_email` function AND the modification part within `process_single_report_request` to call it. Clearly mark where to insert the call. Include necessary imports.
        """
        code_response = await call_llm_api(client, delivery_prompt, model="google/gemini-1.5-pro-latest")
        if code_response and isinstance(code_response.get("raw_inference"), str):
            code_snippet = code_response["raw_inference"].strip()
            if "_send_delivery_email" in code_snippet and "asyncio.create_task" in code_snippet:
                target_file = "autonomous_agency/app/agents/report_generator.py"
                print(f"[MCOL] SUGGESTION: Modify '{target_file}'. Add necessary imports. Add the `_send_delivery_email` function. Add the `asyncio.create_task` call after the final DB commit inside `process_single_report_request`.\n```python\n{code_snippet}\n```")
                return {"status": "SUGGESTED", "result": f"Code modification suggested for {target_file} to implement report delivery."}
            else:
                 print("[MCOL] LLM generated incomplete/invalid code for report delivery.")
                 return {"status": "FAILED", "result": f"LLM generated invalid code. Preview: {code_snippet[:500]}"}
        else:
            print("[MCOL] Failed to generate report delivery code modification from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate report delivery code."}

    async def suggest_linkedin_actions():
        # ... (Implementation remains the same - suggests manual action) ...
        print("[MCOL] Generating suggestions for manual LinkedIn actions...")
        result_text = "Suggest operator manually review prospects with identified executives in DB (Prospect.key_executives). Craft personalized LinkedIn connection requests referencing recent signals/pain points."
        print(f"[MCOL] {result_text}")
        return {"status": "SUGGESTED", "result": result_text}

    # --- Strategy Execution Mapping ---
    implementation_result = {"status": "FAILED", "result": "Strategy not recognized or executable."}

    # Prioritize critical path: Website -> Payments -> Delivery
    if "website" in strategy_name and ("generate" in strategy_name or "seo" in strategy_name):
        # Always try to execute website generation if needed
        implementation_result = await generate_website_files()
    elif "payment" in strategy_name and ("implement" in strategy_name or "debug" in strategy_name):
         print("[MCOL] WARNING: Lemon Squeezy strategy requires manual product setup in Lemon Squeezy dashboard first.")
         implementation_result = await generate_payment_endpoint() # Returns SUGGESTED status
    elif "report delivery" in strategy_name:
        implementation_result = await generate_report_delivery_code() # Returns SUGGESTED status
    elif "linkedin prospecting" in strategy_name:
         implementation_result = await suggest_linkedin_actions()
    # Add more handlers...
    else:
         implementation_result = {"status": "SUGGESTED", "result": f"Suggest manually implementing strategy: {strategy['name']}. Desc: {strategy['description']}"}


    print(f"[MCOL] Implementation outcome for '{strategy['name']}': {implementation_result['status']} - {implementation_result['result']}")
    return {
        "status": implementation_result["status"],
        "result": implementation_result["result"],
        "parameters": action_details
    }


# --- Main MCOL Agent Cycle ---
async def run_mcol_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single Monitor -> Analyze -> Strategize -> Implement -> Verify cycle."""
    if shutdown_event.is_set(): return
    print(f"[MCOL] Starting cycle at {datetime.datetime.now(datetime.timezone.utc)}")
    client: Optional[httpx.AsyncClient] = None
    current_snapshot: Optional[models.KpiSnapshot] = None
    decision_log_id: Optional[int] = None

    try:
        client = await get_httpx_client()

        # 1. Monitor: Create KPI Snapshot
        current_snapshot = await crud.create_kpi_snapshot(db)
        await db.commit()
        kpi_str = format_kpis_for_llm(current_snapshot)
        print(f"[MCOL] {kpi_str}")

        # 2. Analyze & Prioritize
        analysis = await analyze_performance_and_prioritize(client, kpi_str)
        if not analysis:
            print("[MCOL] No critical problems identified or analysis failed. Ending cycle.")
            if client: await client.aclose()
            return

        decision = await crud.log_mcol_decision(
            db, kpi_snapshot_id=current_snapshot.snapshot_id,
            priority_problem=analysis["problem"], analysis_summary=analysis["reasoning"],
            action_status='ANALYZED'
        )
        await db.commit()
        decision_log_id = decision.log_id

        # 3. Strategize
        strategies = await generate_solution_strategies(client, analysis["problem"], analysis["reasoning"], kpi_str)
        if not strategies:
            await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY')
            await db.commit()
            print("[MCOL] Failed to generate strategies. Ending cycle.")
            if client: await client.aclose()
            return

        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=strategies) # Store list directly if DB field is JSON
        await db.commit()

        # 4. Choose Strategy
        chosen_strategy = choose_strategy(strategies)
        if not chosen_strategy:
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY')
             await db.commit()
             if client: await client.aclose()
             return

        await crud.update_mcol_decision_log(db, decision_log_id, chosen_action=chosen_strategy['name'])
        await db.commit()

        # 5. Implement (Suggest or Attempt Execute)
        implementation_result = await implement_strategy(client, chosen_strategy)

        await crud.update_mcol_decision_log(
            db, decision_log_id,
            action_status=implementation_result["status"],
            action_result=implementation_result["result"],
            action_parameters=implementation_result.get("parameters")
        )
        await db.commit()

        print(f"[MCOL] Cycle finished. Action status: {implementation_result['status']}")

    except Exception as e:
        print(f"[MCOL] CRITICAL Error during MCOL cycle: {e}")
        import traceback
        traceback.print_exc()
        if decision_log_id:
            try:
                await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {traceback.format_exc()}")
                await db.commit()
            except Exception as db_err:
                 print(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                 await db.rollback()
        else:
             await db.rollback()
    finally:
        if client:
            await client.aclose()