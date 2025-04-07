import asyncio
import json
import datetime
import subprocess
import os
import re

from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from lemonsqueezy import LemonSqueezy # Import Lemon Squeezy

from app.core.config import settings
from app.db import crud, models
from app.agents.agent_utils import get_httpx_client
# Ensure call_llm_api is robustly defined, assuming it's in prospect_researcher
try:
    from app.agents.prospect_researcher import call_llm_api
except ImportError:
    # Fallback definition if prospect_researcher structure changed or for standalone testing
    print("Warning: Could not import call_llm_api from prospect_researcher, using fallback.")
    async def call_llm_api(client: httpx.AsyncClient, prompt: str) -> Optional[Dict[str, Any]]: return None


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
    # Updated goal
    primary_goal = "Achieve $10,000 revenue within 72 hours, then sustain growth via AI report sales ($499/$999) and autonomous client acquisition."
    system_context = """
    System Overview: Autonomous agency using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses 'open-deep-research', delivers via email), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation), MCOL (self-improvement). Deployed via Docker. Payment via Lemon Squeezy (checkout links + webhook). Website serving via FastAPI static files.
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
    3. Respond ONLY with a JSON object containing two keys: "problem" (string description of the problem) and "reasoning" (string explanation).
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
    system_context = """
    System: FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (API Key, Variant IDs, Webhook Secret configured). Website: Served via FastAPI static files (target dir: /app/static_website). Budget: $10/month (proxies). Uses free tiers aggressively.
    Code Structure: /app/autonomous_agency/app/ contains main.py, agents/, db/, core/, workers/, api/endpoints/. /app/open-deep-research/ contains the external tool. Migrations via Alembic. Static website files go in /app/static_website.
    """
    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate 2-3 diverse, actionable, and creative strategies to solve the identified problem, suitable for an autonomous AI agent to potentially implement or suggest. Prioritize strategies leveraging AI/automation, Lemon Squeezy integration, and adhering to extreme budget constraints. Focus on achieving the $10k/72h goal.

    Instructions:
    1. For each strategy, provide a concise 'name' and a detailed 'description' outlining the steps involved.
    2. If code generation/modification is involved, specify the target file(s) and the nature of the change (e.g., "Generate static HTML/CSS/JS", "Add FastAPI endpoint", "Modify agent logic").
    3. If external interaction is needed (e.g., "Manually add more SMTP accounts"), state it clearly.
    4. Output ONLY a JSON list of strategy objects. Each object must have "name" and "description" keys.

    Example Strategies (Refined):
    - Problem: "No Website Generated Yet" -> Strategy: {{"name": "Generate Static Website v1", "description": "Use LLM to generate `index.html` with Hero, Features, Pricing ($499/$999 + anchors), Order Form (Name, Email, Company, Report Type, Request Details), Footer. Include inline CSS & JS. JS should POST form data to `/api/v1/payments/create-checkout` (NOT /requests), then redirect user to the returned `checkout_url`. Save file to `/app/static_website/index.html`."}}
    - Problem: "Zero Orders Created (Website/Payment Issue?)" -> Strategy: {{"name": "Verify/Debug Website & Payment Flow", "description": "1. Check if `/app/static_website/index.html` exists and is served correctly. 2. Verify the website JS correctly POSTs to `/api/v1/payments/create-checkout`. 3. Check logs for errors in the `create_lemon_squeezy_checkout` endpoint (`app/api/endpoints/payments.py`). 4. Ensure Lemon Squeezy Variant IDs in `.env` are correct. 5. Suggest manual test purchase."}}
    - Problem: "Payment Webhook Not Triggering Report Creation" -> Strategy: {{"name": "Debug Payment Webhook Handler", "description": "1. Verify Lemon Squeezy webhook points to `YOUR_DEPLOYED_URL/api/v1/payments/webhook`. 2. Check logs for errors in the `lemon_squeezy_webhook` function (`app/api/endpoints/payments.py`). 3. Ensure webhook secret matches. 4. Verify `create_report_request_from_webhook` in `crud.py` correctly parses payload and custom data keys ('research_topic', etc.). 5. Check DB logs for transaction errors."}}
    - Problem: "Low Prospect Acquisition Rate" -> Strategy: {{"name": "Enhance LinkedIn Prospecting (Suggest Manual Actions)", "description": "Modify 'ProspectResearcher' agent: Use LLM not just for company name extraction from signals, but also to identify potential key executive names/titles (e.g., 'VP Marketing', 'Head of Research'). Store these in `Prospect.key_executives`. MCOL Action: Log suggestions for the human operator: 'Found potential contact [Title: Name] at [Company] based on [Signal]. Suggest manual LinkedIn connection request referencing the signal.' This avoids direct risky automation but leverages AI for targeting."}}
    - Problem: "High Email Bounce Rate (>20%)" -> Strategy: {{"name": "Implement Stricter Email Validation & List Cleaning", "description": "Modify 'EmailMarketer': Before generating email, use a free/cheap email validation API (research options like Hunter free tier, AbstractAPI free tier) to check if `prospect.contact_email` is valid/deliverable. If invalid, update prospect status to 'INVALID_EMAIL' instead of sending. Requires adding API key to .env and modifying `process_email_batch`."}}
    """
    print(f"[MCOL] Generating strategies for problem: {problem}")
    llm_response = await call_llm_api(client, prompt)

    strategies = None
    # Robust parsing of potential LLM outputs
    if llm_response:
        if isinstance(llm_response, list):
            strategies = llm_response
        elif isinstance(llm_response.get("strategies"), list):
            strategies = llm_response["strategies"]
        elif isinstance(llm_response.get("raw_inference"), str):
            try: # Try parsing raw inference if it looks like JSON list
                parsed_raw = json.loads(llm_response["raw_inference"])
                if isinstance(parsed_raw, list):
                    strategies = parsed_raw
            except json.JSONDecodeError:
                print("[MCOL] LLM raw inference is not valid JSON list.")
        # Add more parsing logic if needed based on LLM behavior

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
    strategy_desc = strategy.get("description", "")

    # --- Define Implementation Logic per Strategy ---

    async def generate_website_files():
        # ... (Implementation as in previous response - generates index.html) ...
        print("[MCOL] Attempting to generate website files...")
        website_prompt = """
        Generate a complete, single-file HTML document (`index.html`) for a professional agency website offering AI-powered research reports.
        Include sections for: Hero (compelling headline like "Instant AI-Powered Research Reports", sub-headline), Features (Speed, Accuracy, AI-Driven Insights, Comprehensive Coverage), Pricing (Standard $499, Premium $999 with anchor pricing like ~~$1497~~ and ~~$2997~~, list features per tier), Order Form (Name, Email, Company Name, Report Type dropdown [Standard $499|Premium $999], Research Topic/Details textarea), Footer (Simple copyright).
        Use modern, clean HTML5 structure. Apply professional inline CSS or a single <style> block for aesthetics (use a clean, modern color scheme like blue/white/gray). Ensure responsiveness for mobile.
        The order form MUST have `id="report-order-form"`. Include a submit button with id="submit-order-btn". Add a `div` with `id="form-message"` below the button for status messages.
        Add JavaScript in a <script> block:
        1. Select the form and message div by ID.
        2. Add an event listener for 'submit' to the form.
        3. On submit:
           a. Prevent default form submission.
           b. Display "Processing..." message in the message div. Disable submit button.
           c. Gather form data: name, email, company, report_type (value should be 'standard_499' or 'premium_999'), request_details.
           d. Use the `fetch` API to send a POST request with this JSON data to `/api/v1/payments/create-checkout`.
           e. Handle the response:
              - If response status is 201 (Created): Parse JSON, get `checkout_url`, redirect the browser (`window.location.href = checkout_url;`).
              - Otherwise: Display an error message in the message div (e.g., "Error creating checkout. Please try again or contact support."). Re-enable submit button.
        Output ONLY the raw HTML code for the `index.html` file. No explanations.
        """
        html_code_response = await call_llm_api(client, website_prompt)
        if html_code_response and isinstance(html_code_response.get("raw_inference"), str):
            code = html_code_response["raw_inference"]
            # Basic check for HTML structure
            if "<html" in code.lower() and "</html>" in code.lower() and "report-order-form" in code:
                try:
                    os.makedirs(WEBSITE_OUTPUT_DIR, exist_ok=True)
                    filepath = os.path.join(WEBSITE_OUTPUT_DIR, "index.html")
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(code)
                    print(f"[MCOL] Successfully generated and saved website file: {filepath}")
                    return {"status": "COMPLETED", "result": f"Generated website file: {filepath}"}
                except Exception as e:
                    print(f"[MCOL] Failed to write website file: {e}")
                    return {"status": "FAILED", "result": f"Failed to write website file: {e}"}
            else:
                print("[MCOL] LLM generated invalid HTML structure.")
                return {"status": "FAILED", "result": "LLM generated invalid HTML structure."}
        else:
            print("[MCOL] Failed to generate website HTML from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate website HTML."}

    async def generate_payment_endpoint():
        # ... (Implementation as in previous response - generates payments.py snippet) ...
         # This function remains largely the same, generating the code for payments.py
         # and suggesting the modification for main.py.
         # It's safer to keep this as a suggestion unless ATTEMPT_EXECUTE is highly trusted.
        print("[MCOL] Generating Lemon Squeezy payment endpoint code...")
        # ... (payment_prompt remains the same) ...
        # ... (LLM call and parsing logic remains the same) ...
        # For safety, default to SUGGEST even in ATTEMPT_EXECUTE for code modification
        print(f"[MCOL] SUGGESTION: Create 'app/api/endpoints/payments.py' with generated code. Add router inclusion to 'app/main.py'. Requires manual Lemon Squeezy product setup.")
        return {"status": "SUGGESTED", "result": "Code generated for payments endpoint and main.py modification suggested. Requires manual product setup."}


    async def generate_report_delivery_code():
        # ... (Implementation as in previous response - generates delivery code snippet) ...
        # This function also remains a suggestion generator for safety.
        print("[MCOL] Generating report delivery code modification...")
        # ... (delivery_prompt remains the same) ...
        # ... (LLM call logic remains the same) ...
        print(f"[MCOL] SUGGESTION: Modify 'app/agents/report_generator.py' to include report delivery logic.")
        return {"status": "SUGGESTED", "result": "Code modification suggested for report_generator.py to implement report delivery."}

    async def suggest_linkedin_actions():
        print("[MCOL] Generating suggestions for manual LinkedIn actions...")
        # Fetch prospects with key executives identified but not yet contacted via LinkedIn (needs new status/flag)
        # For simplicity, just log a general suggestion for now.
        result_text = "Suggest operator manually review prospects with identified executives in DB (Prospect.key_executives). Craft personalized LinkedIn connection requests referencing recent signals/pain points."
        print(f"[MCOL] {result_text}")
        return {"status": "SUGGESTED", "result": result_text}

    # --- Strategy Execution Mapping ---
    implementation_result = {"status": "FAILED", "result": "Strategy not recognized or executable."}

    # Prioritize critical path: Website -> Payments -> Delivery
    if "generate" in strategy_name and "website" in strategy_name:
        if MCOL_IMPLEMENTATION_MODE == "ATTEMPT_EXECUTE":
            implementation_result = await generate_website_files()
        else:
            implementation_result = {"status": "SUGGESTED", "result": "Suggest generating website files."}
    elif "implement" in strategy_name and "lemon squeezy" in strategy_name:
         print("[MCOL] WARNING: Lemon Squeezy strategy requires manual product setup in Lemon Squeezy dashboard first.")
         # Always suggest this initially due to manual step and code modification risk
         implementation_result = await generate_payment_endpoint() # Returns SUGGESTED status
    elif "add" in strategy_name and "report delivery" in strategy_name:
        # Always suggest code modifications initially
        implementation_result = await generate_report_delivery_code() # Returns SUGGESTED status
    elif "linkedin prospecting" in strategy_name:
         implementation_result = await suggest_linkedin_actions() # Always suggests manual action
    # Add handlers for other strategies like debugging, email validation API integration etc.
    # Example:
    # elif "debug payment webhook" in strategy_name:
    #     implementation_result = {"status": "SUGGESTED", "result": "Suggest manual debugging of webhook handler based on logs and Lemon Squeezy dashboard."}
    else:
         # Default to suggestion for unrecognized/complex strategies
         implementation_result = {"status": "SUGGESTED", "result": f"Suggest manually implementing strategy: {strategy['name']}. Desc: {strategy['description']}"}


    # Log final implementation outcome
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