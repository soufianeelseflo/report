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
from app.agents.prospect_researcher import call_llm_api # Reuse LLM utility

# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS
MCOL_IMPLEMENTATION_MODE = settings.MCOL_IMPLEMENTATION_MODE # "SUGGEST" or "ATTEMPT_EXECUTE"
WEBSITE_OUTPUT_DIR = "/app/static_website" # Matches Dockerfile and main.py

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    if not snapshot: return "No KPI data available."
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    lines.append(f"- Reports: Pending={snapshot.pending_reports}, Processing={snapshot.processing_reports}, Completed(24h)={snapshot.completed_reports_24h}, Failed(24h)={snapshot.failed_reports_24h}, AvgTime={snapshot.avg_report_time_seconds or 0:.2f}s")
    lines.append(f"- Prospecting: New(24h)={snapshot.new_prospects_24h}")
    lines.append(f"- Email: Sent(24h)={snapshot.emails_sent_24h}, ActiveAccounts={snapshot.active_email_accounts}, Deactivated(24h)={snapshot.deactivated_accounts_24h}, BounceRate(24h)={snapshot.bounce_rate_24h or 0:.2f}%")
    lines.append(f"- Revenue(24h): ${snapshot.revenue_24h:.2f}") # Will be 0 until payment integrated
    return "\n".join(lines)

async def analyze_performance_and_prioritize(client: httpx.AsyncClient, kpi_data_str: str) -> Optional[Dict[str, str]]:
    """Uses LLM to analyze KPIs, identify the biggest problem, and explain why."""
    # Updated goal
    primary_goal = "Achieve $10,000 revenue within 72 hours, then sustain growth via AI report sales ($499/$999) and autonomous client acquisition."
    system_context = """
    System Overview: Autonomous agency using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses external 'open-deep-research' tool), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation), MCOL (self-improvement). Deployed via Docker. Payment via Lemon Squeezy (planned). Website serving via FastAPI (planned).
    """
    prompt = f"""
    Analyze the following system performance data for an autonomous AI reporting agency aiming for rapid revenue generation.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL bottleneck currently preventing the achievement of the primary goal ($10k in 72h). Consider the entire funnel: client acquisition -> website interaction -> order placement -> payment -> report generation -> delivery.
    2. Briefly explain the reasoning for selecting this problem (impact on revenue/growth).
    3. Respond ONLY with a JSON object containing two keys: "problem" (string description of the problem) and "reasoning" (string explanation).
    Example Problems: "No Website for Order Intake", "Payment Processing Not Implemented", "Zero Report Requests Received", "Low Prospect-to-Email Conversion", "High Email Bounce Rate", "Report Generation Failing".
    If KPIs look healthy AND revenue is tracking towards the goal, respond: {{"problem": "None", "reasoning": "Current KPIs indicate progress towards goal."}}
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
            return None # No problem identified
    else:
        print("[MCOL] Failed to get valid analysis from LLM.")
        return None

async def generate_solution_strategies(client: httpx.AsyncClient, problem: str, reasoning: str, kpi_data_str: str) -> Optional[List[Dict[str, str]]]:
    """Uses LLM to generate potential solution strategies for the identified problem."""
    system_context = """
    System: FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (API Key available). Website: Served via FastAPI static files (target dir: /app/static_website). Budget: $10/month (proxies). Uses free tiers aggressively.
    Code Structure: /app/autonomous_agency/app/ contains main.py, agents/, db/, core/, workers/. /app/open-deep-research/ contains the external tool. Migrations via Alembic. Static website files go in /app/static_website.
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
    3. If external interaction is needed (e.g., "Manual Lemon Squeezy product setup"), state it clearly.
    4. Output ONLY a JSON list of strategy objects. Each object must have "name" and "description" keys.

    Example Strategies based on Potential Problems:
    - Problem: "No Website for Order Intake" -> Strategy: {{"name": "Generate Basic Static Website", "description": "Use LLM to generate 3 static files (index.html, pricing.html, order.html) with professional copy, $499/$999 pricing, anchor pricing, and an order form. Style with basic CSS. Order form JS should POST to '/api/v1/requests'. Save files to '/app/static_website'. FastAPI in 'main.py' already configured to serve this directory."}}
    - Problem: "Payment Processing Not Implemented" -> Strategy: {{"name": "Implement Lemon Squeezy Checkout Links", "description": "Requires manual setup of $499 and $999 products in Lemon Squeezy first. Then, use LLM to generate a FastAPI endpoint (e.g., in a new 'app/api/endpoints/payments.py' and include router in 'main.py') that takes a report_type ('standard_499' or 'premium_999') and uses the 'lemonsqueezy.py' library and API key to create a checkout link for the corresponding product. Modify the generated website's order button JS to call this endpoint and redirect the user to the checkout URL."}}
    - Problem: "No Report Delivery" -> Strategy: {{"name": "Add Automated Report Delivery Email", "description": "Modify 'app/agents/report_generator.py'. In 'process_single_report_request', after status is set to 'COMPLETED' and output_path exists, add logic using 'aiosmtplib' (similar to 'email_marketer.py') to fetch an active EmailAccount, decrypt password, and send an email to 'request.client_email' with a success message and attaching the report file from 'output_path'."}}
    - Problem: "Low Prospect Email Open/Reply Rate" -> Strategy: {{"name": "Refine Email Generation Prompt & A/B Test", "description": "Modify the LLM prompt in 'app/agents/email_marketer.py -> generate_personalized_email'. Experiment with different tones, CTAs, or personalization angles. Implement simple A/B testing by generating two versions of the prompt/email and tracking reply rates per version (requires adding a version field to Prospect or logging)."}}
    - Problem: "Contact Acquisition Rate Zero" -> Strategy: {{"name": "Explore LinkedIn via LLM Analysis (Indirect)", "description": "Since direct automation is risky: Modify 'ProspectResearcher' to search for company news/signals AND identify key executive names/titles using LLM. Log these names. MCOL can then generate highly personalized connection request drafts or profile analysis summaries for *manual* use on LinkedIn by the operator, focusing on executives found via signals."}}
    """
    print(f"[MCOL] Generating strategies for problem: {problem}")
    llm_response = await call_llm_api(client, prompt)

    strategies = None
    if llm_response and isinstance(llm_response, list):
        strategies = llm_response
    elif llm_response and isinstance(llm_response.get("strategies"), list):
        strategies = llm_response["strategies"]

    if strategies and all(isinstance(s, dict) and "name" in s and "description" in s for s in strategies):
        print(f"[MCOL] LLM generated {len(strategies)} strategies.")
        return strategies
    else:
        print("[MCOL] Failed to get valid strategies from LLM.")
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
        print("[MCOL] Attempting to generate website files...")
        # Simplified: Generate index.html only first
        website_prompt = """
        Generate a complete, single-file HTML document (`index.html`) for a professional agency website offering AI-powered research reports.
        Include sections for: Hero (compelling headline, sub-headline), Features (list key benefits like speed, accuracy, AI-power), Pricing (Standard $499, Premium $999 with anchor pricing like ~~$1497~~ and ~~$2997~~), Order Form (Name, Email, Company, Report Type dropdown, Request Details textarea), Footer.
        Use modern, clean HTML5 structure. Apply professional inline CSS or a single <style> block for aesthetics (no external CSS files).
        The order form MUST have `id="report-order-form"`. Include a submit button.
        Add basic JavaScript in a <script> block:
        1. Select the form by its ID.
        2. Add an event listener for 'submit'.
        3. On submit: prevent default form submission, gather form data (name, email, company, report_type, request_details) into a JSON object.
        4. Use the `fetch` API to send a POST request with the JSON data to `/api/v1/requests`.
        5. Handle the response: Show a success message (e.g., in a div with id="form-message") if status is 202, or an error message otherwise.
        Output ONLY the raw HTML code for the `index.html` file. No explanations.
        """
        html_code = await call_llm_api(client, website_prompt)
        if html_code and isinstance(html_code.get("raw_inference"), str):
            code = html_code["raw_inference"]
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
            print("[MCOL] Failed to generate website HTML from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate website HTML."}

    async def generate_payment_endpoint():
        print("[MCOL] Attempting to generate Lemon Squeezy payment endpoint...")
        payment_prompt = """
        Generate a Python code snippet for a FastAPI API endpoint to create Lemon Squeezy checkout links.
        Endpoint details:
        - Path: `/api/v1/payments/create-checkout` (POST method)
        - Request Body Schema (Pydantic): `report_type` (string, e.g., 'standard_499' or 'premium_999'), `client_email` (string, optional).
        - Logic:
            - Map `report_type` to Lemon Squeezy product variant IDs (use placeholders like 'STANDARD_VARIANT_ID' and 'PREMIUM_VARIANT_ID' - these need manual replacement or config).
            - Use the `lemonsqueezy.py` library (assume `ls = LemonSqueezy(api_key=settings.LEMONSQUEEZY_API_KEY)` is initialized).
            - Call `ls.create_checkout()` with the correct `store_id` (from `settings.LEMONSQUEEZY_STORE_ID`), `variant_id`, and prefill `checkout_data={'email': client_email}` if email is provided.
            - Return the `checkout_url` from the response in a JSON object `{"checkout_url": url}`.
        - Include necessary imports (FastAPI, Pydantic, LemonSqueezy, settings).
        - Structure as an APIRouter in a file `app/api/endpoints/payments.py`.
        - Also generate the code needed in `app/main.py` to include this new router.
        Output ONLY the raw Python code for `payments.py` and the modification for `main.py`, clearly separated.
        """
        code_response = await call_llm_api(client, payment_prompt)
        if code_response and isinstance(code_response.get("raw_inference"), str):
            raw_code = code_response["raw_inference"]
            # Basic parsing - needs improvement for robustness
            try:
                payments_py_code = re.search(r"# payments\.py START(.*?)# payments\.py END", raw_code, re.DOTALL | re.IGNORECASE).group(1)
                main_py_mod = re.search(r"# main\.py mod START(.*?)# main\.py mod END", raw_code, re.DOTALL | re.IGNORECASE).group(1)

                # Write payments.py
                payments_path = "/app/autonomous_agency/app/api/endpoints/payments.py"
                os.makedirs(os.path.dirname(payments_path), exist_ok=True)
                with open(payments_path, "w", encoding="utf-8") as f:
                    f.write(payments_py_code.strip())
                print(f"[MCOL] Generated payment endpoint file: {payments_path}")

                # Suggest modification for main.py (safer than auto-editing)
                print(f"[MCOL] Suggest adding router to main.py:\n{main_py_mod.strip()}")
                return {"status": "COMPLETED", "result": f"Generated {payments_path}. Suggested modification for main.py logged."}

            except Exception as e:
                print(f"[MCOL] Failed to parse or write payment endpoint code: {e}")
                return {"status": "FAILED", "result": f"Failed to parse/write payment code: {e}. Raw LLM output: {raw_code[:500]}"}
        else:
            print("[MCOL] Failed to generate payment endpoint code from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate payment endpoint code."}

    async def generate_report_delivery_code():
        print("[MCOL] Attempting to generate report delivery code modification...")
        delivery_prompt = """
        Given the existing function `process_single_report_request` in `app/agents/report_generator.py`, generate the Python code modifications needed to send an email with the report attached upon successful completion.
        Modifications needed:
        1. Inside the `try` block, after the line `final_status = "COMPLETED"` and before the `else` block for empty files.
        2. Add logic to:
           - Get an active email account using `await crud.get_active_email_account_for_sending(db)`. Handle case where no account is available (log error, proceed without sending).
           - Decrypt the password using `decrypt_data`. Handle decryption failure.
           - Create an `EmailMessage` object.
           - Set 'Subject': e.g., f"Your AI Research Report for '{request.request_details[:30]}...' is Ready"
           - Set 'From': sending account email.
           - Set 'To': `request.client_email`.
           - Set 'Date' and 'Message-ID'.
           - Set a simple text body: e.g., "Please find your requested AI research report attached."
           - **Attach the file:** Use `msg.add_attachment()` reading the file content from `output_path`. Determine the correct maintype/subtype (e.g., 'text/markdown').
           - Use `aiosmtplib.SMTP` (like in `email_marketer.py`) to connect, login, and send the message. Include basic error handling for the SMTP process. Log success or failure.
           - Increment the email account's sent count using `await crud.increment_email_sent_count(db, account.account_id)` if send is successful.
        3. Ensure necessary imports are added at the top of the file (EmailMessage, formatdate, make_msgid, aiosmtplib, crud, decrypt_data, etc.).
        Output ONLY the modified Python code snippet to be inserted, starting from the necessary imports down to the end of the new email sending logic. Assume the surrounding function structure exists.
        """
        code_response = await call_llm_api(client, delivery_prompt)
        if code_response and isinstance(code_response.get("raw_inference"), str):
            code_snippet = code_response["raw_inference"]
            # Suggest modification (safer than auto-editing)
            target_file = "app/agents/report_generator.py"
            print(f"[MCOL] SUGGESTION: Modify '{target_file}' to include report delivery logic. Add necessary imports and insert the following code block after 'final_status = \"COMPLETED\"':\n```python\n{code_snippet}\n```")
            return {"status": "SUGGESTED", "result": f"Code modification suggested for {target_file} to implement report delivery."}
        else:
            print("[MCOL] Failed to generate report delivery code modification from LLM.")
            return {"status": "FAILED", "result": "LLM failed to generate report delivery code."}

    # --- Strategy Execution Mapping ---
    implementation_result = {"status": "FAILED", "result": "Strategy not recognized or executable."}
    if "generate basic static website" in strategy_name:
        if MCOL_IMPLEMENTATION_MODE == "ATTEMPT_EXECUTE":
            implementation_result = await generate_website_files()
        else:
            implementation_result = {"status": "SUGGESTED", "result": "Suggest generating website files."}
    elif "implement lemon squeezy checkout links" in strategy_name:
         # Requires manual product setup first!
         print("[MCOL] WARNING: Lemon Squeezy strategy requires manual product setup in Lemon Squeezy dashboard first.")
         if MCOL_IMPLEMENTATION_MODE == "ATTEMPT_EXECUTE":
             implementation_result = await generate_payment_endpoint()
         else:
            implementation_result = {"status": "SUGGESTED", "result": "Suggest generating Lemon Squeezy endpoint code (requires manual product setup)."}
    elif "add automated report delivery email" in strategy_name:
        # Always suggest code modifications for safety unless explicitly overridden
        implementation_result = await generate_report_delivery_code() # Suggests modification
    # Add more strategy handlers here...
    else:
         # Default to suggestion for unrecognized strategies
         implementation_result = {"status": "SUGGESTED", "result": f"Suggest manually implementing strategy: {strategy['name']}. Desc: {strategy['description']}"}


    # Log final implementation outcome
    print(f"[MCOL] Implementation outcome for '{strategy['name']}': {implementation_result['status']} - {implementation_result['result']}")
    return {
        "status": implementation_result["status"],
        "result": implementation_result["result"],
        "parameters": action_details # Return original strategy details too
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
        await db.commit() # Commit snapshot
        kpi_str = format_kpis_for_llm(current_snapshot)
        print(f"[MCOL] {kpi_str}")

        # 2. Analyze & Prioritize
        analysis = await analyze_performance_and_prioritize(client, kpi_str)
        if not analysis:
            print("[MCOL] No critical problems identified or analysis failed. Ending cycle.")
            if client: await client.aclose()
            return

        # Log initial decision context
        decision = await crud.log_mcol_decision(
            db,
            kpi_snapshot_id=current_snapshot.snapshot_id,
            priority_problem=analysis["problem"],
            analysis_summary=analysis["reasoning"],
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

        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=json.dumps(strategies)) # Log all generated strategies
        await db.commit()

        # 4. Choose Strategy
        chosen_strategy = choose_strategy(strategies)
        if not chosen_strategy: # Should not happen if strategies exist
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY')
             await db.commit()
             if client: await client.aclose()
             return

        await crud.update_mcol_decision_log(db, decision_log_id, chosen_action=chosen_strategy['name'])
        await db.commit()

        # 5. Implement (Suggest or Attempt Execute)
        implementation_result = await implement_strategy(client, chosen_strategy)

        # Log implementation outcome
        await crud.update_mcol_decision_log(
            db,
            decision_log_id,
            action_status=implementation_result["status"],
            action_result=implementation_result["result"],
            action_parameters=implementation_result.get("parameters", action_details) # Use updated details if available
        )
        await db.commit()

        # 6. Verify (Basic - relies on next cycle's KPI snapshot)
        print(f"[MCOL] Cycle finished. Action status: {implementation_result['status']}")


    except Exception as e:
        print(f"[MCOL] Error during MCOL cycle: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging MCOL errors
        # Log error to DB if possible
        if decision_log_id:
            try:
                await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {e}")
                await db.commit()
            except Exception as db_err:
                 print(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                 await db.rollback() # Rollback logging attempt
        else:
             # If snapshot exists but decision log failed, try to rollback snapshot? Or leave it?
             # Let's assume snapshot is valuable even if rest of cycle fails.
             # await db.rollback()
             pass
    finally:
        if client:
            await client.aclose()