# autonomous_agency/app/agents/mcol_agent.py
import asyncio
import json
import datetime
import subprocess
import os
import re
import traceback
import logging
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
import httpx

# Corrected relative imports for package structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api, get_worker_session # Added get_worker_session
    # Import worker control functions if MCOL needs to start/stop workers
    # from Acumenis.app.main import _start_worker_task, stop_all_workers # Example
except ImportError:
    print("[MCOL Agent] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api, get_worker_session # Added get_worker_session
    # from app.main import _start_worker_task, stop_all_workers # Example

# Setup logger
logger = logging.getLogger(__name__)
# Ensure logger is configured (e.g., in main.py or here)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS or 300
# --- MODIFICATION: Default to EXECUTE_SAFE_CONFIG ---
MCOL_IMPLEMENTATION_MODE = settings.MCOL_IMPLEMENTATION_MODE or "EXECUTE_SAFE_CONFIG"
# --- END MODIFICATION ---
WEBSITE_OUTPUT_DIR = "/app/static_website"
API_KEY_LOW_THRESHOLD = getattr(settings, 'API_KEY_LOW_THRESHOLD', 5) # Threshold for warning
PROMPT_TUNING_ENABLED = getattr(settings, 'MCOL_PROMPT_TUNING_ENABLED', True) # Allow disabling prompt tuning suggestions
MCOL_MAX_STRATEGIES = getattr(settings, 'MCOL_MAX_STRATEGIES', 3) # Max strategies to generate

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    if not snapshot: return "No KPI data available."
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    lines.append(f"- Reports: AwaitingGen={getattr(snapshot, 'awaiting_generation_reports', 0)}, PendingTask={getattr(snapshot, 'pending_reports', 0)}, Processing={getattr(snapshot, 'processing_reports', 0)}, Completed(24h)={getattr(snapshot, 'completed_reports_24h', 0)}, Failed(24h)={getattr(snapshot, 'failed_reports_24h', 0)}, DeliveryFailed(24h)={getattr(snapshot, 'delivery_failed_reports_24h', 0)}, AvgTime={snapshot.avg_report_time_seconds or 0:.2f}s")
    lines.append(f"- Prospecting: New(24h)={getattr(snapshot, 'new_prospects_24h', 0)}")
    lines.append(f"- Email: Sent(24h)={getattr(snapshot, 'emails_sent_24h', 0)}, ActiveAccounts={getattr(snapshot, 'active_email_accounts', 0)}, Deactivated(24h)={getattr(snapshot, 'deactivated_accounts_24h', 0)}, BounceRate(24h)={snapshot.bounce_rate_24h or 0:.2f}%")
    lines.append(f"- Revenue: Orders(24h)={getattr(snapshot, 'orders_created_24h', 0)}, Revenue(24h)=${snapshot.revenue_24h:.2f}")
    lines.append(f"- API Keys: Active={getattr(snapshot, 'active_api_keys', 0)}, Deactivated(24h)={getattr(snapshot, 'deactivated_api_keys_24h', 0)}")
    return "\n".join(lines)

async def analyze_performance_and_prioritize(client: httpx.AsyncClient, kpi_data_str: str) -> Optional[Dict[str, str]]:
    """Uses LLM (Acumenis Prime) to analyze KPIs, identify the biggest problem, and explain why."""
    primary_goal = "Achieve $10,000 revenue within 72 hours, then sustain growth via AI report sales ($499/$999) and autonomous client acquisition."
    system_context = f"""
    System Overview: Autonomous agency 'Acumenis' using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses 'open-deep-research', delivers via email), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation), KeyAcquirer (automated key scraping - HIGH RISK), MCOL (self-improvement analysis). Deployed via Docker at {settings.AGENCY_BASE_URL}. Payment via Lemon Squeezy. Website serving via FastAPI static files. Core LLM: Acumenis Prime (You).
    """
    prompt = f"""
    Analyze the following system performance data for Acumenis, an autonomous AI reporting agency aiming for rapid revenue generation.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL bottleneck currently preventing rapid revenue growth or system operation. Consider the entire funnel: Key Acquisition -> Prospecting -> Email Outreach -> Website Visit -> Order Attempt -> Payment Success -> Report Generation -> Delivery. Look for zero values or critical failures (e.g., Active API Keys=0, Orders=0, High Bounce Rate, High Report Failures).
    2. Briefly explain the reasoning for selecting this problem (impact on revenue/growth).
    3. Respond ONLY with a JSON object containing "problem" and "reasoning".
    Example Problems: "Zero Active API Keys (KeyAcquirer Failure)", "Zero Orders Created (Website/Payment Issue?)", "Low Prospect Acquisition Rate", "High Email Bounce Rate (>20%)", "Payment Webhook Not Triggering Report Creation", "Report Generation Failing Frequently".
    If Revenue(24h) is significantly positive and growing, identify the next major bottleneck to scaling. If all looks optimal, respond: {{"problem": "None", "reasoning": "Current KPIs indicate strong progress towards goal."}}
    """
    logger.info("[MCOL] Analyzing KPIs with LLM...")
    llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest") # Use a capable model

    if llm_response and isinstance(llm_response, dict) and "problem" in llm_response and "reasoning" in llm_response:
        problem = llm_response["problem"].strip()
        reasoning = llm_response["reasoning"].strip()
        if problem and problem.lower() != "none":
            logger.info(f"[MCOL] Identified Priority Problem: {problem} (Reason: {reasoning})")
            return {"problem": problem, "reasoning": reasoning}
        else:
            logger.info("[MCOL] LLM analysis indicates no critical problems currently, or goal is being met.")
            return None
    else:
        logger.error(f"[MCOL] Failed to get valid analysis from LLM. Response: {llm_response}")
        return None

async def generate_solution_strategies(client: httpx.AsyncClient, problem: str, reasoning: str, kpi_data_str: str) -> Optional[List[Dict[str, str]]]:
    """Uses LLM (Acumenis Prime) to generate potential solution strategies for the identified problem."""
    system_context = f"""
    System: Acumenis Agency. FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, KeyAcquirer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (configured). Website: Served via FastAPI static files at {settings.AGENCY_BASE_URL}. Budget: $5/month (data center proxies). Uses free tiers/acquired keys aggressively. Core LLM: Acumenis Prime (You). MCOL Mode: {MCOL_IMPLEMENTATION_MODE}.
    Code Structure: /app/ contains main.py, agents/, db/, core/, workers/, api/endpoints/. /app/open-deep-research-main/ contains the external tool. Migrations via Alembic.
    """
    # --- MODIFICATION: Adjust prompt based on MCOL_IMPLEMENTATION_MODE ---
    implementation_focus = ""
    if MCOL_IMPLEMENTATION_MODE == "SUGGEST":
        implementation_focus = "focus on generating *clear, actionable suggestions* for a human operator. If code changes are needed, describe the logic clearly. If prompt tuning is needed, provide example improved prompts. Consider suggesting experiments (A/B tests) the operator can run."
    elif MCOL_IMPLEMENTATION_MODE == "EXECUTE_SAFE_CONFIG":
        implementation_focus = "focus on generating strategies that involve *modifying configuration parameters* (e.g., email delays, API thresholds, model selection) or *triggering specific agent tasks* (e.g., start KeyAcquirer). Provide the exact parameter name and the suggested new value or the exact task to trigger. Avoid suggesting direct code modifications."
    # Add other modes like EXECUTE_PROMPT_TUNING or EXECUTE_CODE later if needed
    else: # Default to SUGGEST if mode is unknown
        implementation_focus = "focus on generating *clear, actionable suggestions* for a human operator."

    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate {MCOL_MAX_STRATEGIES} diverse, actionable, and creative strategies to solve the identified problem for Acumenis Agency. Prioritize AI/automation, budget constraints ($5 proxies), and rapid revenue generation.
    Implementation Focus: Since MCOL_IMPLEMENTATION_MODE='{MCOL_IMPLEMENTATION_MODE}', {implementation_focus}

    Instructions:
    1. Provide 'name' (concise action verb phrase), 'target_component' (e.g., "EmailMarketer Agent", "KeyAcquirer Worker", "Database Config", "ProspectResearcher Prompt"), and 'action_details' (specific steps, parameter changes, task triggers, or prompt examples).
    2. Structure the 'action_details' clearly, outlining the hypothesis (if applicable), the suggested action/experiment, and the expected outcome metric to monitor.
    3. Output ONLY a JSON list of strategy objects.

    Example Strategies (Focus on '{MCOL_IMPLEMENTATION_MODE}' Mode):
    - Problem: "Zero Active API Keys (KeyAcquirer Failure)" -> Strategy: {{"name": "Trigger KeyAcquirer", "target_component": "KeyAcquirer Worker", "action_details": {{"hypothesis": "KeyAcquirer may have stopped or failed.", "action": "TRIGGER_WORKER_START", "worker_name": "key_acquirer", "expected_outcome_metric": "Increase in `active_api_keys` KPI."}}}}
    - Problem: "High Email Bounce Rate (>15%)" -> Strategy: {{"name": "Increase Email Send Delay", "target_component": "Database Config", "action_details": {{"hypothesis": "Sending too fast triggers spam filters/limits.", "action": "UPDATE_CONFIG", "parameter_name": "EMAIL_SEND_DELAY_MAX", "new_value": 7.0, "expected_outcome_metric": "Reduction in `bounce_rate_24h`."}}}}
    - Problem: "Low Report Generation Success Rate" -> Strategy: {{"name": "Switch to Standard LLM for Reports", "target_component": "Database Config", "action_details": {{"hypothesis": "Premium model might be failing more often or hitting limits.", "action": "UPDATE_CONFIG", "parameter_name": "STANDARD_REPORT_MODEL", "new_value": "google/gemini-1.5-flash-latest", "notes": "Also update PREMIUM_REPORT_MODEL if needed", "expected_outcome_metric": "Increase in `completed_reports_24h`, decrease in `failed_reports_24h`."}}}}
    """
    # --- END MODIFICATION ---
    logger.info(f"[MCOL] Generating strategies for problem: {problem} (Mode: {MCOL_IMPLEMENTATION_MODE})")
    llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest") # Use capable model

    strategies = None
    if llm_response:
        if isinstance(llm_response, list): strategies = llm_response
        elif isinstance(llm_response.get("strategies"), list): strategies = llm_response["strategies"]
        elif isinstance(llm_response.get("raw_inference"), str):
            try:
                raw_inf = llm_response["raw_inference"]
                # Try to find JSON list within potential markdown code blocks or direct output
                json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_inf, re.IGNORECASE | re.DOTALL)
                json_str = json_match.group(1).strip() if json_match else raw_inf.strip()
                # Handle potential leading/trailing text before/after JSON list
                start_index = json_str.find('[')
                end_index = json_str.rfind(']')
                if start_index != -1 and end_index != -1:
                    json_str = json_str[start_index : end_index + 1]
                    parsed_raw = json.loads(json_str)
                    if isinstance(parsed_raw, list): strategies = parsed_raw
                else:
                    logger.error(f"[MCOL] Could not find valid JSON list structure in raw inference.")

            except (json.JSONDecodeError, AttributeError, IndexError) as parse_e:
                logger.error(f"[MCOL] LLM raw inference is not valid JSON list. Error: {parse_e}. Response: {raw_inf[:500]}...")

    # Validate strategy structure
    if strategies and all(isinstance(s, dict) and "name" in s and "action_details" in s and "target_component" in s for s in strategies):
        logger.info(f"[MCOL] LLM generated {len(strategies)} strategies.")
        return strategies[:MCOL_MAX_STRATEGIES] # Limit number of strategies
    else:
        logger.error(f"[MCOL] Failed to get valid strategies from LLM. Response: {llm_response}")
        return None

def choose_strategy(strategies: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Selects the best strategy (simple: pick the first one for now)."""
    if not strategies: return None
    # Future: Implement LLM-based ranking or heuristic selection based on feasibility/impact.
    chosen = strategies[0]
    logger.info(f"[MCOL] Choosing strategy: {chosen['name']}")
    return chosen

# --- MODIFICATION: Add execute_strategy function ---
async def execute_strategy(db: AsyncSession, strategy: Dict[str, str]) -> Dict[str, Any]:
    """Attempts to execute a safe strategy based on MCOL_IMPLEMENTATION_MODE."""
    action_details = strategy.get("action_details", {})
    action_type = action_details.get("action")
    strategy_name = strategy.get("name", "Unknown Strategy")
    target_component = strategy.get("target_component", "Unknown Target")
    result_status = "FAILED_EXECUTION"
    result_message = "Action type not recognized or not executable in current mode."

    logger.info(f"[MCOL] Attempting to execute strategy: {strategy_name} (Target: {target_component}, Action: {action_type})")

    if MCOL_IMPLEMENTATION_MODE == "EXECUTE_SAFE_CONFIG":
        if action_type == "UPDATE_CONFIG":
            param_name = action_details.get("parameter_name")
            new_value = action_details.get("new_value")
            if param_name and new_value is not None:
                try:
                    # IMPORTANT: Need a safe way to update config.
                    # Option 1: Update a dedicated 'dynamic_config' table in DB
                    # success = await crud.update_dynamic_config(db, param_name, new_value)
                    # Option 2: Call an internal API endpoint to reload config (if feasible)
                    # Option 3 (Placeholder): Log the intended change
                    logger.warning(f"EXECUTION SIMULATION: Would update config '{param_name}' to '{new_value}'. (DB update function not implemented)")
                    # Assume success for simulation
                    success = True # Replace with actual success check
                    if success:
                        result_status = "EXECUTED_CONFIG_UPDATE"
                        result_message = f"Successfully executed: Set config '{param_name}' to '{new_value}'."
                    else:
                        result_message = f"Failed to execute config update for '{param_name}'."
                except Exception as e:
                    logger.error(f"Error executing config update strategy: {e}", exc_info=True)
                    result_message = f"Error executing config update: {e}"
            else:
                result_message = "Missing parameter_name or new_value for UPDATE_CONFIG action."

        elif action_type == "TRIGGER_WORKER_START":
            worker_name = action_details.get("worker_name")
            if worker_name:
                try:
                    # IMPORTANT: Requires access to worker control functions.
                    # This might involve calling an internal API endpoint of the main app
                    # or directly invoking functions if running in the same process space (less ideal).
                    # Example using an internal API call:
                    async with get_httpx_client() as client:
                         # Construct the full URL for the control endpoint
                         control_url = f"{settings.AGENCY_BASE_URL.rstrip('/')}/control/start/{worker_name}"
                         logger.info(f"Attempting to trigger worker start via API: POST {control_url}")
                         response = await client.post(control_url)
                         response.raise_for_status() # Raise exception for 4xx/5xx errors
                         result_message = f"Successfully triggered start for worker '{worker_name}' via API. Response: {response.json()}"
                         result_status = "EXECUTED_WORKER_TRIGGER"
                    # logger.warning(f"EXECUTION SIMULATION: Would trigger start for worker '{worker_name}'. (Worker trigger mechanism not implemented)")
                    # Assume success for simulation
                    # result_status = "EXECUTED_WORKER_TRIGGER"
                    # result_message = f"Successfully executed: Triggered start for worker '{worker_name}'."
                except httpx.RequestError as req_err:
                    logger.error(f"HTTP Request Error triggering worker '{worker_name}': {req_err}", exc_info=True)
                    result_message = f"HTTP Request Error triggering worker '{worker_name}': {req_err}"
                except httpx.HTTPStatusError as status_err:
                    logger.error(f"HTTP Status Error triggering worker '{worker_name}': {status_err.response.status_code} - {status_err.response.text}", exc_info=True)
                    result_message = f"HTTP Status Error triggering worker '{worker_name}': {status_err.response.status_code}"
                except Exception as e:
                    logger.error(f"Error executing worker trigger strategy: {e}", exc_info=True)
                    result_message = f"Error triggering worker '{worker_name}': {e}"
            else:
                result_message = "Missing worker_name for TRIGGER_WORKER_START action."
        else:
             result_message = f"Action type '{action_type}' not supported in EXECUTE_SAFE_CONFIG mode."

    elif MCOL_IMPLEMENTATION_MODE == "SUGGEST":
        # Format the suggestion clearly for the operator log
        suggestion_output = f"""
**MCOL Suggestion for Operator**

*   **Priority Problem:** {strategy.get('problem_context', 'N/A')}
*   **Chosen Strategy:** {strategy_name}
*   **Target Component:** `{target_component}`

**Suggested Action/Analysis:**

{action_details.get('description', 'No description provided.')}
"""
        result_status = "SUGGESTED"
        result_message = suggestion_output.strip()
    else:
        # Handle other modes or default to suggestion
        result_status = "MODE_NOT_SUPPORTED"
        result_message = f"Implementation mode '{MCOL_IMPLEMENTATION_MODE}' not fully supported for execution. Suggestion logged."
        # Log suggestion as fallback
        suggestion_output = f"""
**MCOL Suggestion (Fallback due to Mode)**

*   **Priority Problem:** {strategy.get('problem_context', 'N/A')}
*   **Chosen Strategy:** {strategy_name}
*   **Target Component:** `{target_component}`

**Suggested Action/Analysis:**

{action_details.get('description', 'No description provided.')}
"""
        result_message = suggestion_output.strip()


    logger.info(f"[MCOL] Strategy execution outcome for '{strategy_name}': {result_status}")
    return {
        "status": result_status,
        "result": result_message,
        "parameters": strategy # Store original strategy details for context
    }
# --- END MODIFICATION ---


async def check_api_key_status(db: AsyncSession) -> Optional[Dict[str, Any]]:
    """Checks the status of API keys and returns a structured suggestion/action if low."""
    try:
        # Use the specific CRUD function to count active keys
        active_keys_count = await crud.count_active_api_keys(db, provider=settings.LLM_PROVIDER or "openrouter")
        logger.info(f"[MCOL] Active API Keys Check: Found {active_keys_count} active keys.")

        if active_keys_count < API_KEY_LOW_THRESHOLD:
            problem = f"Low Active API Keys ({active_keys_count} < {API_KEY_LOW_THRESHOLD})"
            reasoning = "Insufficient active API keys risk halting all LLM-dependent operations."

            # Define the strategy based on execution mode
            if MCOL_IMPLEMENTATION_MODE == "EXECUTE_SAFE_CONFIG":
                 strategy = {
                     "name": "Trigger KeyAcquirer Due to Low Keys",
                     "target_component": "KeyAcquirer Worker",
                     "action_details": {
                         "hypothesis": "Key count is below threshold, need to acquire more.",
                         "action": "TRIGGER_WORKER_START",
                         "worker_name": "key_acquirer",
                         "expected_outcome_metric": f"Increase in `active_api_keys` KPI above {API_KEY_LOW_THRESHOLD}."
                     },
                     "problem_context": problem # Pass problem context
                 }
                 # Execute the strategy directly
                 execution_result = await execute_strategy(db, strategy)

            else: # Default to SUGGEST mode
                strategy_desc = (
                    f"**Hypothesis:** KeyAcquirer may be failing or keys are being rapidly deactivated.\n"
                    f"**Suggested Action:**\n"
                    f"1. **Diagnose KeyAcquirer:** Review `run_key_acquirer_worker.py` logs for recent errors (CAPTCHA, 403, etc.).\n"
                    f"2. **Manual Intervention (if blocked/urgent):** Manually acquire ~{API_KEY_LOW_THRESHOLD - active_keys_count + 5} new OpenRouter keys and add via DB tool/script. Consider setting `KEY_ACQUIRER_RUN_ON_STARTUP=False` in `.env` if automation is clearly blocked.\n"
                    f"3. **Run KeyAcquirer (if potentially viable):** If logs show only transient errors, try starting the worker via `/control/start/key_acquirer`. Monitor its logs closely.\n"
                    f"**Expected Outcome Metric:** `active_api_keys` KPI increases above {API_KEY_LOW_THRESHOLD}."
                )
                strategy = {
                    "name": "Suggest Addressing Low API Key Count",
                    "target_component": "KeyAcquirer Worker / Operator",
                    "action_details": {"description": strategy_desc},
                    "problem_context": problem
                 }
                 # Simulate suggestion logging (execute_strategy handles SUGGEST mode)
                execution_result = await execute_strategy(db, strategy)


            return {
                "problem": problem,
                "reasoning": reasoning,
                "executed_action": execution_result # Contains status, result, parameters
            }
    except Exception as e:
        logger.error(f"[MCOL] Error checking API key status: {e}", exc_info=True)
    return None


# --- Main MCOL Agent Cycle ---
async def run_mcol_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single Monitor -> Analyze -> Strategize -> Implement/Execute cycle."""
    if shutdown_event.is_set(): return
    logger.info(f"[MCOL] Starting cycle at {datetime.datetime.now(datetime.timezone.utc)}")
    client: Optional[httpx.AsyncClient] = None
    current_snapshot: Optional[models.KpiSnapshot] = None
    decision_log_id: Optional[int] = None

    try:
        # 0. Pre-Cycle Check: API Key Status (High Priority)
        key_status_check_result = await check_api_key_status(db)
        if key_status_check_result:
            logger.warning(f"[MCOL] Priority Issue Detected: {key_status_check_result['problem']}")
            action_to_log = key_status_check_result['executed_action']
            # Log this critical action/suggestion immediately
            decision = await crud.log_mcol_decision(
                db, kpi_snapshot_id=None, # No full KPI cycle run yet
                priority_problem=key_status_check_result["problem"],
                analysis_summary=key_status_check_result["reasoning"],
                chosen_action=action_to_log['parameters']['name'],
                action_status=action_to_log['status'],
                action_result=action_to_log['result'],
                action_parameters=action_to_log.get("parameters")
            )
            await db.commit()
            logger.info(f"[MCOL] Logged critical API key action/suggestion (Log ID: {decision.log_id}). Proceeding to main cycle.")
            # Potentially return early if a critical action was taken? Or continue analysis? Continue for now.

        # 1. Monitor: Create KPI Snapshot
        current_snapshot = await crud.create_kpi_snapshot(db)
        await db.commit() # Commit snapshot separately
        if not current_snapshot:
             logger.error("[MCOL] Failed to create KPI snapshot. Skipping cycle.")
             return
        kpi_str = format_kpis_for_llm(current_snapshot)
        logger.info(f"[MCOL] {kpi_str}")

        # 2. Analyze & Prioritize
        client = await get_httpx_client()
        analysis = await analyze_performance_and_prioritize(client, kpi_str)
        if not analysis:
            logger.info("[MCOL] No critical problems identified by LLM or analysis failed. Ending cycle.")
            if client: await client.aclose()
            return

        # Log initial decision phase
        decision = await crud.log_mcol_decision(
            db, kpi_snapshot_id=current_snapshot.snapshot_id,
            priority_problem=analysis["problem"], analysis_summary=analysis["reasoning"],
            action_status='ANALYZED'
        )
        await db.commit() # Commit analysis phase
        decision_log_id = decision.log_id

        # 3. Strategize
        strategies = await generate_solution_strategies(client, analysis["problem"], analysis["reasoning"], kpi_str)
        if not strategies:
            await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="LLM failed to generate valid strategies.")
            await db.commit()
            logger.error("[MCOL] Failed to generate strategies. Ending cycle.")
            if client: await client.aclose()
            return

        # Store strategies as JSON string in the log
        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=json.dumps(strategies))
        await db.commit() # Commit generated strategies

        # 4. Choose Strategy
        chosen_strategy = choose_strategy(strategies)
        if not chosen_strategy:
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="Strategy choice failed.")
             await db.commit()
             logger.error("[MCOL] Strategy choice failed unexpectedly. Ending cycle.")
             if client: await client.aclose()
             return

        # Add problem context for execution/logging
        chosen_strategy['problem_context'] = analysis["problem"]
        await crud.update_mcol_decision_log(db, decision_log_id, chosen_action=chosen_strategy['name'])
        await db.commit() # Commit chosen strategy

        # 5. Implement / Execute
        execution_result = await execute_strategy(db, chosen_strategy)

        # Update the log with the final suggestion/outcome
        await crud.update_mcol_decision_log(
            db, decision_log_id,
            action_status=execution_result["status"],
            action_result=execution_result["result"],
            action_parameters=execution_result.get("parameters") # Store strategy details
        )
        await db.commit() # Commit final outcome

        logger.info(f"[MCOL] Cycle finished. Action status: {execution_result['status']}")

    except Exception as e:
        logger.error(f"[MCOL] CRITICAL Error during MCOL cycle: {e}", exc_info=True)
        if decision_log_id:
            try:
                await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {traceback.format_exc(limit=1000)}") # Longer limit for traceback
                await db.commit()
            except Exception as db_err:
                 logger.error(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                 await db.rollback()
        else:
             # If no decision log entry was even created
             await db.rollback()
    finally:
        if client:
            await client.aclose()