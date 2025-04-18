# autonomous_agency/app/agents/mcol_agent.py
# Optimized for SINGLE API KEY monitoring and SUGGEST mode

import asyncio
import json
import datetime
import traceback
import logging
from typing import Optional, List, Dict, Any, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
import httpx

# Corrected relative imports
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.db.base import get_worker_session
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api
    from Acumenis.app.agents.agent_utils import NoValidApiKeyError, APIKeyInvalidError # Import specific errors
except ImportError:
    print("[MCOL Agent] WARNING: Using fallback imports.")
    from app.core.config import settings
    from app.db import crud, models
    from app.db.base import get_worker_session
    from app.agents.agent_utils import get_httpx_client, call_llm_api
    from app.agents.agent_utils import NoValidApiKeyError, APIKeyInvalidError

# Setup logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
     logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS
# Force SUGGEST mode for safety in single-key scenario, log if overridden
MCOL_IMPLEMENTATION_MODE = "SUGGEST"
if settings.MCOL_IMPLEMENTATION_MODE != "SUGGEST":
    logger.warning(f"MCOL_IMPLEMENTATION_MODE '{settings.MCOL_IMPLEMENTATION_MODE}' overridden to 'SUGGEST' due to single-key strategy risks.")

MCOL_MAX_STRATEGIES = settings.MCOL_MAX_STRATEGIES
SINGLE_KEY_PROVIDER = settings.LLM_PROVIDER or "openrouter" # Provider for the single key

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    # (Function unchanged)
    if not snapshot: return "No KPI data available."
    # ... (rest of formatting as before) ...
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    lines.append(f"- Reports: AwaitingGen={getattr(snapshot, 'awaiting_generation_reports', 0)}, PendingTask={getattr(snapshot, 'pending_reports', 0)}, Processing={getattr(snapshot, 'processing_reports', 0)}, Completed(24h)={getattr(snapshot, 'completed_reports_24h', 0)}, Failed(24h)={getattr(snapshot, 'failed_reports_24h', 0)}, DeliveryFailed(24h)={getattr(snapshot, 'delivery_failed_reports_24h', 0)}, AvgTime={snapshot.avg_report_time_seconds or 0:.2f}s")
    lines.append(f"- Prospecting: New(24h)={getattr(snapshot, 'new_prospects_24h', 0)}")
    lines.append(f"- Email: Sent(24h)={getattr(snapshot, 'emails_sent_24h', 0)}, ActiveAccounts={getattr(snapshot, 'active_email_accounts', 0)}, Deactivated(24h)={getattr(snapshot, 'deactivated_accounts_24h', 0)}, BounceRate(24h)={snapshot.bounce_rate_24h or 0:.2f}%")
    lines.append(f"- Revenue: Orders(24h)={getattr(snapshot, 'orders_created_24h', 0)}, Revenue(24h)=${snapshot.revenue_24h:.2f}")
    lines.append(f"- API Keys: Active={getattr(snapshot, 'active_api_keys', 0)}, Deactivated(24h)={getattr(snapshot, 'deactivated_api_keys_24h', 0)}")
    return "\n".join(lines)


async def get_single_key_status_info(db: AsyncSession) -> Dict[str, Any]:
    """
    Fetches status information for the single operational API key.
    Returns a dict with 'status', 'reason', 'rate_limited_until', 'exists'.
    """
    key_value = settings.OPENROUTER_API_KEY
    if not key_value:
        return {"status": "missing", "reason": "OPENROUTER_API_KEY not set in environment.", "rate_limited_until": None, "exists": False}

    # Inefficiently find the key by value (necessary if ID isn't stored/known)
    # This requires decrypting keys until a match is found.
    # A better approach would be to store the *ID* of the operational key in config,
    # but sticking to the current structure for now.
    target_key_obj: Optional[models.ApiKey] = None
    all_keys_stmt = select(models.ApiKey).where(models.ApiKey.provider == SINGLE_KEY_PROVIDER)
    all_keys_res = await db.execute(all_keys_stmt)
    for key_obj in all_keys_res.scalars().all():
        try:
            dec_key = decrypt_data(key_obj.api_key_encrypted)
            if dec_key == key_value:
                target_key_obj = key_obj
                break
        except Exception:
            logger.warning(f"Failed to decrypt key ID {key_obj.id} during status check.")
            continue # Skip keys that fail decryption

    if target_key_obj:
        logger.info(f"Found operational key ID {target_key_obj.id} in DB. Status: {target_key_obj.status}")
        return {
            "status": target_key_obj.status,
            "reason": target_key_obj.last_failure_reason,
            "rate_limited_until": target_key_obj.rate_limited_until,
            "exists": True,
            "id": target_key_obj.id # Include ID if found
        }
    else:
        # Key from settings is not found in the database
        logger.warning(f"The API key configured in settings (OPENROUTER_API_KEY) was not found in the database for provider '{SINGLE_KEY_PROVIDER}'.")
        # Treat this as if the key is missing/invalid for operational purposes
        return {"status": "missing", "reason": "Key from settings not found in DB.", "rate_limited_until": None, "exists": False}


async def check_single_api_key_health(db: AsyncSession) -> Optional[Dict[str, Any]]:
    """
    Checks the health of the single operational API key.
    Generates a CRITICAL suggestion if the key is not 'active' and operational.
    """
    try:
        key_info = await get_single_key_status_info(db)
        key_status = key_info["status"]
        failure_reason = key_info["reason"]
        rate_limited_until = key_info["rate_limited_until"]
        key_exists = key_info["exists"]
        key_id = key_info.get("id") # Get ID if found

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        is_problem = False
        problem_desc = ""
        reasoning = ""

        if not key_exists:
            is_problem = True
            problem_desc = "CRITICAL: Operational API Key Not Found in DB"
            reasoning = "The API key specified in OPENROUTER_API_KEY environment variable is missing from the database. Add the key manually or correct the environment variable."
        elif key_status != 'active':
            is_problem = True
            problem_desc = f"CRITICAL: Operational API Key Status is '{key_status}'"
            reasoning = f"The sole operational API key (ID: {key_id or 'N/A'}) is not active. Reason: {failure_reason or 'N/A'}. All LLM operations are blocked."
        elif rate_limited_until:
             # Ensure rate_limited_until is timezone-aware for comparison
             if rate_limited_until.tzinfo is None:
                 rate_limited_until = rate_limited_until.replace(tzinfo=datetime.timezone.utc)
             if now_utc < rate_limited_until:
                is_problem = True
                problem_desc = f"CRITICAL: Operational API Key is Rate-Limited"
                reasoning = f"The sole operational API key (ID: {key_id or 'N/A'}) is rate-limited until {rate_limited_until.isoformat()}. LLM operations blocked until then."

        if is_problem:
            logger.critical(f"[MCOL Pre-Check] {problem_desc}. Reason: {reasoning}")
            strategy_desc = (
                f"**CRITICAL ALERT:** Agency LLM function compromised due to API key issue.\n"
                f"**Detected Status:** {key_status}\n"
                f"**Rate Limited Until:** {rate_limited_until.isoformat() if rate_limited_until else 'N/A'}\n"
                f"**Last Failure Reason:** {failure_reason or 'N/A'}\n"
                f"**REQUIRED IMMEDIATE ACTION:**\n"
                f"1. **INVESTIGATE:** Check OpenRouter dashboard for key status, usage, and billing.\n"
                f"2. **REPLACE KEY:** If key is invalid/banned/missing, obtain a NEW key and update the `OPENROUTER_API_KEY` environment variable.\n"
                f"3. **WAIT (if rate-limited):** If rate-limited, wait until the cooldown expires ({rate_limited_until.isoformat() if rate_limited_until else 'Unknown'}). Consider suggesting increased delays if this repeats (see other MCOL suggestions).\n"
                f"4. **RESTART APPLICATION:** After changing the environment variable, restart the application service."
            )
            strategy = {
                "name": "CRITICAL ALERT: Single API Key Failure/Unavailable",
                "target_component": "Operator / Environment Config",
                "action_details": {"description": strategy_desc, "action": "OPERATOR_ALERT"},
                "problem_context": problem_desc
            }
            # Execute will just log the suggestion in SUGGEST mode
            execution_result = await execute_strategy(db, strategy)

            return {
                "problem": problem_desc,
                "reasoning": reasoning,
                "executed_action": execution_result
            }
    except Exception as e:
        logger.error(f"[MCOL] Error checking API key status: {e}", exc_info=True)
    return None # No critical issue detected or error occurred during check


async def analyze_performance_and_prioritize(client: httpx.AsyncClient, kpi_data_str: str) -> Optional[Dict[str, str]]:
    """Uses LLM to analyze KPIs, identify the biggest problem (excluding key status)."""
    primary_goal = "Maximize profitable report sales ($499/$999) via autonomous prospecting and email marketing, operating reliably with a single API key and proxy rotation."
    system_context = f"""
    System Overview: Autonomous agency 'Acumenis' using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Deployed via Docker at {settings.AGENCY_BASE_URL}. Payment via Lemon Squeezy. Website serving via FastAPI static files. Core LLM: Acumenis Prime (You). CRITICAL CONSTRAINT: System relies on a SINGLE OpenRouter API key with proxy rotation. KeyAcquirer is DISABLED.
    """
    prompt = f"""
    Analyze the following system performance data for Acumenis, an autonomous AI reporting agency operating under a SINGLE API KEY constraint.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Assume the single API key is currently functional (checked separately). Identify the single MOST CRITICAL bottleneck HINDERING REVENUE GENERATION or causing operational inefficiency. Focus on the funnel: Prospecting -> Email Outreach -> Website Visit -> Order -> Payment -> Report Generation -> Delivery.
    2. Look for zero values or critical failures OTHER THAN API key status (e.g., Zero Orders Created, High Email Bounce Rate (>25%), Report Generation Failing Frequently (>15%), No New Prospects, Low Email Send Rate).
    3. Briefly explain the reasoning for selecting this problem (direct impact on revenue or core workflow).
    4. Respond ONLY with a JSON object containing "problem" and "reasoning".
    Example Problems: "Zero Orders Created (Website/Payment/Webhook Issue?)", "Low Prospect Acquisition Rate (ODR Service/Queries Ineffective?)", "High Email Bounce Rate (>25%) (Email Accounts/Content Issue?)", "Report Generation Failing Frequently (ODR Service/LLM Issue?)".
    If Revenue(24h) > $100 and major KPIs look healthy, identify the NEXT bottleneck to scaling revenue further. If all looks optimal, respond: {{"problem": "None", "reasoning": "Current KPIs indicate acceptable performance under single-key constraint. Focus on scaling outreach/conversion."}}
    """
    logger.info("[MCOL] Analyzing KPIs (excluding key status) with LLM...")
    try:
        llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest") # Use a capable model
    except (NoValidApiKeyError, APIKeyInvalidError) as key_err:
         logger.error(f"[MCOL] Cannot analyze performance, API key issue detected: {key_err}")
         return None
    except Exception as e:
         logger.error(f"[MCOL] Error calling LLM for analysis: {e}", exc_info=True)
         return None


    if llm_response and isinstance(llm_response, dict) and "problem" in llm_response and "reasoning" in llm_response:
        problem = llm_response["problem"].strip()
        reasoning = llm_response["reasoning"].strip()
        if problem and problem.lower() != "none":
            logger.info(f"[MCOL] Identified Priority Problem (Non-Key): {problem} (Reason: {reasoning})")
            return {"problem": problem, "reasoning": reasoning}
        else:
            logger.info("[MCOL] LLM analysis indicates no critical non-key problems currently.")
            return None
    else:
        logger.error(f"[MCOL] Failed to get valid analysis from LLM. Response: {llm_response}")
        return None


async def generate_solution_strategies(client: httpx.AsyncClient, problem: str, reasoning: str, kpi_data_str: str) -> Optional[List[Dict[str, str]]]:
    """Generates strategies, focusing on SUGGEST or SAFE_CONFIG, avoiding KeyAcquirer triggers."""
    system_context = f"""
    System: Acumenis Agency. FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Core tool: 'open-deep-research' (internal service). Payment: Lemon Squeezy. Website: Static files. Budget: Minimal ($5 proxies). Uses SINGLE OpenRouter API key. Core LLM: Acumenis Prime (You). MCOL Mode: {MCOL_IMPLEMENTATION_MODE} (Forced to SUGGEST). KeyAcquirer DISABLED.
    """
    # Force SUGGEST mode focus
    implementation_focus = "focus ONLY on generating *clear, actionable suggestions* for a human operator. If config changes are needed (e.g., email delays, model choice, query lists), describe the parameter and suggested value clearly for manual update. If prompt tuning is needed, provide examples. Suggest checks the operator should perform. DO NOT suggest automated actions like starting workers or modifying code."

    prompt = f"""
    Problem Diagnosis:
    - Identified Problem (Non-Key): "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate {MCOL_MAX_STRATEGIES} diverse, actionable strategies as SUGGESTIONS for the operator to solve the identified problem for Acumenis Agency. Prioritize low-cost, high-impact actions suitable for a single API key operation.
    Implementation Focus: {implementation_focus}

    Instructions:
    1. Provide 'name' (concise action verb phrase), 'target_component' (e.g., "EmailMarketer Config", "ProspectResearcher Queries", "Payment Webhook", "Operator"), and 'action_details'.
    2. Structure 'action_details' with: 'action': 'SUGGEST_CONFIG_UPDATE' or 'OPERATOR_CHECK' or 'SUGGEST_PROMPT_TUNING', 'description': (Detailed steps/checks for the operator), 'parameter_name': (if config update), 'suggested_value': (if config update), 'expected_outcome_metric'.
    3. Output ONLY a JSON list of strategy objects. DO NOT include strategies involving KeyAcquirer or adding more API keys. Ensure descriptions are clear instructions for a human.

    Example Strategies (Focus on SUGGESTIONS):
    - Problem: "High Email Bounce Rate (>25%)" -> Strategy: {{"name": "Suggest Increasing Email Send Delays", "target_component": "Operator / Environment Config", "action_details": {{"action": "SUGGEST_CONFIG_UPDATE", "description": "High bounce rate detected. Suggest increasing email delays to avoid spam filters. Operator should update EMAIL_SEND_DELAY_MIN to 3.0 and EMAIL_SEND_DELAY_MAX to 10.0 in the environment variables and restart the application.", "parameter_name": "EMAIL_SEND_DELAY_MIN/MAX", "suggested_value": "3.0 / 10.0", "expected_outcome_metric": "Reduction in `bounce_rate_24h` below 15%."}}}}
    - Problem: "Low Prospect Acquisition Rate" -> Strategy: {{"name": "Suggest Reviewing Prospecting Queries", "target_component": "Operator / Environment Config", "action_details": {{"action": "SUGGEST_CONFIG_UPDATE", "description": "Low new prospect rate. Suggest reviewing and refining the PROSPECTING_QUERIES list in the environment variables. Consider more specific or diverse queries targeting different industries or signals. Test new queries manually if possible.", "parameter_name": "PROSPECTING_QUERIES", "suggested_value": "[Example new query 1, Example new query 2]", "expected_outcome_metric": "Increase in `new_prospects_24h`."}}}}
    - Problem: "Report Generation Failing Frequently" -> Strategy: {{"name": "Suggest Checking ODR Service & LLM Model", "target_component": "Operator / ODR Service / Config", "action_details": {{"action": "OPERATOR_CHECK", "description": "High report failures. Operator should: 1. Check logs for the 'odr-service' container for errors. 2. Check logs for the 'app' container (ReportGenerator worker) for specific failure reasons. 3. Consider suggesting a switch to a potentially more stable/cheaper LLM via STANDARD/PREMIUM_REPORT_MODEL env vars if errors seem model-related.", "expected_outcome_metric": "Decrease in `failed_reports_24h`."}}}}
    """
    logger.info(f"[MCOL] Generating strategies for problem: {problem} (Mode: SUGGEST)")
    try:
        llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest")
    except (NoValidApiKeyError, APIKeyInvalidError) as key_err:
         logger.error(f"[MCOL] Cannot generate strategies, API key issue detected: {key_err}")
         return None
    except Exception as e:
         logger.error(f"[MCOL] Error calling LLM for strategy generation: {e}", exc_info=True)
         return None

    # (Parsing logic remains the same)
    strategies = None
    # ... (parsing logic as before) ...
    if llm_response:
        if isinstance(llm_response, list): strategies = llm_response
        elif isinstance(llm_response.get("strategies"), list): strategies = llm_response["strategies"]
        elif isinstance(llm_response.get("raw_inference"), str):
            try:
                raw_inf = llm_response["raw_inference"]
                json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_inf, re.IGNORECASE | re.DOTALL)
                json_str = json_match.group(1).strip() if json_match else raw_inf.strip()
                start_index = json_str.find('[')
                end_index = json_str.rfind(']')
                if start_index != -1 and end_index != -1:
                    json_str = json_str[start_index : end_index + 1]
                    parsed_raw = json.loads(json_str)
                    if isinstance(parsed_raw, list): strategies = parsed_raw
                else: logger.error(f"[MCOL] Could not find valid JSON list structure in raw inference.")
            except (json.JSONDecodeError, AttributeError, IndexError) as parse_e:
                logger.error(f"[MCOL] LLM raw inference is not valid JSON list. Error: {parse_e}. Response: {raw_inf[:500]}...")

    # Validate strategy structure
    if strategies and all(isinstance(s, dict) and "name" in s and "action_details" in s and "target_component" in s for s in strategies):
        # Filter out any accidental KeyAcquirer suggestions
        strategies = [s for s in strategies if "keyacquirer" not in s.get("target_component", "").lower() and s.get("action_details", {}).get("worker_name") != "key_acquirer"]
        logger.info(f"[MCOL] LLM generated {len(strategies)} valid strategies (post-filter).")
        return strategies[:MCOL_MAX_STRATEGIES]
    else:
        logger.error(f"[MCOL] Failed to get valid strategies from LLM or validation failed. Response: {llm_response}")
        return None


def choose_strategy(strategies: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Selects the first valid strategy."""
    # (Function unchanged)
    if not strategies: return None
    chosen = strategies[0]
    logger.info(f"[MCOL] Choosing strategy: {chosen['name']}")
    return chosen


async def execute_strategy(db: AsyncSession, strategy: Dict[str, str]) -> Dict[str, Any]:
    """Formats the strategy as a suggestion for the operator."""
    action_details = strategy.get("action_details", {})
    action_type = action_details.get("action", "UNKNOWN_ACTION")
    strategy_name = strategy.get("name", "Unknown Strategy")
    target_component = strategy.get("target_component", "Unknown Target")

    logger.info(f"[MCOL] Processing strategy: {strategy_name} (Target: {target_component}, Action: {action_type}) - Mode: SUGGEST (Forced)")

    # Format the suggestion clearly for the operator log
    suggestion_output = f"""
**MCOL Suggestion for Operator**
---------------------------------
**Priority Problem:** {strategy.get('problem_context', 'N/A')}
**Chosen Strategy:** {strategy_name}
**Target Component:** `{target_component}`
**Action Type:** `{action_type}`

**Suggested Action/Analysis:**
{action_details.get('description', json.dumps(action_details))}
"""
    # Add specific parameter suggestions if available
    if action_type == "SUGGEST_CONFIG_UPDATE":
        param = action_details.get('parameter_name')
        value = action_details.get('suggested_value')
        if param and value is not None:
            suggestion_output += f"\n**Parameter:** `{param}`\n**Suggested Value:** `{value}`"

    result_status = "SUGGESTED"
    result_message = suggestion_output.strip()

    # Log the suggestion clearly at WARNING level for visibility
    logger.warning(f"[MCOL SUGGESTION] {result_message}")

    return {
        "status": result_status,
        "result": result_message,
        "parameters": strategy # Store original strategy details for context
    }


# --- Main MCOL Agent Cycle ---
async def run_mcol_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single Monitor -> Analyze -> Strategize -> Suggest cycle."""
    if shutdown_event.is_set(): return
    logger.info(f"[MCOL] Starting analysis cycle...")
    client: Optional[httpx.AsyncClient] = None
    current_snapshot: Optional[models.KpiSnapshot] = None
    decision_log_id: Optional[int] = None
    session_committed = False # Track if commit happened

    try:
        # 0. Pre-Cycle Check: Single API Key Health (CRITICAL)
        key_health_result = await check_single_api_key_health(db)
        if key_health_result:
            action_to_log = key_health_result['executed_action']
            decision = await crud.log_mcol_decision(
                db, kpi_snapshot_id=None,
                priority_problem=key_health_result["problem"],
                analysis_summary=key_health_result["reasoning"],
                chosen_action=action_to_log['parameters']['name'],
                action_status=action_to_log['status'],
                action_result=action_to_log['result'],
                action_parameters=action_to_log.get("parameters")
            )
            if decision:
                await db.commit()
                session_committed = True
                logger.critical(f"[MCOL] Logged CRITICAL API key suggestion (Log ID: {decision.log_id}). Cycle paused.")
            else:
                 logger.error("[MCOL] Failed to log critical API key suggestion.")
                 await db.rollback() # Rollback if logging failed
            return # Stop cycle if key is unhealthy

        # 1. Monitor: Create KPI Snapshot
        current_snapshot = await crud.create_kpi_snapshot(db)
        if not current_snapshot:
             logger.error("[MCOL] Failed to create KPI snapshot. Skipping cycle.")
             await db.rollback()
             return
        await db.commit() # Commit snapshot separately
        session_committed = True
        kpi_str = format_kpis_for_llm(current_snapshot)
        logger.info(f"[MCOL] KPI Snapshot {current_snapshot.snapshot_id} created.")
        logger.debug(f"[MCOL] KPIs: {kpi_str}")

        # 2. Analyze & Prioritize (Non-Key Issues)
        client = await get_httpx_client()
        if not client:
             logger.error("[MCOL] Failed to create HTTP client for analysis. Skipping cycle.")
             return
        analysis = await analyze_performance_and_prioritize(client, kpi_str)
        await client.aclose() # Close client after use
        client = None # Reset client variable

        if not analysis:
            logger.info("[MCOL] No critical non-key problems identified or analysis failed. Ending cycle.")
            return

        # Log initial decision phase
        decision = await crud.log_mcol_decision(
            db, kpi_snapshot_id=current_snapshot.snapshot_id,
            priority_problem=analysis["problem"], analysis_summary=analysis["reasoning"],
            action_status='ANALYZED'
        )
        if not decision:
             logger.error("[MCOL] Failed to log initial analysis. Aborting cycle.")
             await db.rollback()
             return
        await db.commit() # Commit analysis phase
        session_committed = True
        decision_log_id = decision.log_id

        # 3. Strategize
        client = await get_httpx_client() # New client for strategy generation
        if not client:
             logger.error("[MCOL] Failed to create HTTP client for strategy generation. Aborting cycle.")
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="Internal HTTP client error.")
             await db.commit()
             return
        strategies = await generate_solution_strategies(client, analysis["problem"], analysis["reasoning"], kpi_str)
        await client.aclose()
        client = None

        if not strategies:
            await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="LLM failed to generate valid strategies.")
            await db.commit()
            session_committed = True
            logger.error("[MCOL] Failed to generate strategies. Ending cycle.")
            return

        # Store strategies and choose one
        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=json.dumps(strategies))
        chosen_strategy = choose_strategy(strategies) # Choose *after* logging generated strategies
        if not chosen_strategy:
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="Strategy choice failed.")
             await db.commit()
             session_committed = True
             logger.error("[MCOL] Strategy choice failed unexpectedly. Ending cycle.")
             return

        # Log chosen strategy
        chosen_strategy['problem_context'] = analysis["problem"] # Add context
        await crud.update_mcol_decision_log(db, decision_log_id, chosen_action=chosen_strategy['name'])
        await db.commit()
        session_committed = True

        # 5. Execute (Suggest)
        execution_result = await execute_strategy(db, chosen_strategy)

        # Update the log with the final suggestion outcome
        await crud.update_mcol_decision_log(
            db, decision_log_id,
            action_status=execution_result["status"],
            action_result=execution_result["result"],
            action_parameters=execution_result.get("parameters")
        )
        await db.commit() # Commit final outcome
        session_committed = True

        logger.info(f"[MCOL] Cycle finished. Suggestion logged: {execution_result['status']}")

    except Exception as e:
        logger.error(f"[MCOL] CRITICAL Error during MCOL cycle: {e}", exc_info=True)
        if not session_committed: # Rollback only if no commits were successful
             await db.rollback()
        # Attempt to log the failure if a log entry was created
        if decision_log_id:
            try:
                # Use a new session for logging the error to avoid transaction issues
                log_session = await get_worker_session()
                try:
                    await crud.update_mcol_decision_log(log_session, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {traceback.format_exc(limit=1000)}")
                    await log_session.commit()
                except Exception as db_err:
                     logger.error(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                     await log_session.rollback()
                finally:
                     await log_session.close()
            except Exception as log_session_err:
                 logger.error(f"[MCOL] Failed to get session for logging cycle error: {log_session_err}")

    finally:
        if client: # Ensure client is closed if an error occurred mid-process
            await client.aclose()