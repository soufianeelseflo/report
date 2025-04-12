
# autonomous_agency/app/agents/mcol_agent.py
import asyncio
import json
import datetime
import subprocess
import os
import re
import traceback
import logging
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
import httpx
# from lemonsqueezy import LemonSqueezy # Keep if direct LS API calls needed later

# Corrected relative imports for package structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api
except ImportError:
    print("[MCOL Agent] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.agents.agent_utils import get_httpx_client, call_llm_api

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS or 300
MCOL_IMPLEMENTATION_MODE = settings.MCOL_IMPLEMENTATION_MODE or "SUGGEST"
WEBSITE_OUTPUT_DIR = "/app/static_website" # Served by FastAPI
API_KEY_LOW_THRESHOLD = getattr(settings, 'API_KEY_LOW_THRESHOLD', 5) # Threshold for warning

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    if not snapshot: return "No KPI data available."
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    # Use getattr for safe access to potentially new fields
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
    Analyze the following system performance data for Acumenis, an autonomous AI reporting agency aiming for $10k/72h.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL bottleneck currently preventing the achievement of the primary goal ($10k in 72h). Consider the entire funnel: Key Acquisition -> Prospecting -> Email Outreach -> Website Visit -> Order Attempt -> Payment Success -> Report Generation -> Delivery. Look for zero values or critical failures (e.g., Active API Keys=0, Orders=0, High Bounce Rate).
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
    System: Acumenis Agency. FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, KeyAcquirer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (configured). Website: Served via FastAPI static files (target dir: /app/static_website) at {settings.AGENCY_BASE_URL}. Budget: $5/month (data center proxies). Uses free tiers/acquired keys aggressively. Core LLM: Acumenis Prime (You). MCOL Mode: {MCOL_IMPLEMENTATION_MODE}.
    Code Structure: /app/ contains main.py, agents/, db/, core/, workers/, api/endpoints/. /app/open-deep-research/ contains the external tool. Migrations via Alembic.
    """
    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate 2-3 diverse, actionable, and creative strategies to solve the identified problem for Acumenis Agency. Prioritize AI/automation, budget constraints ($5 proxies), and the $10k/72h goal. Since MCOL_IMPLEMENTATION_MODE='{MCOL_IMPLEMENTATION_MODE}', focus on generating *clear, actionable suggestions* for a human operator. If code changes are needed, describe the logic clearly. If prompt tuning is needed, provide example improved prompts.

    Instructions:
    1. Provide 'name' (concise action verb phrase) and 'description' (detailed steps/logic/prompt examples for the operator) for each strategy.
    2. Specify target files/agents involved.
    3. Output ONLY a JSON list of strategy objects.

    Example Strategies (Focus on Operator Suggestions):
    - Problem: "Zero Active API Keys (KeyAcquirer Failure)" -> Strategy: {{"name": "Diagnose & Address KeyAcquirer Failure", "description": "Suggest Operator: 1. Review KeyAcquirer logs (`run_key_acquirer_worker.py`) for specific errors (CAPTCHA, 403, selector failure). 2. If CAPTCHA/IP block: Manually acquire 5-10 OpenRouter keys and add them via a DB script/tool. Update `settings.KEY_ACQUIRER_RUN_ON_STARTUP` to False temporarily. 3. If selector failure: Identify the broken CSS selector on the temp email or OpenRouter site and update the corresponding setting (`TEMP_EMAIL_SELECTOR`, `API_KEY_DISPLAY_SELECTOR`) in `.env` or `config.py`. 4. Verify proxy list (`settings.PROXY_LIST`) contains valid, working data center proxies."}}
    - Problem: "High Email Bounce Rate (>15%)" -> Strategy: {{"name": "Suggest Email Prompt & Validation Tuning", "description": "Suggest Operator: 1. Review EmailMarketer logs for bounce reasons. 2. Consider integrating a free email validation service (e.g., search 'free email validation api'). Modify `prospect_researcher.py` to call this API after guessing email and update status to 'INVALID_EMAIL' if needed. 3. Suggest A/B testing email subject lines/body prompts. Example improved prompt element: 'Focus less on 'discovered signal', more on direct value prop like 'AI competitor report for [Company] in hours'.' Log suggested prompt variations."}}
    - Problem: "Zero Orders Created" -> Strategy: {{"name": "Suggest Website CRO Analysis (Multimodal)", "description": "Suggest Operator: 1. Manually test the order flow (`/order` page -> Lemon Squeezy checkout). 2. Verify LS webhook setup. 3. Provide screenshots of `index.html` and `order.html` to Acumenis Prime (me) via a future interface or manual input. I will analyze for CRO issues (clarity, trust, friction) using vision capabilities and provide specific HTML/CSS suggestions."}}
    """
    logger.info(f"[MCOL] Generating strategies for problem: {problem}")
    llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest") # Use capable model

    strategies = None
    if llm_response:
        if isinstance(llm_response, list): strategies = llm_response
        elif isinstance(llm_response.get("strategies"), list): strategies = llm_response["strategies"]
        elif isinstance(llm_response.get("raw_inference"), str):
            try:
                # Attempt to clean and parse JSON from raw inference
                raw_inf = llm_response["raw_inference"]
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_inf, re.IGNORECASE | re.DOTALL)
                json_str = json_match.group(1).strip() if json_match else raw_inf.strip()
                # Further clean potential leading/trailing non-JSON chars
                json_str = json_str[json_str.find('['):json_str.rfind(']')+1]
                parsed_raw = json.loads(json_str)
                if isinstance(parsed_raw, list): strategies = parsed_raw
            except (json.JSONDecodeError, AttributeError) as parse_e:
                logger.error(f"[MCOL] LLM raw inference is not valid JSON list. Error: {parse_e}. Response: {raw_inf[:500]}...")

    if strategies and all(isinstance(s, dict) and "name" in s and "description" in s for s in strategies):
        logger.info(f"[MCOL] LLM generated {len(strategies)} strategies.")
        return strategies
    else:
        logger.error(f"[MCOL] Failed to get valid strategies from LLM. Response: {llm_response}")
        return None

def choose_strategy(strategies: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Selects the best strategy (simple: pick the first one for now)."""
    if not strategies: return None
    # Future enhancement: Could use LLM to rank strategies based on feasibility/impact/cost.
    chosen = strategies[0]
    logger.info(f"[MCOL] Choosing strategy: {chosen['name']}")
    return chosen

async def implement_strategy(client: httpx.AsyncClient, strategy: Dict[str, str]) -> Dict[str, Any]:
    """Formats the chosen strategy as a suggestion for the operator."""
    action_details = {"name": strategy["name"], "description": strategy["description"]}
    strategy_name = strategy.get("name", "Unknown Strategy")
    strategy_desc = strategy.get("description", "No description provided.")

    logger.info(f"[MCOL] Processing strategy (Suggest Mode): {strategy_name}")

    # Format the suggestion clearly for the operator log
    suggestion_output = f"""
    **MCOL Suggestion for Operator**

    **Strategy:** {strategy_name}

    **Problem Context:** (Inferred from previous MCOL step - add if available)

    **Suggested Action/Analysis:**
    {strategy_desc}
    """

    # Log the suggestion
    # The actual logging to DB happens in the main cycle using this result
    implementation_result = {
        "status": "SUGGESTED",
        "result": suggestion_output.strip() # Store the formatted suggestion
    }

    logger.info(f"[MCOL] Implementation outcome for '{strategy_name}': {implementation_result['status']}")
    return {
        "status": implementation_result["status"],
        "result": implementation_result["result"],
        "parameters": action_details # Store original strategy details for context
    }

async def check_api_key_status(db: AsyncSession) -> Optional[Dict[str, Any]]:
    """Checks the status of API keys and returns a suggestion if low."""
    try:
        active_keys_count = await crud.count_active_api_keys(db, provider=settings.LLM_PROVIDER or "openrouter")
        logger.info(f"[MCOL] Active API Keys Check: Found {active_keys_count} active keys.")
        if active_keys_count < API_KEY_LOW_THRESHOLD:
            problem = f"Low Active API Keys ({active_keys_count} < {API_KEY_LOW_THRESHOLD})"
            reasoning = "Insufficient active API keys risk halting all LLM-dependent operations (MCOL, Prospecting, Marketing, Reporting)."
            strategy_desc = f"Suggest Operator: 1. Manually acquire {API_KEY_LOW_THRESHOLD - active_keys_count + 5} new OpenRouter API keys. 2. Add keys via DB script/tool. 3. Consider running the KeyAcquirer worker (`/control/start/key_acquirer`) if automated acquisition was previously successful, but monitor its logs closely for failures (CAPTCHA/blocks)."
            strategy = {"name": "Address Low API Key Count", "description": strategy_desc}
            suggestion = await implement_strategy(None, strategy) # No client needed for suggestion formatting
            return {
                "problem": problem,
                "reasoning": reasoning,
                "suggestion": suggestion # Contains status, result, parameters
            }
    except Exception as e:
        logger.error(f"[MCOL] Error checking API key status: {e}", exc_info=True)
    return None


# --- Main MCOL Agent Cycle ---
async def run_mcol_cycle(db: AsyncSession, shutdown_event: asyncio.Event):
    """Runs a single Monitor -> Analyze -> Strategize -> Implement cycle."""
    if shutdown_event.is_set(): return
    logger.info(f"[MCOL] Starting cycle at {datetime.datetime.now(datetime.timezone.utc)}")
    client: Optional[httpx.AsyncClient] = None
    current_snapshot: Optional[models.KpiSnapshot] = None
    decision_log_id: Optional[int] = None
    suggestion_to_log = None

    try:
        # 0. Pre-Cycle Check: API Key Status (High Priority)
        key_status_suggestion = await check_api_key_status(db)
        if key_status_suggestion:
            logger.warning(f"[MCOL] Priority Issue Detected: {key_status_suggestion['problem']}")
            suggestion_to_log = key_status_suggestion['suggestion']
            # Log this critical suggestion immediately
            decision = await crud.log_mcol_decision(
                db, kpi_snapshot_id=None, # No full KPI cycle run yet
                priority_problem=key_status_suggestion["problem"],
                analysis_summary=key_status_suggestion["reasoning"],
                chosen_action=suggestion_to_log['parameters']['name'],
                action_status=suggestion_to_log['status'],
                action_result=suggestion_to_log['result'],
                action_parameters=suggestion_to_log.get("parameters")
            )
            await db.commit()
            logger.info(f"[MCOL] Logged critical API key suggestion (Log ID: {decision.log_id}). Proceeding to main cycle.")
            # Continue to main cycle, but this critical suggestion is logged.

        # 1. Monitor: Create KPI Snapshot
        current_snapshot = await crud.create_kpi_snapshot(db)
        await db.commit() # Commit snapshot separately
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

        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=strategies)
        await db.commit() # Commit generated strategies

        # 4. Choose Strategy
        chosen_strategy = choose_strategy(strategies)
        if not chosen_strategy:
             # Should not happen if strategies were generated, but handle defensively
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY', action_result="Strategy list was empty after generation.")
             await db.commit()
             logger.error("[MCOL] Strategy list empty after generation. Ending cycle.")
             if client: await client.aclose()
             return

        await crud.update_mcol_decision_log(db, decision_log_id, chosen_action=chosen_strategy['name'])
        await db.commit() # Commit chosen strategy

        # 5. Implement (Suggest Mode)
        implementation_result = await implement_strategy(client, chosen_strategy)

        # Update the log with the final suggestion/outcome
        await crud.update_mcol_decision_log(
            db, decision_log_id,
            action_status=implementation_result["status"],
            action_result=implementation_result["result"],
            action_parameters=implementation_result.get("parameters")
        )
        await db.commit() # Commit final outcome

        logger.info(f"[MCOL] Cycle finished. Action status: {implementation_result['status']}")

    except Exception as e:
        logger.error(f"[MCOL] CRITICAL Error during MCOL cycle: {e}", exc_info=True)
        if decision_log_id:
            try:
                # Attempt to log the cycle failure to the existing log entry
                await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {traceback.format_exc(limit=500)}")
                await db.commit()
            except Exception as db_err:
                 logger.error(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                 await db.rollback() # Rollback if logging the error fails
        else:
             # If no decision log entry was even created
             await db.rollback()
    finally:
        if client:
            await client.aclose()
