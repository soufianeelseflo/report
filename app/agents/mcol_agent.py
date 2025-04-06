import asyncio
import json
import datetime
import subprocess
import os

from sqlalchemy.ext.asyncio import AsyncSession
import httpx

from app.core.config import settings
from app.db import crud, models
from app.agents.agent_utils import get_httpx_client
from app.agents.prospect_researcher import call_llm_api # Reuse LLM utility

# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = 300 # How often MCOL runs its cycle (e.g., 5 minutes)
MCOL_IMPLEMENTATION_MODE = "SUGGEST" # "SUGGEST" or "ATTEMPT_EXECUTE" (Use SUGGEST for safety)

# --- Core MCOL Functions ---

def format_kpis_for_llm(snapshot: models.KpiSnapshot) -> str:
    """Formats KPI snapshot data into a string for LLM analysis."""
    if not snapshot: return "No KPI data available."
    lines = [f"KPI Snapshot (Timestamp: {snapshot.timestamp}):"]
    lines.append(f"- Reports: Pending={snapshot.pending_reports}, Processing={snapshot.processing_reports}, Completed(24h)={snapshot.completed_reports_24h}, Failed(24h)={snapshot.failed_reports_24h}, AvgTime={snapshot.avg_report_time_seconds:.2f}s")
    lines.append(f"- Prospecting: New(24h)={snapshot.new_prospects_24h}")
    lines.append(f"- Email: Sent(24h)={snapshot.emails_sent_24h}, ActiveAccounts={snapshot.active_email_accounts}, Deactivated(24h)={snapshot.deactivated_accounts_24h}, BounceRate(24h)={snapshot.bounce_rate_24h:.2f}%")
    lines.append(f"- Revenue(24h): ${snapshot.revenue_24h:.2f}")
    return "\n".join(lines)

async def analyze_performance_and_prioritize(client: httpx.AsyncClient, kpi_data_str: str) -> Optional[Dict[str, str]]:
    """Uses LLM to analyze KPIs, identify the biggest problem, and explain why."""
    primary_goal = "Achieve $1000 revenue within 72 hours, then sustain growth via report sales and client acquisition."
    system_context = """
    System Overview: Autonomous agency using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses external 'open-deep-research' tool), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation). Deployed via Docker.
    """
    prompt = f"""
    Analyze the following system performance data for an autonomous AI reporting agency.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL problem currently hindering the achievement of the primary goal, based *only* on the provided KPIs and context.
    2. Briefly explain the reasoning for selecting this problem (impact on revenue/growth).
    3. Respond ONLY with a JSON object containing two keys: "problem" (string description of the problem) and "reasoning" (string explanation).
    Example: {{"problem": "Zero revenue generated despite email activity", "reasoning": "The core goal is revenue. Emails are being sent but no reports are being purchased, indicating a failure in conversion, pricing, or payment processing."}}
    If KPIs look healthy and aligned with the goal, respond: {{"problem": "None", "reasoning": "Current KPIs indicate progress towards goal."}}
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
            print("[MCOL] LLM analysis indicates no critical problems currently.")
            return None # No problem identified
    else:
        print("[MCOL] Failed to get valid analysis from LLM.")
        return None

async def generate_solution_strategies(client: httpx.AsyncClient, problem: str, reasoning: str, kpi_data_str: str) -> Optional[List[Dict[str, str]]]:
    """Uses LLM to generate potential solution strategies for the identified problem."""
    system_context = """
    System: FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer. Core tool: 'open-deep-research'. Budget: $10/month (proxies). Uses free tiers aggressively.
    Code Structure: /app/autonomous_agency/app/ contains main.py, agents/, db/, core/, workers/. /app/open-deep-research/ contains the external tool. Migrations via Alembic.
    """
    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate 2-3 diverse, actionable, and creative strategies to solve the identified problem, suitable for an autonomous AI agent to potentially implement or suggest. Prioritize strategies leveraging AI/automation and adhering to extreme budget constraints (free tiers, minimal cost).

    Instructions:
    1. For each strategy, provide a concise 'name' and a detailed 'description' outlining the steps involved.
    2. Consider modifying existing agent logic, generating new code (Python/JS/HTML), integrating external free APIs, adjusting configurations, or even suggesting changes to the core 'open-deep-research' tool usage.
    3. If code generation/modification is involved, specify the target file(s) and the nature of the change.
    4. Output ONLY a JSON list of strategy objects. Each object must have "name" and "description" keys.
    Example:
    [
        {{"name": "Implement Basic Payment Link", "description": "Research Stripe Payment Links. Use LLM to generate a FastAPI endpoint in 'main.py' to create a payment link based on report type. Modify frontend (if exists) or email template to include the link. Requires Stripe account setup (manual)."}},
        {{"name": "Add Report Delivery Email", "description": "Modify 'report_generator.py'. After status='COMPLETED', use 'aiosmtplib' (like in 'email_marketer.py') and an available EmailAccount to send the generated report file at 'report_output_path' to the 'client_email' from the 'ReportRequest' table."}}
    ]
    """
    print(f"[MCOL] Generating strategies for problem: {problem}")
    llm_response = await call_llm_api(client, prompt)

    # LLM response might be a list directly, or nested within a key
    strategies = None
    if llm_response and isinstance(llm_response, list):
        strategies = llm_response
    elif llm_response and isinstance(llm_response.get("strategies"), list): # Check common nesting
        strategies = llm_response["strategies"]

    if strategies and all(isinstance(s, dict) and "name" in s and "description" in s for s in strategies):
        print(f"[MCOL] LLM generated {len(strategies)} strategies.")
        return strategies
    else:
        print("[MCOL] Failed to get valid strategies from LLM.")
        # Attempt to parse from raw_inference as fallback? Maybe too complex here.
        return None

def choose_strategy(strategies: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """Selects the best strategy (simple: pick the first one for now)."""
    if not strategies:
        return None
    # Future: Implement more sophisticated selection based on estimated complexity, impact, risk
    print(f"[MCOL] Choosing strategy: {strategies[0]['name']}")
    return strategies[0]

async def implement_strategy(client: httpx.AsyncClient, strategy: Dict[str, str]) -> Dict[str, Any]:
    """Attempts to implement the chosen strategy."""
    action_details = {"name": strategy["name"], "description": strategy["description"]}
    print(f"[MCOL] Attempting implementation for strategy: {strategy['name']}")

    if MCOL_IMPLEMENTATION_MODE == "SUGGEST":
        # Log the suggestion for human review
        print(f"[MCOL] SUGGEST Mode: Suggesting implementation of '{strategy['name']}'. Description: {strategy['description']}")
        return {
            "status": "SUGGESTED",
            "result": f"Suggestion logged for human review: Implement '{strategy['name']}'. Details: {strategy['description']}",
            "parameters": action_details
        }
    elif MCOL_IMPLEMENTATION_MODE == "ATTEMPT_EXECUTE":
        # --- Advanced & Risky: Attempt automated implementation ---
        print(f"[MCOL] ATTEMPT_EXECUTE Mode: Attempting automated implementation of '{strategy['name']}'")
        # 1. Does the strategy involve code generation/modification?
        #    Use LLM again or regex to parse strategy['description'] for keywords like "generate code", "modify file", "add endpoint", "create script".
        # 2. If code needed:
        #    - Call LLM with a specific prompt to generate the code snippet/diff.
        #      Prompt: "Generate the Python code snippet for FastAPI endpoint described here: [strategy description snippet]. Ensure it fits within '/app/autonomous_agency/app/main.py'."
        #    - Parse LLM response for the code block.
        #    - **CRITICAL RISK:** Attempt to write the code to the target file system path. This requires container permissions and careful path handling.
        #      Example (highly simplified and dangerous):
        #      try:
        #          target_file = "/app/autonomous_agency/app/main.py" # Determine target path
        #          code_snippet = parsed_llm_code
        #          with open(target_file, "a") as f: # Append vs Modify? Needs logic
        #              f.write("\n# --- MCOL Auto-Generated Code Start ---\n")
        #              f.write(code_snippet)
        #              f.write("\n# --- MCOL Auto-Generated Code End ---\n")
        #          print(f"[MCOL] Attempted to write code to {target_file}")
        #          # Need to trigger reload/restart?
        #          # Call self.control_api('/control/restart/worker_name') ?
        #      except Exception as e:
        #          print(f"[MCOL] Failed to write generated code: {e}")
        #          return {"status": "FAILED", "result": f"Failed to write generated code: {e}", "parameters": action_details}
        # 3. If DB migration needed:
        #    - Attempt `alembic revision --autogenerate ...` via subprocess.
        #    - Attempt `alembic upgrade head` via subprocess. Handle errors.
        # 4. If external API call needed (e.g., deploy static site):
        #    - Execute the necessary commands via subprocess or httpx calls.

        # Placeholder for complex execution logic
        return {
            "status": "FAILED",
            "result": "ATTEMPT_EXECUTE mode is highly experimental and not fully implemented for safety.",
            "parameters": action_details
        }
    else:
        return {"status": "FAILED", "result": f"Unknown implementation mode: {MCOL_IMPLEMENTATION_MODE}", "parameters": action_details}


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
            return

        await crud.update_mcol_decision_log(db, decision_log_id, generated_strategy=json.dumps(strategies)) # Log all generated strategies
        await db.commit()

        # 4. Choose Strategy
        chosen_strategy = choose_strategy(strategies)
        if not chosen_strategy: # Should not happen if strategies exist
             await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_STRATEGY')
             await db.commit()
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
            action_parameters=implementation_result["parameters"]
        )
        await db.commit()

        # 6. Verify (Basic - relies on next cycle's KPI snapshot)
        # More advanced: Could trigger specific checks or short-term monitoring here.
        print(f"[MCOL] Cycle finished. Action status: {implementation_result['status']}")


    except Exception as e:
        print(f"[MCOL] Error during MCOL cycle: {e}")
        # Log error to DB if possible
        if decision_log_id:
            try:
                await crud.update_mcol_decision_log(db, decision_log_id, action_status='FAILED_CYCLE', action_result=f"Cycle error: {e}")
                await db.commit()
            except Exception as db_err:
                 print(f"[MCOL] Failed to log cycle error to DB: {db_err}")
                 await db.rollback() # Rollback logging attempt
        else:
             await db.rollback() # Rollback snapshot if cycle failed early
    finally:
        if client:
            await client.aclose()