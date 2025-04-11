import asyncio
import json
import datetime
import subprocess
import os
import re
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from lemonsqueezy import LemonSqueezy

# Corrected relative imports for package structure
from Acumenis.app.core.config import settings
from Acumenis.app.db import crud, models
from Acumenis.app.agents.agent_utils import get_httpx_client, call_llm_api

# --- MCOL Configuration ---
MCOL_ANALYSIS_INTERVAL_SECONDS = settings.MCOL_ANALYSIS_INTERVAL_SECONDS
MCOL_IMPLEMENTATION_MODE = settings.MCOL_IMPLEMENTATION_MODE
WEBSITE_OUTPUT_DIR = "/app/static_website" # Served by FastAPI

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
    System Overview: Autonomous agency 'Acumenis' using FastAPI, SQLAlchemy, PostgreSQL. Agents: ReportGenerator (uses 'open-deep-research', delivers via email), ProspectResearcher (signal-based, LLM inference), EmailMarketer (LLM personalized emails, SMTP rotation), MCOL (self-improvement). Deployed via Docker at {settings.AGENCY_BASE_URL}. Payment via Lemon Squeezy (checkout links + webhook). Website serving via FastAPI static files.
    """
    prompt = f"""
    Analyze the following system performance data for Acumenis, an autonomous AI reporting agency aiming for $10k/72h.
    Primary Goal: {primary_goal}
    System Context: {system_context}
    Current KPIs:
    {kpi_data_str}

    Instructions:
    1. Identify the single MOST CRITICAL bottleneck currently preventing the achievement of the primary goal ($10k in 72h). Consider the entire funnel: Prospecting -> Email Outreach -> Website Visit -> Order Attempt -> Payment Success -> Report Generation -> Delivery. Look at KPIs like Orders(24h), Revenue(24h), New Prospects, Email Sent/Bounce Rate, Pending Reports.
    2. Briefly explain the reasoning for selecting this problem (impact on revenue/growth).
    3. Respond ONLY with a JSON object containing "problem" and "reasoning".
    Example Problems: "No Website Generated Yet", "Zero Orders Created (Website/Payment Issue?)", "Low Prospect Acquisition Rate", "High Email Bounce Rate (>20%)", "Payment Webhook Not Triggering Report Creation", "Report Generation Failing Frequently".
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
    System: Acumenis Agency. FastAPI, SQLAlchemy, PostgreSQL, Docker. Agents: ReportGenerator, ProspectResearcher, EmailMarketer, MCOL. Core tool: 'open-deep-research'. Payment: Lemon Squeezy (API Key, Variant IDs, Webhook Secret configured). Website: Served via FastAPI static files (target dir: /app/static_website) at {settings.AGENCY_BASE_URL}. Budget: $10/month (proxies). Uses free tiers aggressively.
    Code Structure: /app/Acumenis/app/ contains main.py, agents/, db/, core/, workers/, api/endpoints/. /app/open-deep-research/ contains the external tool. Migrations via Alembic. Static website files go in /app/static_website.
    """
    prompt = f"""
    Problem Diagnosis:
    - Identified Problem: "{problem}"
    - Reasoning: "{reasoning}"
    - Current KPIs: {kpi_data_str}
    - System Context: {system_context}

    Objective: Generate 2-3 diverse, actionable, and creative strategies to solve the identified problem for Acumenis Agency. Prioritize AI/automation, Lemon Squeezy integration, budget constraints, and the $10k/72h goal. Incorporate rule-bending, algorithm-aware tactics.

    Instructions:
    1. Provide 'name' and detailed 'description' for each strategy.
    2. Specify target files and changes if code generation/modification is needed.
    3. State any required manual interactions clearly.
    4. Output ONLY a JSON list of strategy objects.

    Example Strategies (Refined & Aggressive):
    - Problem: "No Website Generated Yet" -> Strategy: {{"name": "Generate Multi-Page Website v1", "description": "Use LLM (Gemini 2.5 Pro) to generate 3 HTML files: `index.html` (Hero, How it Works, Features, Pricing Teaser, Testimonials, CTA), `pricing.html` (Detailed comparison table, FAQs, CTAs), `order.html` (Focused order form). Ensure consistent navigation, professional design (embedded CSS), SEO optimization (keywords: AI research reports, market analysis, competitor intelligence, Acumenis), and conversion focus. Order form JS on `order.html` POSTs to `/api/v1/payments/create-checkout` and redirects. Save files to `/app/static_website/`."}}
    - Problem: "Zero Orders Created" -> Strategy: {{"name": "Aggressive CRO & Payment Debug", "description": "1. Verify website files exist and JS POSTs correctly to payment endpoint. 2. Check payment endpoint logs. 3. Use LLM to analyze generated website HTML/JS for conversion bottlenecks (CTA clarity, trust signals, speed). 4. Suggest A/B tests for headlines/pricing (MCOL logs). 5. Ensure LS Variant IDs are correct. 6. Suggest manual test purchase."}}
    - Problem: "Payment Webhook Not Triggering Report Creation" -> Strategy: {{"name": "Debug Payment Webhook & DB Insertion", "description": "1. Verify LS webhook points to `{settings.AGENCY_BASE_URL}/api/v1/payments/webhook`. 2. Check `payments.py -> lemon_squeezy_webhook` logs for signature/processing errors. 3. Ensure webhook secret matches. 4. Verify `crud.py -> create_report_request_from_webhook` parses payload/custom data and inserts with 'PENDING' status. Check DB logs."}}
    - Problem: "Low Prospect Acquisition Rate" -> Strategy: {{"name": "Hyper-Targeted LinkedIn Outreach Prep", "description": "Modify 'ProspectResearcher': Use LLM to identify 2-3 specific, high-ranking executives (CEO, VP Marketing, Head of Strategy) at companies identified via signals. Store names/titles/LinkedIn URLs (if findable) in `Prospect` table. MCOL Action: Log detailed suggestions for manual operator: 'Connect with [Name, Title] at [Company] on LinkedIn. Reference [Specific Signal/Pain Point]. Suggest discussing how Acumenis rapid AI reports address [Benefit].'"}}
    - Problem: "High Email Bounce Rate (>15%)" -> Strategy: {{"name": "Integrate Free Email Validation & Aggressive List Pruning", "description": "Research free tier email validation APIs. Use LLM to generate code modification for 'EmailMarketer -> process_email_batch' to call the API before `generate_personalized_email`. If invalid, update prospect status to 'INVALID_EMAIL' and skip. Requires adding API key to .env."}}
    - Problem: "Low Website Traffic/SEO Ranking" -> Strategy: {{"name": "AI-Driven SEO Content & Signal Boost", "description": "Use LLM (Gemini 2.5 Pro) to generate 2-3 high-quality blog posts relevant to 'AI market research', 'competitive analysis tools', 'Acumenis reports'. Save as HTML in `/app/static_website/blog/`. Modify website HTML to link to them. MCOL Action: Simulate social sharing/backlinks by prompting LLM to generate realistic-sounding forum posts/comments mentioning Acumenis/blog posts (log these for potential manual posting)."}}
    """
    print(f"[MCOL] Generating strategies for problem: {problem}")
    llm_response = await call_llm_api(client, prompt, model="google/gemini-1.5-pro-latest")

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
    # Note: Website generation is removed as per instructions.

    async def analyze_website_conversion():
        """Placeholder: Reads website files and asks LLM for conversion improvements."""
        print("[MCOL] Attempting to analyze existing website files for conversion...")
        results = []
        files_to_analyze = ["index.html", "order.html"] # Focus on key pages
        for filename in files_to_analyze:
            filepath = os.path.join(WEBSITE_OUTPUT_DIR, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    analysis_prompt = f"""
                    Analyze the following HTML/JS content from the '{filename}' page of the Acumenis website.
                    Identify potential conversion rate optimization (CRO) bottlenecks.
                    Suggest 2-3 specific, actionable improvements focusing on clarity, call-to-action effectiveness, trust signals, and reducing friction.
                    Keep suggestions concise. Output suggestions as a bulleted list.

                    HTML/JS Content:
                    ```html
                    {content[:5000]}
                    ```
                    """ # Limit content length sent to LLM
                    llm_response = await call_llm_api(client, analysis_prompt, model="google/gemini-1.5-flash-latest") # Use fast model
                    if llm_response and isinstance(llm_response.get("raw_inference"), str):
                        results.append(f"Analysis for {filename}:\n{llm_response['raw_inference'].strip()}")
                    else:
                        results.append(f"Analysis for {filename}: Failed to get suggestions from LLM.")
                except Exception as e:
                    results.append(f"Analysis for {filename}: Error reading file - {e}")
            else:
                results.append(f"Analysis for {filename}: File not found.")
        
        return {"status": "SUGGESTED", "result": "\n---\n".join(results)}

    async def suggest_manual_action(description: str):
        """Formats a suggestion for manual action."""
        print(f"[MCOL] Suggesting Manual Action: {description}")
        return {"status": "SUGGESTED", "result": f"Suggestion: {description}"}

    # --- Strategy Execution Mapping ---
    implementation_result = {"status": "FAILED", "result": "Strategy implementation logic not defined or failed."}
    suggestion_text = f"Suggest manually implementing strategy: {strategy['name']}. Description: {strategy['description']}"

    # --- Strategy Execution Mapping (Focus on SUGGEST mode) ---
    if MCOL_IMPLEMENTATION_MODE == "SUGGEST":
        # In SUGGEST mode, most strategies just log their description as a suggestion.
        # Specific analysis strategies might run but still result in suggestions.
        if "analyze website conversion" in strategy_name:
             implementation_result = await analyze_website_conversion()
        # Add other specific analysis handlers here if needed
        # elif "analyze email performance" in strategy_name: ...
        else:
             # Default for SUGGEST mode is just logging the strategy description
             implementation_result = await suggest_manual_action(strategy['description'])

    # --- Placeholder for Future Execution Modes ---
    # elif MCOL_IMPLEMENTATION_MODE == "EXECUTE_PROMPT_TUNING":
    #     if "tune email prompts" in strategy_name:
    #         # implementation_result = await execute_prompt_update(...) # Example
    #         pass
    #     else:
    #         implementation_result = await suggest_manual_action(strategy['description']) # Fallback to suggest
    # elif MCOL_IMPLEMENTATION_MODE == "EXECUTE_SAFE_CONFIG":
    #      # Example: Adjusting delays, enabling/disabling features via DB flags
    #      pass
    else: # Default or unknown mode -> Suggest
        implementation_result = await suggest_manual_action(strategy['description'])


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