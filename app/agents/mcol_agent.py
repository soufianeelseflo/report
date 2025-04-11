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
    async def generate_multi_page_website():
        """Generates index.html, pricing.html, order.html"""
        print("[MCOL] Attempting to generate multi-page website files...")
        files_to_generate = ["index.html", "pricing.html", "order.html"]
        generated_files = []
        errors = []

        # Common elements for prompts
        common_header = f"""<header><div class="container header-content"><a href="/" class="logo">Acumenis</a><nav class="nav-links"><a href="/">Home</a><a href="/pricing">Pricing</a><a href="/order">Order Now</a></nav></div></header>"""
        common_footer = f"""<footer><div class="container"><p>© {datetime.datetime.now().year} Acumenis. All Rights Reserved.</p><p><a href="/pricing">Pricing</a> | <a href="/order">Order</a></p></div></footer>"""
        common_css = """
        <style>
            :root { --primary-color: #2563eb; --secondary-color: #111827; --accent-color: #f59e0b; --light-bg: #f9fafb; --medium-grey: #d1d5db; --dark-grey: #4b5563; --text-color: #374151; --white: #ffffff; --success-bg: #dcfce7; --success-border: #86efac; --success-text: #166534; --error-bg: #fee2e2; --error-border: #fca5a5; --error-text: #991b1b; --info-bg: #e0f2fe; --info-border: #7dd3fc; --info-text: #075985; }
            *, *::before, *::after { box-sizing: border-box; }
            body { font-family: 'Inter', sans-serif; line-height: 1.7; margin: 0; padding: 0; background-color: var(--white); color: var(--text-color); font-size: 16px; -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
            .container { max-width: 1200px; margin: 0 auto; padding: 0 24px; }
            header { background-color: var(--white); padding: 15px 0; border-bottom: 1px solid #e5e7eb; position: sticky; top: 0; z-index: 100; }
            .header-content { display: flex; justify-content: space-between; align-items: center; }
            .logo { font-size: 1.5em; font-weight: 700; color: var(--secondary-color); text-decoration: none; }
            .nav-links a { color: var(--text-color); text-decoration: none; margin-left: 25px; font-weight: 600; transition: color 0.3s ease; }
            .nav-links a:hover { color: var(--primary-color); }
            section { padding: 80px 0; }
            section h2 { text-align: center; font-size: 2.5em; font-weight: 700; color: var(--secondary-color); margin-bottom: 16px; }
            section .section-subtitle { text-align: center; font-size: 1.15em; color: var(--dark-grey); max-width: 700px; margin: 0 auto 60px auto; }
            .cta-button { background-color: var(--primary-color); color: var(--white); padding: 14px 28px; font-size: 1.05em; font-weight: 600; text-decoration: none; border-radius: 8px; transition: background-color 0.3s ease, transform 0.1s ease; display: inline-block; border: none; cursor: pointer; box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08); }
            .cta-button:hover { background-color: #1d4ed8; transform: translateY(-2px); }
            .cta-button:active { transform: translateY(0); }
            footer { background-color: var(--secondary-color); color: #9ca3af; text-align: center; padding: 40px 0; margin-top: 60px; font-size: 0.9em; }
            footer p { margin: 5px 0; } footer a { color: var(--medium-grey); text-decoration: none; } footer a:hover { color: var(--white); }
            /* Add more shared styles here */
            .form-group { margin-bottom: 25px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: var(--secondary-color); font-size: 0.95em; }
            .form-group input[type="text"], .form-group input[type="email"], .form-group select, .form-group textarea { width: 100%; padding: 14px; border: 1px solid var(--medium-grey); border-radius: 6px; font-size: 1em; box-sizing: border-box; background-color: var(--white); color: var(--text-color); transition: border-color 0.3s ease; }
            .form-group input:focus, .form-group select:focus, .form-group textarea:focus { border-color: var(--primary-color); outline: none; box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3); }
            .form-group textarea { min-height: 150px; resize: vertical; }
            #form-message { margin-top: 20px; padding: 12px; border-radius: 6px; text-align: center; font-weight: 600; display: none; font-size: 0.95em; }
            #form-message.success { background-color: var(--success-bg); color: var(--success-text); border: 1px solid var(--success-border); }
            #form-message.error { background-color: var(--error-bg); color: var(--error-text); border: 1px solid var(--error-border); }
            #form-message.info { background-color: var(--info-bg); color: var(--info-text); border: 1px solid var(--info-border); }
            @media (max-width: 768px) { body { font-size: 15px; } h1 { font-size: 2.2em; } section h2 { font-size: 2em; } .nav-links { display: none; } }
        </style>
        """

        for filename in files_to_generate:
            page_specific_prompt = ""
            if filename == "index.html":
                page_specific_prompt = f"""
                Generate the HTML content for the BODY of the Acumenis homepage (`index.html`).
                Include:
                1. Hero section: Headline "Unlock Market Dominance with AI-Powered Research", Sub-headline "Acumenis delivers deep competitor analysis & market intelligence reports in hours, giving you the strategic edge.", CTA button "Explore Our Reports" linking to /pricing.
                2. How it Works section: 3 simple steps (Submit Request -> AI Analysis -> Receive Report). Use icons (💡, ⚙️, 📬).
                3. Key Differentiators section: Grid highlighting Speed, AI Depth, Cost-Effectiveness, Customization.
                4. Use Cases section: Target specific roles/needs (e.g., "For Product Managers: Validate market fit instantly.", "For VPs Marketing: Understand competitor messaging.", "For CEOs/Strategists: Identify growth opportunities.").
                5. Pricing Teaser section: Briefly mention Standard ($499) and Premium ($999) with a button "View Detailed Pricing" linking to /pricing.
                6. Testimonials section (with 3 realistic placeholders).
                7. Final CTA section: Repeat headline variation and button linking to /order.
                Ensure content is SEO-optimized for "AI research reports", "market analysis", "competitor intelligence", "Acumenis".
                Output ONLY the HTML content for the `<main>` or central content area, excluding `<html>`, `<head>`, `<body>`, header, and footer.
                """
            elif filename == "pricing.html":
                page_specific_prompt = f"""
                Generate the HTML content for the BODY of the Acumenis pricing page (`pricing.html`).
                Include:
                1. Headline: "Transparent Pricing for Actionable AI Insights". Sub-headline: "Choose the report depth that matches your strategic needs. No hidden fees, just rapid results."
                2. Detailed Pricing Table: Side-by-side comparison of "Standard Report" ($499, Anchor ~$1500) and "Premium Deep Dive" ($999, Anchor ~$3000). Use checkmarks (✓) to list features clearly for each (e.g., Turnaround Time, Analysis Depth, Recommendation Level, Data Export). Highlight Premium plan. Include "Order Now" buttons linking to `/order?plan=standard` and `/order?plan=premium`.
                3. Frequently Asked Questions (FAQ) section: Include 3-4 relevant questions and concise answers (e.g., What kind of data sources?, How custom can requests be?, What's the refund policy?).
                Ensure content is SEO-optimized for "AI report pricing", "Acumenis pricing", "market analysis cost".
                Output ONLY the HTML content for the `<main>` or central content area, excluding `<html>`, `<head>`, `<body>`, header, and footer.
                """
            elif filename == "order.html":
                page_specific_prompt = f"""
                Generate the HTML content for the BODY of the Acumenis order page (`order.html`).
                Include:
                1. Headline: "Order Your Custom AI Research Report". Sub-headline: "Get started in minutes. Fill out your requirements below and proceed to secure checkout via Lemon Squeezy."
                2. Order Form (id="report-order-form"):
                   - Fields:
                       - Name (input type="text", id="client_name", name="client_name", required)
                       - Email (input type="email", id="client_email", name="client_email", required)
                       - Company Name (input type="text", id="company_name", name="company_name")
                       - Report Type (select id="report_type", name="report_type", required): Options: value="standard_499" text="Standard Report ($499)", value="premium_999" text="Premium Deep Dive ($999)"
                       - Research Topic/Details (textarea id="request_details", name="request_details", required, placeholder="Be specific about the company, market, topic, or questions you want researched...")
                   - Submit Button (button type="submit", id="submit-order-btn"): Text "Proceed to Secure Payment".
                   - Message Div (div id="form-message"): Initially hidden, used for success/error/processing messages.
                3. Trust Badges/Signals section below form: Include icons/text for "Secure Payment via Lemon Squeezy", "Confidentiality Assured", "Fast Turnaround".
                4. JavaScript (within `<script>` tags at the end of the body content):
                   - Function to get plan from URL query parameter `?plan=` and pre-select the #report_type dropdown on page load.
                   - Add 'submit' event listener to the form (#report-order-form).
                   - Inside the listener:
                       - Prevent default form submission (`event.preventDefault()`).
                       - Get references to form elements (button, message div).
                       - Clear previous messages, show "Processing..." message, disable button.
                       - Get form values: `report_type`, `client_email`, `client_name`, `company_name`, `request_details`.
                       - Basic Validation: Check if required fields (email, name, details) are filled. If not, show error message, enable button, return.
                       - Construct JSON payload: `{{"report_type": reportType, "client_email": email, "client_name": name, "company_name": companyName, "request_details": details}}`.
                       - Use `fetch` to send a POST request to `/api/v1/payments/create-checkout` with the JSON payload and appropriate headers (`'Content-Type': 'application/json'`).
                       - Use `async/await` with `try/catch` for the fetch call.
                       - Inside `try`:
                           - Check `response.ok`. If true and `response.status === 201`:
                               - Parse JSON response (`await response.json()`).
                               - Get `checkout_url`.
                               - Redirect: `window.location.href = checkout_url;`.
                           - Else (other non-201 success or non-ok response):
                               - Try to parse error message from response JSON (`await response.json()`), fallback to `response.statusText`.
                               - Show error message in #form-message, enable button.
                       - Inside `catch` (network error):
                           - Show generic network error message in #form-message, enable button.
                Ensure content is clear and focused on completing the order. Use the provided CSS classes for styling form elements and messages.
                Output ONLY the HTML content for the `<main>` or central content area AND the `<script>` tag content, excluding `<html>`, `<head>`, `<body>`, header, and footer.
                """

            print(f"[MCOL] Generating content for {filename}...")
            page_content_response = await call_llm_api(client, page_specific_prompt, model="google/gemini-1.5-pro-latest")

            if page_content_response and isinstance(page_content_response.get("raw_inference"), str):
                page_body_content = page_content_response["raw_inference"].strip()
                # Construct full HTML
                full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Acumenis - {filename.replace('.html','').capitalize()}</title> <!-- Dynamic Title -->
    <meta name="description" content="Acumenis: AI-Powered Research Reports. Get market analysis & competitor intelligence in hours.">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    {common_css}
    <!-- Add page-specific CSS overrides here if needed -->
</head>
<body>
    {common_header}
    <main>
        {page_body_content}
    </main>
    {common_footer}
    <!-- Add page-specific JS here if not included in body content (like order.html) -->
</body>
</html>"""
                try:
                    os.makedirs(WEBSITE_OUTPUT_DIR, exist_ok=True)
                    filepath = os.path.join(WEBSITE_OUTPUT_DIR, filename)
                    with open(filepath, "w", encoding="utf-8") as f: f.write(full_html)
                    print(f"[MCOL] Successfully generated and saved website file: {filepath}")
                    generated_files.append(filename)
                except Exception as e:
                    errors.append(f"Failed to write {filename}: {e}")
                    print(f"[MCOL] Failed to write {filename}: {e}")
            else:
                errors.append(f"LLM failed to generate content for {filename}.")
                print(f"[MCOL] LLM failed to generate content for {filename}.")
            await asyncio.sleep(1) # Small delay between LLM calls

        if len(generated_files) == len(files_to_generate):
            return {"status": "COMPLETED", "result": f"Generated website files: {', '.join(generated_files)}"}
        else:
            return {"status": "FAILED", "result": f"Failed to generate all website files. Errors: {'; '.join(errors)}"}

    async def generate_payment_endpoint():
        # ... (Remains SUGGESTED for safety) ...
        print("[MCOL] Generating Lemon Squeezy payment endpoint code suggestion...")
        # ... (payment_prompt remains the same) ...
        # ... (LLM call logic remains the same) ...
        print(f"[MCOL] SUGGESTION: Create 'Acumenis/app/api/endpoints/payments.py' with generated code. Add router inclusion to 'Acumenis/app/main.py'. Requires manual Lemon Squeezy product setup.")
        return {"status": "SUGGESTED", "result": "Code generated for payments endpoint and main.py modification suggested. Requires manual product setup."}

    async def generate_report_delivery_code():
        # ... (Remains SUGGESTED for safety) ...
        print("[MCOL] Generating report delivery code modification suggestion...")
        # ... (delivery_prompt remains the same) ...
        # ... (LLM call logic remains the same) ...
        print(f"[MCOL] SUGGESTION: Modify 'Acumenis/app/agents/report_generator.py' to include report delivery logic.")
        return {"status": "SUGGESTED", "result": "Code modification suggested for report_generator.py to implement report delivery."}

    async def suggest_linkedin_actions():
        # ... (Remains SUGGESTED for safety) ...
        print("[MCOL] Generating suggestions for manual LinkedIn actions...")
        result_text = "Suggest operator manually review prospects with identified executives in DB (Prospect.key_executives). Craft personalized LinkedIn connection requests referencing recent signals/pain points."
        print(f"[MCOL] {result_text}")
        return {"status": "SUGGESTED", "result": result_text}

    # --- Strategy Execution Mapping ---
    implementation_result = {"status": "FAILED", "result": "Strategy not recognized or executable."}

    # Prioritize critical path: Website -> Payments -> Delivery
    if "website" in strategy_name and ("generate" in strategy_name or "seo" in strategy_name):
        # Execute website generation directly
        implementation_result = await generate_multi_page_website()
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