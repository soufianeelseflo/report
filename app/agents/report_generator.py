import asyncio
import subprocess
import os
import json
import mimetypes
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from typing import List, Optional

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession
import httpx # Needed for LLM call

# Assuming agent_utils defines get_httpx_client and call_llm_api
from autonomous_agency.app.agents.agent_utils import get_httpx_client, call_llm_api
from autonomous_agency.app.core.config import settings
from autonomous_agency.app.db import crud, models
from autonomous_agency.app.core.security import decrypt_data

# Define where reports will be stored (relative to the app root inside the container)
REPORTS_OUTPUT_DIR = "/app/generated_reports"
os.makedirs(REPORTS_OUTPUT_DIR, exist_ok=True)

async def _send_delivery_email(db: AsyncSession, request: models.ReportRequest, report_path: str):
    """Internal function to handle sending the delivery email with attachment."""
    print(f"[ReportDelivery] Attempting delivery for request ID: {request.request_id} to {request.client_email}")
    delivery_account = await crud.get_active_email_account_for_sending(db)
    if not delivery_account:
        print(f"[ReportDelivery] No active sending account found for request {request.request_id}.")
        return False

    decrypted_password = decrypt_data(delivery_account.smtp_password_encrypted)
    if not decrypted_password:
        print(f"[ReportDelivery] CRITICAL: Failed to decrypt password for account {delivery_account.email_address}. Cannot send report {request.request_id}.")
        return False

    report_filename = os.path.basename(report_path)
    subject = f"Your AI Research Report is Ready! ({report_filename})"
    body = f"""Hi {request.client_name or 'Client'},

Your requested AI research report '{report_filename}' has been generated successfully.

Please find the report attached.

If you have any questions or need further analysis, please don't hesitate to reach out.

Best regards,
The Autonomous AI Reporting Agency
{settings.AGENCY_BASE_URL}
"""

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = delivery_account.email_address
    msg['To'] = request.client_email
    msg['Date'] = formatdate(localtime=True)
    msg['Message-ID'] = make_msgid(domain=delivery_account.email_address.split('@')[-1])
    msg.set_content(body)

    # Attach the file
    if os.path.exists(report_path):
        ctype, encoding = mimetypes.guess_type(report_path)
        if ctype is None or encoding is not None: ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        try:
            with open(report_path, 'rb') as fp:
                msg.add_attachment(fp.read(), maintype=maintype, subtype=subtype, filename=report_filename)
            print(f"[ReportDelivery] Attached report file: {report_filename}")
        except Exception as attach_e:
            print(f"[ReportDelivery] Failed to attach report file {report_path}: {attach_e}")
            msg.set_content(body + f"\n\n[Attachment Error: Could not attach report file '{report_filename}'. Please contact support.]")
    else:
        print(f"[ReportDelivery] Report file not found at path: {report_path}. Sending email without attachment.")
        msg.set_content(body + f"\n\n[Delivery Error: Report file '{report_filename}' not found. Please contact support.]")

    # Send using aiosmtplib
    try:
        async with aiosmtplib.SMTP(hostname=delivery_account.smtp_host, port=delivery_account.smtp_port, use_tls=True, timeout=30) as smtp:
            await smtp.login(delivery_account.smtp_user, decrypted_password)
            await smtp.send_message(msg)
        print(f"[ReportDelivery] Delivery email sent successfully for request ID: {request.request_id}")
        await crud.increment_email_sent_count(db, delivery_account.account_id)
        await db.commit()
        return True
    except Exception as smtp_e:
        print(f"[ReportDelivery] SMTP Error sending report for request ID {request.request_id} via {delivery_account.email_address}: {smtp_e}")
        await db.rollback()
        return False

async def get_opendeepresearch_args(client: httpx.AsyncClient, query: str, report_type: str, output_path: str) -> List[str]:
    """
    Attempts to determine the correct CLI arguments for open-deep-research using LLM analysis.
    Falls back to default if LLM fails or provides invalid args.
    """
    default_args = [
        settings.NODE_EXECUTABLE_PATH,
        os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT),
        "--query", query,
        "--output", output_path,
    ]
    # Add default depth based on report type
    if report_type == 'premium_999':
        default_args.extend(["--depth", "5"]) # Example depth for premium

    prompt = f"""
    Analyze the requirements for running the 'open-deep-research' Node.js tool (likely located at {settings.OPEN_DEEP_RESEARCH_REPO_PATH}/{settings.OPEN_DEEP_RESEARCH_ENTRY_POINT}).
    The goal is to generate a research report based on a user query.
    User Query: "{query}"
    Requested Report Type: "{report_type}" (standard_499 or premium_999)
    Desired Output File Path: "{output_path}"

    Based on common Node.js CLI patterns and the likely function of a 'deep research' tool:
    1. Infer the most probable command-line arguments needed to run this tool.
    2. Assume standard arguments like `--query` for the user query and `--output` for the file path.
    3. Consider if the report type ('standard_499' vs 'premium_999') might map to arguments like `--depth`, `--breadth`, `--model`, or `--sources`. A premium report likely requires greater depth or breadth.
    4. Construct the full command as a JSON list of strings, including the node executable path (`{settings.NODE_EXECUTABLE_PATH}`) and the script path.
    Example Output: ["{settings.NODE_EXECUTABLE_PATH}", "{os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT)}", "--query", "{query}", "--output", "{output_path}", "--depth", "5"]

    Respond ONLY with the JSON list of command arguments. If unsure, return the default command: {json.dumps(default_args)}
    """
    print("[ReportGenerator] Attempting to determine open-deep-research args via LLM...")
    llm_response = await call_llm_api(client, prompt)

    if llm_response and isinstance(llm_response, list) and len(llm_response) > 3:
        # Basic validation: check if it looks like a command list
        if llm_response[0] == settings.NODE_EXECUTABLE_PATH and output_path in llm_response:
            print(f"[ReportGenerator] Using LLM-generated args: {' '.join(llm_response)}")
            return llm_response
        else:
             print("[ReportGenerator] LLM response doesn't look like valid args, using default.")
             return default_args
    elif llm_response and isinstance(llm_response.get("raw_inference"), str):
         # Try parsing raw inference if it's a JSON list string
         try:
             parsed_args = json.loads(llm_response["raw_inference"])
             if isinstance(parsed_args, list) and len(parsed_args) > 3 and parsed_args[0] == settings.NODE_EXECUTABLE_PATH:
                  print(f"[ReportGenerator] Using LLM-generated args (parsed from raw): {' '.join(parsed_args)}")
                  return parsed_args
             else:
                  print("[ReportGenerator] LLM raw inference is not a valid args list, using default.")
                  return default_args
         except json.JSONDecodeError:
              print("[ReportGenerator] LLM raw inference is not JSON, using default.")
              return default_args
    else:
        print("[ReportGenerator] LLM failed to provide args, using default.")
        return default_args


async def process_single_report_request(db: AsyncSession, request: models.ReportRequest):
    """
    Processes a single report request using the open-deep-research tool,
    updates status, and attempts email delivery on completion.
    """
    print(f"[ReportGenerator] Processing request ID: {request.request_id} for '{request.request_details[:50]}...'")
    client: Optional[httpx.AsyncClient] = None # Define client for LLM call

    try:
        # 1. Update status to PROCESSING
        await crud.update_report_request_status(db, request.request_id, status="PROCESSING")
        await db.commit()

        # 2. Prepare arguments dynamically
        query = request.request_details
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query_part = "".join(c if c.isalnum() else "_" for c in query[:30])
        output_filename = f"report_{request.request_id}_{safe_query_part}_{timestamp}.md"
        output_path = os.path.join(REPORTS_OUTPUT_DIR, output_filename)

        client = await get_httpx_client()
        cmd = await get_opendeepresearch_args(client, query, request.report_type, output_path)

        print(f"[ReportGenerator] Executing command: {' '.join(cmd)}")
        error_message = None
        final_status = "FAILED"
        process_success = False

        # 3. Execute the open-deep-research tool
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.OPEN_DEEP_RESEARCH_REPO_PATH
        )
        stdout, stderr = await process.communicate()
        stdout_decoded = stdout.decode(errors='ignore').strip() if stdout else ""
        stderr_decoded = stderr.decode(errors='ignore').strip() if stderr else ""

        print(f"[ReportGenerator] Subprocess exited with code: {process.returncode}")
        if stdout_decoded: print(f"[ReportGenerator] Subprocess STDOUT:\n{stdout_decoded[:1000]}...")
        if stderr_decoded: print(f"[ReportGenerator] Subprocess STDERR:\n{stderr_decoded[:1000]}...")

        # 4. Check result
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10:
            print(f"[ReportGenerator] Report generated successfully: {output_path}")
            final_status = "COMPLETED"
            process_success = True
        elif process.returncode == 0:
             error_message = "Report generation process succeeded but output file is empty or too small."
             print(f"[ReportGenerator] Error: {error_message}")
             if os.path.exists(output_path): os.remove(output_path)
             output_path = None
        else:
            error_message = f"Report generation failed. Exit code: {process.returncode}. Stderr: {stderr_decoded[:500]}"
            print(f"[ReportGenerator] Error: {error_message}")
            if os.path.exists(output_path): os.remove(output_path)
            output_path = None

    except FileNotFoundError:
        error_message = f"Execution failed: '{cmd[0]}' or script '{cmd[1]}' not found."
        print(f"[ReportGenerator] Error: {error_message}")
        output_path = None
    except Exception as e:
        error_message = f"An unexpected error occurred during report generation: {str(e)}"
        print(f"[ReportGenerator] Error: {error_message}")
        import traceback
        traceback.print_exc()
        if output_path and os.path.exists(output_path):
             try: os.remove(output_path)
             except OSError: pass
        output_path = None
    finally:
         if client: await client.aclose() # Close httpx client

    # 5. Final DB update for report status
    updated_request = await crud.update_report_request_status(
        db=db,
        request_id=request.request_id,
        status=final_status,
        output_path=output_path,
        error_message=error_message
    )
    await db.commit()
    print(f"[ReportGenerator] Finished processing request ID: {request.request_id} with status: {final_status}")

    # --- 6. Deliver Report on Completion ---
    if process_success and final_status == "COMPLETED" and output_path and updated_request:
        # Run delivery in background task
        asyncio.create_task(_send_delivery_email(db, updated_request, output_path))
    elif final_status == "FAILED":
         print(f"[ReportGenerator] Report generation failed for request {request.request_id}. No delivery attempted.")