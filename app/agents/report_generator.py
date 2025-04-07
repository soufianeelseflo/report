import asyncio
import subprocess
import os
import json
import mimetypes
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

import aiosmtplib
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db import crud, models
from app.core.security import decrypt_data # Import decrypt

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
        # Deactivate account? MCOL should handle this based on logs/KPIs
        # await crud.set_email_account_inactive(db, delivery_account.account_id, "Decryption failed during report delivery")
        # await db.commit()
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
""" # Add base URL for context

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
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream' # Default if guess fails
        maintype, subtype = ctype.split('/', 1)
        try:
            with open(report_path, 'rb') as fp:
                msg.add_attachment(fp.read(),
                                   maintype=maintype,
                                   subtype=subtype,
                                   filename=report_filename)
            print(f"[ReportDelivery] Attached report file: {report_filename}")
        except Exception as attach_e:
            print(f"[ReportDelivery] Failed to attach report file {report_path}: {attach_e}")
            # Proceed without attachment or fail delivery? Let's proceed without.
            msg.set_content(body + f"\n\n[Attachment Error: Could not attach report file '{report_filename}'. Please contact support.]") # Modify body
    else:
        print(f"[ReportDelivery] Report file not found at path: {report_path}. Sending email without attachment.")
        msg.set_content(body + f"\n\n[Delivery Error: Report file '{report_filename}' not found. Please contact support.]") # Modify body

    # Send using aiosmtplib
    try:
        async with aiosmtplib.SMTP(
            hostname=delivery_account.smtp_host,
            port=delivery_account.smtp_port,
            use_tls=True,
            timeout=30, # Use config setting?
        ) as smtp:
            await smtp.login(delivery_account.smtp_user, decrypted_password)
            await smtp.send_message(msg)
        print(f"[ReportDelivery] Delivery email sent successfully for request ID: {request.request_id}")
        # Increment count after successful send
        await crud.increment_email_sent_count(db, delivery_account.account_id)
        await db.commit() # Commit count increment separately
        return True
    except Exception as smtp_e:
        print(f"[ReportDelivery] SMTP Error sending report for request ID {request.request_id} via {delivery_account.email_address}: {smtp_e}")
        # Don't deactivate account here, let MCOL handle persistent failures
        await db.rollback() # Rollback potential changes from failed send attempt
        return False


async def process_single_report_request(db: AsyncSession, request: models.ReportRequest):
    """
    Processes a single report request using the open-deep-research tool,
    updates status, and attempts email delivery on completion.
    """
    print(f"[ReportGenerator] Processing request ID: {request.request_id} for '{request.request_details[:50]}...'")

    # 1. Update status to PROCESSING
    await crud.update_report_request_status(db, request.request_id, status="PROCESSING")
    await db.commit() # Commit status change immediately

    # 2. Prepare arguments for the open-deep-research tool
    query = request.request_details
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure filename is safe for filesystems
    safe_query_part = "".join(c if c.isalnum() else "_" for c in query[:30])
    output_filename = f"report_{request.request_id}_{safe_query_part}_{timestamp}.md"
    output_path = os.path.join(REPORTS_OUTPUT_DIR, output_filename)

    # --- Adjust command based on actual open-deep-research CLI ---
    # Needs verification by browsing repo or MCOL adapting based on errors
    cmd = [
        settings.NODE_EXECUTABLE_PATH,
        os.path.join(settings.OPEN_DEEP_RESEARCH_REPO_PATH, settings.OPEN_DEEP_RESEARCH_ENTRY_POINT),
        "--query", query,
        "--output", output_path,
        # Add other parameters based on report_type or defaults
        # e.g., if request.report_type == 'premium_999': cmd.extend(["--depth", "5"])
    ]
    # --- End command adjustment ---

    print(f"[ReportGenerator] Executing command: {' '.join(cmd)}")
    error_message = None
    final_status = "FAILED" # Assume failure unless successful
    process_success = False

    try:
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
        if stdout_decoded: print(f"[ReportGenerator] Subprocess STDOUT:\n{stdout_decoded[:1000]}...") # Limit log size
        if stderr_decoded: print(f"[ReportGenerator] Subprocess STDERR:\n{stderr_decoded[:1000]}...") # Limit log size

        # 4. Check result
        if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 10: # Check size > 10 bytes
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
        error_message = f"Execution failed: '{settings.NODE_EXECUTABLE_PATH}' or script not found at expected path."
        print(f"[ReportGenerator] Error: {error_message}")
        output_path = None
    except Exception as e:
        error_message = f"An unexpected error occurred during report generation: {str(e)}"
        print(f"[ReportGenerator] Error: {error_message}")
        import traceback
        traceback.print_exc() # Log full traceback for unexpected errors
        if output_path and os.path.exists(output_path):
             try: os.remove(output_path)
             except OSError: pass
        output_path = None

    # 5. Final DB update for report status
    updated_request = await crud.update_report_request_status(
        db=db,
        request_id=request.request_id,
        status=final_status,
        output_path=output_path,
        error_message=error_message
    )
    await db.commit() # Commit the final status
    print(f"[ReportGenerator] Finished processing request ID: {request.request_id} with status: {final_status}")

    # --- 6. Deliver Report on Completion ---
    if process_success and final_status == "COMPLETED" and output_path and updated_request:
        # Run delivery in background task to avoid blocking worker loop if SMTP is slow
        asyncio.create_task(_send_delivery_email(db, updated_request, output_path))
    elif final_status == "FAILED":
         # Optionally notify client of failure? Or just log it? MCOL might decide.
         print(f"[ReportGenerator] Report generation failed for request {request.request_id}. No delivery attempted.")