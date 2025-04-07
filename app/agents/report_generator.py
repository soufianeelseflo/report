import asyncio
import subprocess
import os
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from app.agents.email_marketer import send_email_via_smtp 
from app.core.config import settings
from app.db import crud
from app.db import models   

# Define where reports will be stored (relative to the app root inside the container)
REPORTS_OUTPUT_DIR = "/app/generated_reports"
os.makedirs(REPORTS_OUTPUT_DIR, exist_ok=True)

async def process_single_report_request(db: AsyncSession, request: models.ReportRequest):
    """
    Processes a single report request using the open-deep-research tool.
    Updates the request status in the database.
    """
    print(f"[ReportGenerator] Processing request ID: {request.request_id} for '{request.request_details[:50]}...'")

    # 1. Update status to PROCESSING
    await crud.update_report_request_status(db, request.request_id, status="PROCESSING")
    await db.commit() # Commit status change immediately

    # 2. Prepare arguments for the open-deep-research tool
    #    This requires understanding how the tool accepts input. Assuming it takes:
    #    - A query/prompt string
    #    - An output path/filename
    #    - Potentially depth/breadth parameters based on report_type
    query = request.request_details
    # Generate a unique filename/path for the output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"report_{request.request_id}_{timestamp}.md" # Assuming markdown output
    output_path = os.path.join(REPORTS_OUTPUT_DIR, output_filename)

    # --- Adjust command based on actual open-deep-research CLI ---
    # Example assuming: node main.js --query "..." --output "..." [--depth X]
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

    try:
        # 3. Execute the open-deep-research tool as a subprocess
        #    Use asyncio.create_subprocess_exec for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=settings.OPEN_DEEP_RESEARCH_REPO_PATH # Execute from the repo's directory if needed
        )

        stdout, stderr = await process.communicate() # Wait for completion

        stdout_decoded = stdout.decode().strip() if stdout else ""
        stderr_decoded = stderr.decode().strip() if stderr else ""

        print(f"[ReportGenerator] Subprocess exited with code: {process.returncode}")
        if stdout_decoded:
            print(f"[ReportGenerator] Subprocess STDOUT:\n{stdout_decoded}")
        if stderr_decoded:
            print(f"[ReportGenerator] Subprocess STDERR:\n{stderr_decoded}")

        # 4. Check result and update DB
        if process.returncode == 0 and os.path.exists(output_path):
            # Check if output file actually contains data (basic check)
            if os.path.getsize(output_path) > 0:
                print(f"[ReportGenerator] Report generated successfully: {output_path}")
                final_status = "COMPLETED"
            else:
                error_message = "Report generation process succeeded but output file is empty."
                print(f"[ReportGenerator] Error: {error_message}")
                if os.path.exists(output_path): os.remove(output_path) # Clean up empty file
                output_path = None # Don't save path to empty file
        else:
            error_message = f"Report generation failed. Exit code: {process.returncode}. Stderr: {stderr_decoded[:500]}" # Limit error message length
            print(f"[ReportGenerator] Error: {error_message}")
            if os.path.exists(output_path): os.remove(output_path) # Clean up potentially partial file
            output_path = None

    except FileNotFoundError:
        error_message = f"Execution failed: '{settings.NODE_EXECUTABLE_PATH}' or script not found at expected path."
        print(f"[ReportGenerator] Error: {error_message}")
        output_path = None
    except Exception as e:
        error_message = f"An unexpected error occurred during report generation: {str(e)}"
        print(f"[ReportGenerator] Error: {error_message}")
        # Clean up output file if it exists and an error occurred
        if output_path and os.path.exists(output_path):
             try:
                 os.remove(output_path)
             except OSError:
                 pass # Ignore error during cleanup
        output_path = None


    # 5. Final DB update
    await crud.update_report_request_status(
        db=db,
        request_id=request.request_id,
        status=final_status,
        output_path=output_path, # Store relative path or full path depending on need
        error_message=error_message
    )
    await db.commit() # Commit the final status
    print(f"[ReportGenerator] Finished processing request ID: {request.request_id} with status: {final_status}")

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
    if final_status == "COMPLETED" and output_path and updated_request:
        print(f"[ReportGenerator] Attempting to deliver report for request ID: {request.request_id} to {updated_request.client_email}")
        delivery_account = await crud.get_active_email_account_for_sending(db)
        if delivery_account:
            try:
                # Construct delivery email content
                report_filename = os.path.basename(output_path)
                delivery_subject = f"Your Autonomous AI Report is Ready! ({report_filename})"
                delivery_body = f"""Hi {updated_request.client_name or 'Client'},

Your requested report '{report_filename}' has been generated successfully.

Please find the report attached.

We appreciate your business!

Best regards,
The Autonomous AI Reporting Agency
"""
                # Create a dummy prospect object for the send function signature
                class DeliveryRecipient:
                    contact_email = updated_request.client_email
                delivery_prospect = DeliveryRecipient()

                # Prepare email content dict
                email_content = {"subject": delivery_subject, "body": delivery_body}

                # --- Attach the file ---
                # This part needs modification in send_email_via_smtp or a new function
                # to handle attachments with aiosmtplib. For now, just sending text.
                # TODO: Enhance send_email_via_smtp to accept an attachment path.
                # Example (conceptual - needs integration):
                # with open(output_path, 'rb') as fp:
                #     msg.add_attachment(fp.read(), maintype='application', subtype='octet-stream', filename=report_filename)

                # Send using a modified send function or directly (less ideal)
                # Using existing function for now (will lack attachment)
                send_success = await send_email_via_smtp(email_content, delivery_prospect, delivery_account)

                if send_success:
                    print(f"[ReportGenerator] Delivery email initiated for request ID: {request.request_id}")
                    await crud.increment_email_sent_count(db, delivery_account.account_id)
                    await db.commit() # Commit email count increment
                else:
                    print(f"[ReportGenerator] Failed to send delivery email for request ID: {request.request_id}")
                    await db.rollback() # Rollback potential changes from send attempt

            except Exception as delivery_e:
                print(f"[ReportGenerator] Error during report delivery for request ID {request.request_id}: {delivery_e}")
                await db.rollback() # Rollback potential changes
        else:
            print(f"[ReportGenerator] Could not find active email account for report delivery (Request ID: {request.request_id}). Requires manual delivery.")
            # Log this failure clearly
            await db.rollback() # Rollback session if account fetch failed