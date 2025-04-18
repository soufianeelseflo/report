# autonomous_agency/app/workers/run_report_worker.py
import asyncio
import signal
import logging
from sqlalchemy.ext.asyncio import AsyncSession

# Use absolute imports
from app.core.config import settings
from app.db.base import get_worker_session
from app.db import crud
from app.agents.report_generator import process_single_report_request

logger = logging.getLogger(__name__)

async def report_generator_loop(shutdown_event: asyncio.Event):
    """Main loop for the report generator worker."""
    logger.info("Report Generator Worker starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        processed_request_in_cycle = False
        try:
            session = await get_worker_session()
            # Atomically fetch and lock the next pending request
            pending_request = await crud.get_and_lock_pending_report_request(session)

            if pending_request:
                processed_request_in_cycle = True
                logger.info(f"Processing ReportRequest ID: {pending_request.request_id}")
                # Process the request - this function handles its own status updates within the transaction
                await process_single_report_request(session, pending_request)
                await session.commit() # Commit the transaction for this request
                logger.info(f"Successfully processed and committed ReportRequest ID: {pending_request.request_id}")
                # Optional: Short delay after processing one request before checking for the next
                await asyncio.sleep(0.5)
            else:
                # No pending requests found, wait longer before checking again
                await session.rollback() # Rollback the transaction as no request was locked/processed
                wait_time = settings.REPORT_GENERATOR_INTERVAL_SECONDS
                logger.debug(f"No pending report requests found. Waiting {wait_time}s.")
                await asyncio.sleep(wait_time) # Use standard sleep here

        except Exception as e:
            logger.error(f"Error in report_generator_loop: {e}", exc_info=True)
            if session:
                try:
                    await session.rollback()
                except Exception as rollback_e:
                    logger.error(f"Error during session rollback: {rollback_e}", exc_info=True)
            # Avoid busy-looping on persistent errors - wait longer
            wait_time_on_error = settings.REPORT_GENERATOR_INTERVAL_SECONDS * 4
            logger.info(f"Error occurred. Waiting {wait_time_on_error} seconds before next cycle.")
            await asyncio.sleep(wait_time_on_error)
        finally:
            if session:
                try:
                    await session.close()
                except Exception as close_e:
                    logger.error(f"Error closing session: {close_e}", exc_info=True)

        # Check shutdown event again, especially if a request was processed quickly
        if processed_request_in_cycle and shutdown_event.is_set():
            logger.info("Shutdown signal received after processing request, exiting loop.")
            break
        elif not processed_request_in_cycle and shutdown_event.is_set():
             # This case is handled by the wait_for logic below if sleep was interrupted
             pass

        # If no request was processed, the main wait logic applies
        if not processed_request_in_cycle and not shutdown_event.is_set():
             try:
                 await asyncio.wait_for(shutdown_event.wait(), timeout=settings.REPORT_GENERATOR_INTERVAL_SECONDS)
                 logger.info("Shutdown signal received during wait, exiting loop.")
                 break
             except asyncio.TimeoutError:
                 continue # Normal timeout
             except Exception as e:
                  logger.error(f"Error during worker wait: {e}", exc_info=True)
                  break

    logger.info("Report Generator Worker stopped.")


async def run_report_generator_worker(shutdown_event: asyncio.Event):
    """Entry point for running the worker."""
    await report_generator_loop(shutdown_event)


# --- Direct execution capability (for testing) ---
async def main():
    print("Starting Report Generator Worker directly (for testing)...")
    local_shutdown_event = asyncio.Event()

    print("Attempting to load .env file...")
    from dotenv import load_dotenv
    import os
    script_dir = os.path.dirname(__file__)
    dotenv_path = os.path.abspath(os.path.join(script_dir, '..', '..', '.env'))
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')

    def signal_handler():
        print("Shutdown signal received! Signaling worker to stop...")
        local_shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await run_report_generator_worker(local_shutdown_event)
    except Exception as e:
        print(f"Worker exited with error: {e}")
    finally:
        print("Report Generator Worker main task finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Report Generator Worker stopped by user (KeyboardInterrupt).")