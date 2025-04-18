# autonomous_agency/app/workers/run_email_worker.py
import asyncio
import signal
import logging
from sqlalchemy.ext.asyncio import AsyncSession

# Use absolute imports for clarity and robustness
from app.core.config import settings
from app.db.base import get_worker_session
from app.agents.email_marketer import process_email_batch

logger = logging.getLogger(__name__)

async def email_marketer_loop(shutdown_event: asyncio.Event):
    """Main loop for the email marketer worker."""
    logger.info("Email Marketer Worker starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        cycle_start_time = asyncio.get_event_loop().time()
        try:
            logger.debug("Starting email batch processing cycle.")
            session = await get_worker_session()
            # Pass shutdown_event down to the batch processor for early exit within the batch
            await process_email_batch(session, shutdown_event)
            # process_email_batch handles its own commits/rollbacks per prospect

            cycle_duration = asyncio.get_event_loop().time() - cycle_start_time
            wait_time = max(0, settings.EMAIL_MARKETER_INTERVAL_SECONDS - cycle_duration)
            logger.debug(f"Email batch cycle finished in {cycle_duration:.2f}s. Waiting {wait_time:.2f}s.")

        except Exception as e:
            logger.error(f"Error in email_marketer_loop: {e}", exc_info=True)
            if session:
                try:
                    await session.rollback() # Ensure rollback on loop-level error
                except Exception as rollback_e:
                    logger.error(f"Error during session rollback: {rollback_e}", exc_info=True)
            # Use a longer, fixed wait time after an error to prevent rapid failure loops
            wait_time = settings.EMAIL_MARKETER_INTERVAL_SECONDS * 3
            logger.info(f"Error occurred. Waiting {wait_time} seconds before next cycle.")
        finally:
            if session:
                try:
                    await session.close()
                except Exception as close_e:
                    logger.error(f"Error closing session: {close_e}", exc_info=True)

        # Wait interval or handle shutdown gracefully
        if not shutdown_event.is_set():
            try:
                # Wait for the interval, but break early if shutdown is signaled
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                logger.info("Shutdown signal received during wait, exiting loop.")
                break # Exit loop if shutdown_event is set during wait
            except asyncio.TimeoutError:
                continue # Normal timeout, continue to next loop iteration
            except Exception as e:
                 logger.error(f"Error during worker wait: {e}", exc_info=True)
                 break # Exit loop on unexpected wait error

    logger.info("Email Marketer Worker stopped.")


async def run_email_marketer_worker(shutdown_event: asyncio.Event):
    """Entry point for running the worker."""
    # Potential setup specific to this worker could go here
    await email_marketer_loop(shutdown_event)


# --- Direct execution capability (for testing) ---
async def main():
    print("Starting Email Marketer Worker directly (for testing)...")
    local_shutdown_event = asyncio.Event()

    # Load .env file for direct execution
    print("Attempting to load .env file...")
    from dotenv import load_dotenv
    import os
    # Adjust path calculation for robustness
    script_dir = os.path.dirname(__file__)
    dotenv_path = os.path.abspath(os.path.join(script_dir, '..', '..', '.env'))
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        print(f"Loaded .env from: {dotenv_path}")
    else:
        print(f"Warning: .env file not found at {dotenv_path}")

    # Setup basic logging for direct run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s')

    def signal_handler():
        print("Shutdown signal received! Signaling worker to stop...")
        local_shutdown_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await run_email_marketer_worker(local_shutdown_event)
    except Exception as e:
        print(f"Worker exited with error: {e}")
    finally:
        print("Email Marketer Worker main task finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Email Marketer Worker stopped by user (KeyboardInterrupt).")