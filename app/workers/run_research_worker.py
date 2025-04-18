# autonomous_agency/app/workers/run_research_worker.py
import asyncio
import signal
import logging
from sqlalchemy.ext.asyncio import AsyncSession

# Use absolute imports
from app.core.config import settings
from app.db.base import get_worker_session
from app.agents.prospect_researcher import run_prospecting_cycle

logger = logging.getLogger(__name__)

async def prospect_researcher_loop(shutdown_event: asyncio.Event):
    """Main loop for the prospect researcher worker."""
    logger.info("Prospect Researcher Worker starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        cycle_start_time = asyncio.get_event_loop().time()
        try:
            logger.info("Starting new prospecting cycle.")
            session = await get_worker_session()
            # Pass shutdown_event down to the cycle runner for early exit
            await run_prospecting_cycle(session, shutdown_event)
            # run_prospecting_cycle handles its own commits/rollbacks per prospect

            cycle_duration = asyncio.get_event_loop().time() - cycle_start_time
            wait_time = max(0, settings.PROSPECT_RESEARCHER_INTERVAL_SECONDS - cycle_duration)
            logger.info(f"Prospecting cycle finished in {cycle_duration:.2f}s. Waiting {wait_time:.2f}s for next cycle.")

        except Exception as e:
            logger.error(f"Error in prospect_researcher_loop: {e}", exc_info=True)
            if session:
                try:
                    await session.rollback()
                except Exception as rollback_e:
                    logger.error(f"Error during session rollback: {rollback_e}", exc_info=True)
            # Use a shorter fixed wait time after an error to potentially recover faster,
            # but not too short to cause hammering if the error is persistent.
            wait_time = settings.PROSPECT_RESEARCHER_INTERVAL_SECONDS / 2
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
                await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
                logger.info("Shutdown signal received during wait, exiting loop.")
                break
            except asyncio.TimeoutError:
                continue # Normal timeout
            except Exception as e:
                 logger.error(f"Error during worker wait: {e}", exc_info=True)
                 break

    logger.info("Prospect Researcher Worker stopped.")


async def run_prospect_researcher_worker(shutdown_event: asyncio.Event):
    """Entry point for running the worker."""
    await prospect_researcher_loop(shutdown_event)


# --- Direct execution capability (for testing) ---
async def main():
    print("Starting Prospect Researcher Worker directly (for testing)...")
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
        await run_prospect_researcher_worker(local_shutdown_event)
    except Exception as e:
        print(f"Worker exited with error: {e}")
    finally:
        print("Prospect Researcher Worker main task finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Prospect Researcher Worker stopped by user (KeyboardInterrupt).")