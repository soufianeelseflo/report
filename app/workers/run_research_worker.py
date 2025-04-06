import asyncio
import signal
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import get_worker_session
# Corrected import path if run_prospecting_cycle is now directly in prospect_researcher.py
from app.agents.prospect_researcher import run_prospecting_cycle

async def prospect_researcher_loop(shutdown_event: asyncio.Event):
    """Main loop for the prospect researcher worker."""
    print("[ResearchWorker] Starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        try:
            print(f"[ResearchWorker] Starting next prospecting cycle at {asyncio.get_event_loop().time():.2f}")
            session = await get_worker_session()
            # Pass shutdown_event down to the cycle runner
            await run_prospecting_cycle(session, shutdown_event)

            # Wait for the configured interval before the next cycle
            wait_time = settings.PROSPECT_RESEARCHER_INTERVAL_SECONDS
            print(f"[ResearchWorker] Cycle finished. Waiting {wait_time} seconds for next cycle.")

        except Exception as e:
            print(f"[ResearchWorker] Error in main loop: {e}")
            # Log the error properly
            if session:
                await session.rollback() # Ensure rollback on loop-level error
            # Use the standard interval even after error to avoid hammering
            wait_time = settings.PROSPECT_RESEARCHER_INTERVAL_SECONDS
            print(f"[ResearchWorker] Error occurred. Waiting {wait_time} seconds before retrying.")
        finally:
            if session:
                await session.close()

        # Wait interval or handle shutdown
        try:
            # Wait for the interval, but break early if shutdown is signaled
            await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
            # If wait_for completes without timeout, shutdown was signaled
            print("[ResearchWorker] Shutdown signal received during wait, exiting loop.")
            break
        except asyncio.TimeoutError:
            # This is the normal case, timeout occurred, continue loop
            pass
        except Exception as e:
            # Handle potential errors in wait_for itself
             print(f"[ResearchWorker] Error during wait: {e}")
             break # Exit loop on unexpected wait error

    print("[ResearchWorker] Stopped.")


async def run_prospect_researcher_worker(shutdown_event: asyncio.Event):
    """Entry point for running the worker, handles graceful shutdown."""
    await prospect_researcher_loop(shutdown_event)


# --- Direct execution capability (optional, for testing) ---
async def main():
    print("Starting Prospect Researcher Worker directly...")
    shutdown_event = asyncio.Event()

    # Load .env file for direct execution if needed
    from dotenv import load_dotenv
    import os
    # Assuming .env is in the parent directory relative to workers/
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)


    def signal_handler():
        print("Shutdown signal received!")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    await run_prospect_researcher_worker(shutdown_event)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Prospect Researcher Worker stopped by user.")