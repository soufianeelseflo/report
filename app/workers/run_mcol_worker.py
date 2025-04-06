import asyncio
import signal
from sqlalchemy.ext.asyncio import AsyncSession

# Adjust config import if needed based on final structure
from app.core.config import settings
from app.db.base import get_worker_session
from app.agents.mcol_agent import run_mcol_cycle, MCOL_ANALYSIS_INTERVAL_SECONDS

async def mcol_worker_loop(shutdown_event: asyncio.Event):
    """Main loop for the MCOL worker."""
    print("[MCOLWorker] Starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        try:
            # print(f"[MCOLWorker] Starting next analysis cycle at {asyncio.get_event_loop().time():.2f}")
            session = await get_worker_session()
            # Pass shutdown_event down if needed by run_mcol_cycle for long tasks
            await run_mcol_cycle(session, shutdown_event)

            # Wait for the configured interval before the next cycle
            wait_time = MCOL_ANALYSIS_INTERVAL_SECONDS

        except Exception as e:
            print(f"[MCOLWorker] Error in main loop: {e}")
            # Log the error properly
            if session:
                await session.rollback() # Ensure rollback on loop-level error
            # Use the standard interval even after error
            wait_time = MCOL_ANALYSIS_INTERVAL_SECONDS
            print(f"[MCOLWorker] Error occurred. Waiting {wait_time} seconds before retrying.")
        finally:
            if session:
                await session.close()

        # Wait interval or handle shutdown
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=wait_time)
            print("[MCOLWorker] Shutdown signal received during wait, exiting loop.")
            break
        except asyncio.TimeoutError:
            pass # Normal timeout, continue loop
        except Exception as e:
             print(f"[MCOLWorker] Error during wait: {e}")
             break # Exit loop on unexpected wait error

    print("[MCOLWorker] Stopped.")


async def run_mcol_worker(shutdown_event: asyncio.Event):
    """Entry point for running the MCOL worker."""
    await mcol_worker_loop(shutdown_event)


# --- Direct execution capability (optional, for testing) ---
async def main():
    print("Starting MCOL Worker directly...")
    shutdown_event = asyncio.Event()

    # Load .env file for direct execution
    from dotenv import load_dotenv
    import os
    # Adjust path relative to this file's location
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    if not os.path.exists(dotenv_path):
         # Try one level up if running from project root perhaps?
         dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load .env from: {dotenv_path}")
    # Verify DB settings loaded
    print(f"DB Server from env: {os.getenv('POSTGRES_SERVER')}")


    def signal_handler():
        print("MCOL Shutdown signal received!")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # Ensure DB is ready before starting (useful for direct run)
    # Add a small delay or a check similar to docker-entrypoint
    print("Waiting 5s for DB to potentially start...")
    await asyncio.sleep(5)

    await run_mcol_worker(shutdown_event)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("MCOL Worker stopped by user.")