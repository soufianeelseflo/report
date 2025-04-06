import asyncio
import signal
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.base import get_worker_session
from app.db import crud
from app.agents.report_generator import process_single_report_request

async def report_generator_loop(shutdown_event: asyncio.Event):
    """Main loop for the report generator worker."""
    print("[ReportWorker] Starting...")
    while not shutdown_event.is_set():
        session: AsyncSession = None
        try:
            session = await get_worker_session()
            # Check for pending requests using SELECT FOR UPDATE SKIP LOCKED
            pending_request = await crud.get_pending_report_request(session)

            if pending_request:
                # Process the request (handles its own commits for status updates)
                await process_single_report_request(session, pending_request)
                # No need to commit here as process_single_report_request handles it
                # Small delay after processing before checking again
                await asyncio.sleep(1)
            else:
                # No pending requests found, wait longer before checking again
                await asyncio.sleep(settings.REPORT_GENERATOR_INTERVAL_SECONDS)

        except Exception as e:
            print(f"[ReportWorker] Error in main loop: {e}")
            # Log the error properly
            if session:
                await session.rollback() # Rollback any potential transaction issues
            # Avoid busy-looping on persistent errors
            await asyncio.sleep(settings.REPORT_GENERATOR_INTERVAL_SECONDS * 2)
        finally:
            if session:
                await session.close()

        # Check shutdown event again before next iteration
        if shutdown_event.is_set():
            print("[ReportWorker] Shutdown signal received, exiting loop.")
            break

    print("[ReportWorker] Stopped.")


async def run_report_generator_worker(shutdown_event: asyncio.Event):
    """Entry point for running the worker, handles graceful shutdown."""
    await report_generator_loop(shutdown_event)


# --- Direct execution capability (optional, for testing) ---
async def main():
    print("Starting Report Generator Worker directly...")
    shutdown_event = asyncio.Event()

    def signal_handler():
        print("Shutdown signal received!")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    await run_report_generator_worker(shutdown_event)

if __name__ == "__main__":
    # Note: Running workers directly like this is mainly for testing.
    # In production, they should be managed by the main app's control API or docker-compose/supervisor.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Report Generator Worker stopped by user.")