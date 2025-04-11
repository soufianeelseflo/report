# autonomous_agency/app/workers/run_key_acquirer_worker.py
import asyncio
import signal
import traceback

# Adjust imports based on final structure
from Acumenis.app.core.config import settings
from Acumenis.app.agents.key_acquirer import run_acquisition_process

async def key_acquirer_worker_loop(shutdown_event: asyncio.Event):
    """Main loop for the Key Acquirer worker."""
    print("[KeyAcquirerWorker] Starting...")

    # Determine if the worker should run based on settings
    run_on_startup = settings.KEY_ACQUIRER_RUN_ON_STARTUP
    target_key_count = settings.KEY_ACQUIRER_TARGET_COUNT

    if not run_on_startup:
        print("[KeyAcquirerWorker] KEY_ACQUIRER_RUN_ON_STARTUP is false. Worker will not run.")
        return # Exit if not configured to run

    if not target_key_count or target_key_count <= 0:
        print(f"[KeyAcquirerWorker] Invalid KEY_ACQUIRER_TARGET_COUNT ({target_key_count}). Worker will not run.")
        return

    print(f"[KeyAcquirerWorker] Configured Target Key Count: {target_key_count}")
    print(f"[KeyAcquirerWorker] Configured Max Concurrency: {settings.KEY_ACQUIRER_CONCURRENCY}")
    print(f"[KeyAcquirerWorker] Configured Proxy List: {'Yes' if settings.PROXY_LIST else 'No'}")

    try:
        # Run the main acquisition process from the agent
        # This function contains its own loop to reach the target
        await run_acquisition_process(target_keys=target_key_count)

        # Check if shutdown was signaled during acquisition
        if shutdown_event.is_set():
            print("[KeyAcquirerWorker] Shutdown signal received during acquisition process.")

    except Exception as e:
        print(f"[KeyAcquirerWorker] Error during key acquisition process: {e}")
        traceback.print_exc() # Log the full traceback for debugging
    finally:
        print("[KeyAcquirerWorker] Finished.")
        # This worker typically runs once to reach the target, then stops.
        # It doesn't loop indefinitely unless designed differently.


async def run_key_acquirer_worker(shutdown_event: asyncio.Event):
    """Entry point for running the Key Acquirer worker."""
    # This worker might only need to run once, unlike others that loop.
    # If it needs to run periodically, the loop logic would be different.
    await key_acquirer_worker_loop(shutdown_event)


# --- Direct execution capability (optional, for testing) ---
async def main():
    print("Starting Key Acquirer Worker directly...")
    shutdown_event = asyncio.Event()

    # Load .env file for direct execution
    from dotenv import load_dotenv
    import os
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
    load_dotenv(dotenv_path=dotenv_path)
    print(f"Attempting to load .env from: {dotenv_path}")
    # Re-initialize settings after loading .env if necessary
    settings.reload() # Assuming settings object has a reload method or re-instantiate
    print(f"DB Server from env: {settings.POSTGRES_SERVER}")
    print(f"Run on startup: {settings.KEY_ACQUIRER_RUN_ON_STARTUP}")
    print(f"Target keys: {settings.KEY_ACQUIRER_TARGET_COUNT}")


    def signal_handler():
        print("Key Acquirer Shutdown signal received!")
        shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # Ensure DB is ready before starting
    print("Waiting 5s for DB to potentially start...")
    await asyncio.sleep(5)

    await run_key_acquirer_worker(shutdown_event)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Key Acquirer Worker stopped by user.")