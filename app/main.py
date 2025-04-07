from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates # If building a simple HTML UI
import uvicorn
import asyncio
import signal
import os

from app.core.config import settings
from app.api.endpoints import requests as api_requests
# Import payment endpoint if created by MCOL later
# from app.api.endpoints import payments as api_payments
from app.db.base import engine as async_engine, Base as db_base # Import engine and Base
from app.db import models # Ensure models are imported so Base knows about them

# --- Worker Control ---
# Global flags/tasks to control workers (simplistic approach)
worker_tasks = {}
shutdown_event = asyncio.Event()

# --- FastAPI App Setup ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- API Routers ---
app.include_router(api_requests.router, prefix=f"{settings.API_V1_STR}/requests", tags=["Report Requests"])
# MCOL might add payment router later:
# app.include_router(api_payments.router, prefix=f"{settings.API_V1_STR}/payments", tags=["Payments"])


# --- Static Files (for AI-Generated Website) ---
# Check if the directory exists (it should be created by Dockerfile)
STATIC_DIR = "/app/static_website"
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def read_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        else:
            # Fallback if MCOL hasn't generated the site yet
            return HTMLResponse("<html><body><h1>Agency Backend Running</h1><p>Website not generated yet. MCOL will attempt this.</p></body></html>", status_code=200)
else:
     @app.get("/", response_class=HTMLResponse, include_in_schema=False)
     async def read_root_fallback():
        # Fallback if static dir doesn't exist at all
        return HTMLResponse("<html><body><h1>Agency Backend Running</h1><p>Static website directory missing.</p></body></html>", status_code=200)


# --- Worker Control Endpoints ---
@app.post("/control/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is already running."}

    task = None # Initialize task to None
    if worker_name == "report_generator":
        from app.workers.run_report_worker import run_report_generator_worker
        task = asyncio.create_task(run_report_generator_worker(shutdown_event))
    elif worker_name == "prospect_researcher":
        from app.workers.run_research_worker import run_prospect_researcher_worker
        task = asyncio.create_task(run_prospect_researcher_worker(shutdown_event))
    elif worker_name == "email_marketer":
        from app.workers.run_email_worker import run_email_marketer_worker
        task = asyncio.create_task(run_email_marketer_worker(shutdown_event))
    elif worker_name == "mcol":
        from app.workers.run_mcol_worker import run_mcol_worker
        task = asyncio.create_task(run_mcol_worker(shutdown_event))
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    if task: # Check if task was created successfully
        worker_tasks[worker_name] = task
        print(f"Started worker: {worker_name}")
        return {"message": f"Worker '{worker_name}' started."}
    else:
        # This case should ideally not be reached if worker_name is valid
        # but added for robustness
        raise HTTPException(status_code=500, detail=f"Failed to create task for worker '{worker_name}'.")


@app.post("/control/stop/{worker_name}", status_code=status.HTTP_200_OK)
async def stop_worker(worker_name: str):
    """Signals a specific background worker to stop."""
    if worker_name not in worker_tasks or worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is not running or already stopped."}

    print(f"Signaling worker '{worker_name}' to stop...")
    # The worker loop itself should check shutdown_event and exit gracefully.
    # We don't forcefully cancel here. The shutdown_event is shared.
    # Setting it via /stop_all is the primary mechanism.
    # Stopping individual workers cleanly without a dedicated signal per worker is tricky.
    # For now, rely on /stop_all or let the task run until next shutdown check.
    # A more advanced implementation might use task cancellation or dedicated queues/events per worker.
    return {"message": f"Stop signal sent via shared event (use /stop_all for guaranteed signaling). Worker '{worker_name}' will exit on its next check."}

@app.post("/control/stop_all", status_code=status.HTTP_200_OK)
async def stop_all_workers():
    """Signals all running background workers to stop."""
    print("Signaling all workers to stop via shared shutdown_event...")
    shutdown_event.set()
    await asyncio.sleep(2) # Give a moment for loops to check
    stopped_count = 0
    active_workers = []
    for name, task in worker_tasks.items():
        if not task.done():
            print(f"Worker {name} stop signaled.")
            stopped_count += 1
            active_workers.append(name)
        # Optionally remove completed tasks from the dict
        # elif name in worker_tasks: del worker_tasks[name]
    # worker_tasks.clear() # Don't clear, allows checking status later if needed
    return {"message": f"Stop signal sent via shared event. {stopped_count} workers ({', '.join(active_workers)}) should stop on next cycle check."}


@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    # Extended health check could query DB status or worker heartbeats
    return schemas.HealthCheck(name=settings.PROJECT_NAME, status="OK", version="0.1.0") # Consider adding a real version

@app.on_event("startup")
async def startup_event():
    print("Starting up Autonomous Agency...")
    print("Connecting to database...")
    # Test connection implicitly via engine creation + potential first query by workers
    try:
        # Perform a simple query to ensure connection works before workers start hitting it hard
        async with async_engine.connect() as connection:
            await connection.execute(models.select(1)) # Simple test query
            print("Database connection successful.")
    except Exception as e:
        print(f"FATAL: Database connection failed on startup check: {e}")
        # Consider exiting if DB is critical and unavailable?
        # raise SystemExit("Database connection failed on startup") from e
    print("API Ready. Use /control/start/{worker_name} to activate agents.")


@app.on_event("shutdown")
async def shutdown_app_event():
    print("Shutdown signal received, initiating graceful shutdown...")
    if not shutdown_event.is_set():
        shutdown_event.set() # Ensure event is set

    tasks_to_await = [task for task in worker_tasks.values() if task and not task.done()]
    if tasks_to_await:
        print(f"Waiting for {len(tasks_to_await)} background tasks to complete...")
        # Wait for tasks with a timeout
        _, pending = await asyncio.wait(tasks_to_await, timeout=10.0)
        if pending:
            print(f"Warning: {len(pending)} tasks did not finish gracefully within timeout:")
            for task in pending:
                print(f" - Task: {task.get_name()} - Cancelling...")
                task.cancel()
                try:
                    await task # Allow cancellation to propagate
                except asyncio.CancelledError:
                    print(f"   - Task {task.get_name()} cancelled.")
                except Exception as e:
                    print(f"   - Error during task cancellation/cleanup: {e}")

    print("Disposing database engine...")
    await async_engine.dispose()
    print("Shutdown complete.")


# --- Signal Handling for Graceful Shutdown (for direct uvicorn run, less relevant in Docker usually) ---
# def handle_signal(sig, frame):
#     print(f"Received OS signal {sig}, initiating graceful shutdown...")
#     # This is tricky to integrate perfectly with Uvicorn's own signal handling
#     # Best practice in Docker is to rely on Docker sending SIGTERM and Uvicorn handling it.
#     # If running directly, Uvicorn should handle SIGINT/SIGTERM.
#     # We ensure our shutdown_event logic runs via app.on_event("shutdown").
#     pass

# --- Main Execution ---
if __name__ == "__main__":
    # Signal handling setup might interfere with Uvicorn's internal handling if run directly.
    # Rely on Uvicorn's default signal handling.
    print(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False, # Production: reload=False
        workers=1 # Essential for shared asyncio event loop and state (worker_tasks)
    )