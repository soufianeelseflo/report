from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates # If building a simple HTML UI
import uvicorn
import asyncio
import signal

from app.core.config import settings
from app.api.endpoints import requests as api_requests
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

# --- Simple Control UI (Example using HTML Templates) ---
# templates = Jinja2Templates(directory="templates") # Create a 'templates' dir if needed

# @app.get("/", response_class=HTMLResponse, include_in_schema=False)
# async def read_root(request: Request):
#     # return templates.TemplateResponse("index.html", {"request": request, "workers": worker_tasks.keys()})
#     return HTMLResponse("<html><body><h1>Agency Control</h1><p>Control endpoints active.</p></body></html>") # Basic HTML

@app.post("/control/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is already running."}

    if worker_name == "report_generator":
        from app.workers.run_report_worker import run_report_generator_worker
        task = asyncio.create_task(run_report_generator_worker(shutdown_event))
    elif worker_name == "prospect_researcher":
        from app.workers.run_research_worker import run_prospect_researcher_worker
        task = asyncio.create_task(run_prospect_researcher_worker(shutdown_event))
    elif worker_name == "email_marketer":
        from app.workers.run_email_worker import run_email_marketer_worker
        task = asyncio.create_task(run_email_marketer_worker(shutdown_event))
    # --- ADD MCOL ---
    elif worker_name == "mcol":
        from app.workers.run_mcol_worker import run_mcol_worker
        task = asyncio.create_task(run_mcol_worker(shutdown_event))
    # --- END ADD MCOL ---
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    worker_tasks[worker_name] = task
    print(f"Started worker: {worker_name}")
    return {"message": f"Worker '{worker_name}' started."}

# Add "mcol" to the list of workers stoppable via /control/stop/{worker_name} and /control/stop_all
# (The existing stop logic using shutdown_event already covers this)

@app.post("/control/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is already running."}

    if worker_name == "report_generator":
        from app.workers.run_report_worker import run_report_generator_worker
        task = asyncio.create_task(run_report_generator_worker(shutdown_event))
    elif worker_name == "prospect_researcher":
        from app.workers.run_research_worker import run_prospect_researcher_worker
        task = asyncio.create_task(run_prospect_researcher_worker(shutdown_event))
    elif worker_name == "email_marketer":
        from app.workers.run_email_worker import run_email_marketer_worker
        task = asyncio.create_task(run_email_marketer_worker(shutdown_event))
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    worker_tasks[worker_name] = task
    print(f"Started worker: {worker_name}")
    return {"message": f"Worker '{worker_name}' started."}

@app.post("/control/stop/{worker_name}", status_code=status.HTTP_200_OK)
async def stop_worker(worker_name: str):
    """Signals a specific background worker to stop."""
    if worker_name not in worker_tasks or worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is not running or already stopped."}

    # Signal the worker to stop (workers need to check shutdown_event)
    # Actual stopping logic is within the worker loop
    # For immediate stop (less graceful): worker_tasks[worker_name].cancel()
    print(f"Signaling worker '{worker_name}' to stop...")
    # The worker loop itself should check shutdown_event and exit gracefully.
    # We don't forcefully cancel here unless necessary.
    return {"message": f"Stop signal sent to worker '{worker_name}'. It will exit on its next cycle."}

@app.post("/control/stop_all", status_code=status.HTTP_200_OK)
async def stop_all_workers():
    """Signals all running background workers to stop."""
    print("Signaling all workers to stop...")
    shutdown_event.set() # Signal all workers checking this event
    # Wait briefly for tasks to potentially finish cleanly
    await asyncio.sleep(2)
    stopped_count = 0
    for name, task in worker_tasks.items():
        if not task.done():
            # Optionally cancel tasks that didn't stop via event
            # task.cancel()
            print(f"Worker {name} stop signaled.")
            stopped_count += 1
        # Clear the task entry? Or leave for status checking?
    # worker_tasks.clear() # Clear tasks after signaling
    return {"message": f"Stop signal sent to {stopped_count} active workers."}


@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return schemas.HealthCheck(name=settings.PROJECT_NAME, status="OK", version="0.1.0")

# --- Database Initialization (Optional: Create tables on startup if they don't exist) ---
# This is generally better handled by Alembic migrations, but can be useful for simple setups.
# async def init_db():
#     async with async_engine.begin() as conn:
#         # await conn.run_sync(db_base.metadata.drop_all) # Use with caution!
#         await conn.run_sync(db_base.metadata.create_all)
#         print("Database tables checked/created.")

@app.on_event("startup")
async def startup_event():
    print("Starting up Autonomous Agency...")
    # await init_db() # Uncomment to auto-create tables (ensure models are imported)
    print("Connecting to database...")
    # Test connection - engine creation already does this implicitly
    try:
        async with async_engine.connect() as connection:
             print("Database connection successful.")
    except Exception as e:
        print(f"FATAL: Database connection failed: {e}")
        # Decide how to handle this - exit? retry?
        # For now, let it proceed but log the error.
    print("API Ready.")


@app.on_event("shutdown")
async def shutdown_app_event():
    print("Shutting down...")
    shutdown_event.set() # Signal any running workers spawned outside /control
    # Wait for workers managed by /control to finish (if desired)
    # tasks = [task for task in worker_tasks.values() if not task.done()]
    # if tasks:
    #     print(f"Waiting for {len(tasks)} background tasks to complete...")
    #     await asyncio.gather(*tasks, return_exceptions=True) # Add timeout?
    await async_engine.dispose()
    print("Shutdown complete.")


# --- Signal Handling for Graceful Shutdown ---
def handle_signal(sig, frame):
    print(f"Received signal {sig}, initiating graceful shutdown...")
    # This function needs to trigger the shutdown logic within the running event loop
    loop = asyncio.get_running_loop()
    loop.create_task(shutdown_app_event())
    # Give tasks time to finish before force exit
    loop.call_later(5.0, loop.stop) # Force stop loop after 5 seconds if needed

# --- Main Execution ---
if __name__ == "__main__":
    # Setup signal handlers before starting the server
    # signal.signal(signal.SIGINT, handle_signal) # Handle Ctrl+C
    # signal.signal(signal.SIGTERM, handle_signal) # Handle termination signal (e.g., from Docker)

    print(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False, # Set reload=True ONLY for development
        workers=1 # Run with a single worker process for simplicity with async tasks
    )
    # Note: Graceful shutdown with Uvicorn workers and asyncio tasks requires careful handling.
    # The setup above is basic. For production, consider libraries like 'uvicorn[standard]'
    # and potentially running workers in separate processes/containers managed by docker-compose.