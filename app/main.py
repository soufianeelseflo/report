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
from app.api.endpoints import payments as api_payments # Import payments router
from app.db.base import engine as async_engine, Base as db_base # Import engine and Base
from app.db import models # Ensure models are imported so Base knows about them

# --- Worker Control ---
worker_tasks = {}
shutdown_event = asyncio.Event()

# --- FastAPI App Setup ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- API Routers ---
app.include_router(api_requests.router, prefix=f"{settings.API_V1_STR}/requests", tags=["Report Requests"])
app.include_router(api_payments.router, prefix=f"{settings.API_V1_STR}/payments", tags=["Payments"]) # Include payments router


# --- Static Files (for AI-Generated Website) ---
STATIC_DIR = "/app/static_website"
# Ensure directory exists (created by Dockerfile)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        # Fallback if MCOL hasn't generated the site yet
        return HTMLResponse("""
        <html>
            <head><title>Agency Backend Running</title></head>
            <body>
                <h1>Autonomous AI Reporting Agency - Backend Active</h1>
                <p>The core engine is running.</p>
                <p>The MCOL agent should generate the public website shortly.</p>
                <p>Check logs for MCOL status and suggestions.</p>
                <p>API available at /docs</p>
            </body>
        </html>
        """, status_code=200)

# --- Worker Control Endpoints ---
@app.post("/control/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is already running."}

    task = None
    if worker_name == "report_generator":
        from app.workers.run_report_worker import run_report_generator_worker
        task = asyncio.create_task(run_report_generator_worker(shutdown_event), name=f"worker_{worker_name}")
    elif worker_name == "prospect_researcher":
        from app.workers.run_research_worker import run_prospect_researcher_worker
        task = asyncio.create_task(run_prospect_researcher_worker(shutdown_event), name=f"worker_{worker_name}")
    elif worker_name == "email_marketer":
        from app.workers.run_email_worker import run_email_marketer_worker
        task = asyncio.create_task(run_email_marketer_worker(shutdown_event), name=f"worker_{worker_name}")
    elif worker_name == "mcol":
        from app.workers.run_mcol_worker import run_mcol_worker
        task = asyncio.create_task(run_mcol_worker(shutdown_event), name=f"worker_{worker_name}")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    if task:
        worker_tasks[worker_name] = task
        print(f"Started worker: {worker_name}")
        return {"message": f"Worker '{worker_name}' started."}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to create task for worker '{worker_name}'.")


@app.post("/control/stop/{worker_name}", status_code=status.HTTP_200_OK)
async def stop_worker(worker_name: str):
    """Signals a specific background worker to stop."""
    if worker_name not in worker_tasks or worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is not running or already stopped."}

    print(f"Signaling worker '{worker_name}' to stop...")
    # Signal via shared event. Actual stop happens in worker loop.
    shutdown_event.set() # Signal all workers
    # Optionally, try to cancel just one task (less graceful)
    # worker_tasks[worker_name].cancel()
    return {"message": f"Stop signal sent via shared event. Worker '{worker_name}' should stop on next check."}

@app.post("/control/stop_all", status_code=status.HTTP_200_OK)
async def stop_all_workers():
    """Signals all running background workers to stop."""
    print("Signaling all workers to stop via shared shutdown_event...")
    shutdown_event.set()
    await asyncio.sleep(2)
    stopped_count = 0
    active_workers = []
    for name, task in worker_tasks.items():
        if task and not task.done():
            print(f"Worker {name} stop signaled.")
            stopped_count += 1
            active_workers.append(name)
    return {"message": f"Stop signal sent via shared event. {stopped_count} workers ({', '.join(active_workers)}) should stop on next cycle check."}


@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return schemas.HealthCheck(name=settings.PROJECT_NAME, status="OK", version="1.0.0") # Use a version

@app.on_event("startup")
async def startup_event():
    print("Starting up Autonomous Agency...")
    print("Connecting to database...")
    try:
        async with async_engine.connect() as connection:
            await connection.execute(models.select(1))
            print("Database connection successful.")
    except Exception as e:
        print(f"FATAL: Database connection failed on startup check: {e}")
        # raise SystemExit("Database connection failed on startup") from e
    print("API Ready. Use /control/start/{worker_name} to activate agents.")


@app.on_event("shutdown")
async def shutdown_app_event():
    print("Shutdown signal received, initiating graceful shutdown...")
    if not shutdown_event.is_set():
        shutdown_event.set()

    tasks_to_await = [task for task in worker_tasks.values() if task and not task.done()]
    if tasks_to_await:
        print(f"Waiting for {len(tasks_to_await)} background tasks to complete...")
        _, pending = await asyncio.wait(tasks_to_await, timeout=10.0)
        if pending:
            print(f"Warning: {len(pending)} tasks did not finish gracefully within timeout:")
            for task in pending:
                task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown_task'
                print(f" - Task: {task_name} - Cancelling...")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    print(f"   - Task {task_name} cancelled.")
                except Exception as e:
                    print(f"   - Error during task {task_name} cancellation/cleanup: {e}")

    print("Disposing database engine...")
    await async_engine.dispose()
    print("Shutdown complete.")


if __name__ == "__main__":
    print(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )