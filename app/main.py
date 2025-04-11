from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import signal
import os

# Corrected imports for new structure
from Acumenis.app.core.config import settings
# from Acumenis.app.api.endpoints import requests as api_requests # Deprecated?
from Acumenis.app.api.endpoints import payments as api_payments
from Acumenis.app.db.base import engine as async_engine, Base as db_base
from Acumenis.app.db import models
from Acumenis.app.api import schemas

# --- Worker Control ---
worker_tasks = {}
shutdown_event = asyncio.Event()

# --- FastAPI App Setup ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- Setup Templates ---
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# --- API Routers ---
api_v1_router = FastAPI()
# api_v1_router.include_router(api_requests.router, prefix="/requests", tags=["Report Requests (Legacy?)"])
api_v1_router.include_router(api_payments.router, prefix="/payments", tags=["Payments & Orders"])
app.mount(settings.API_V1_STR, api_v1_router)


# --- Static Files (For potential assets like CSS, JS, images if separated later) ---
STATIC_DIR = "/app/static_website" # Served from container root
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static_assets")

# --- Serve Static HTML Pages ---
# Serve index.html at the root
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_index():
    path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)

# Serve other HTML pages
@app.get("/pricing", response_class=FileResponse, include_in_schema=False)
async def serve_pricing():
    path = os.path.join(STATIC_DIR, "pricing.html")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="pricing.html not found")
    return FileResponse(path)

@app.get("/order", response_class=FileResponse, include_in_schema=False)
async def serve_order():
    path = os.path.join(STATIC_DIR, "order.html")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="order.html not found")
    return FileResponse(path)

@app.get("/privacy", response_class=FileResponse, include_in_schema=False)
async def serve_privacy():
    path = os.path.join(STATIC_DIR, "privacy.html")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="privacy.html not found")
    return FileResponse(path)

@app.get("/terms", response_class=FileResponse, include_in_schema=False)
async def serve_terms():
    # Assuming terms of service file is named tos.html based on open tabs
    path = os.path.join(STATIC_DIR, "tos.html")
    if not os.path.exists(path): raise HTTPException(status_code=404, detail="tos.html not found")
    return FileResponse(path)

# Serve order success page (if needed by redirect_url)
@app.get("/order-success", response_class=FileResponse, include_in_schema=False)
async def serve_order_success():
    # Create a simple success page or use index/order page?
    # For now, serve index if a dedicated success page doesn't exist
    path = os.path.join(STATIC_DIR, "order_success.html") # Assumes this file exists
    if not os.path.exists(path):
        path = os.path.join(STATIC_DIR, "index.html") # Fallback to index
        if not os.path.exists(path): raise HTTPException(status_code=404, detail="Fallback index.html not found")
    return FileResponse(path)

# --- Control Panel UI Endpoint ---
@app.get("/ui", response_class=HTMLResponse, tags=["Control UI"])
async def get_control_panel(request: Request):
    """Serves the HTML control panel UI."""
    template_path = os.path.join(templates_dir, "control_panel.html")
    if not os.path.exists(template_path):
         return HTMLResponse("<html><body><h1>Error</h1><p>Control panel template not found.</p></body></html>", status_code=500)
    return templates.TemplateResponse("control_panel.html", {"request": request, "workers": worker_tasks})

# --- Worker Control API Endpoints (Mount under /control) ---
control_router = FastAPI()

@control_router.post("/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        return {"message": f"Worker '{worker_name}' is already running."}

    task = None
    try:
        # Use corrected import paths
        if worker_name == "report_generator":
            from Acumenis.app.workers.run_report_worker import run_report_generator_worker
            task = asyncio.create_task(run_report_generator_worker(shutdown_event), name=f"worker_{worker_name}")
        elif worker_name == "prospect_researcher":
            from Acumenis.app.workers.run_research_worker import run_prospect_researcher_worker
            task = asyncio.create_task(run_prospect_researcher_worker(shutdown_event), name=f"worker_{worker_name}")
        elif worker_name == "email_marketer":
            from Acumenis.app.workers.run_email_worker import run_email_marketer_worker
            task = asyncio.create_task(run_email_marketer_worker(shutdown_event), name=f"worker_{worker_name}")
        elif worker_name == "mcol":
            from Acumenis.app.workers.run_mcol_worker import run_mcol_worker
            task = asyncio.create_task(run_mcol_worker(shutdown_event), name=f"worker_{worker_name}")
        elif worker_name == "key_acquirer":
            from Acumenis.app.workers.run_key_acquirer_worker import run_key_acquirer_worker
            task = asyncio.create_task(run_key_acquirer_worker(shutdown_event), name=f"worker_{worker_name}")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

        if task:
            worker_tasks[worker_name] = task
            print(f"Started worker: {worker_name}")
            return {"message": f"Worker '{worker_name}' started."}
        else:
             raise HTTPException(status_code=500, detail=f"Failed to create task for worker '{worker_name}'.")
    except Exception as e:
        print(f"Error starting worker {worker_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting worker {worker_name}: {str(e)}")


@control_router.post("/stop_all", status_code=status.HTTP_200_OK)
async def stop_all_workers():
    """Signals all running background workers to stop."""
    print("Signaling all workers to stop via shared shutdown_event...")
    shutdown_event.set()
    await asyncio.sleep(0.1)
    await asyncio.sleep(1)
    stopped_count = 0
    active_workers = []
    for name, task in worker_tasks.items():
        if task and not task.done():
            print(f"Worker {name} stop signaled (will stop on next check).")
            active_workers.append(name)
        elif task and task.done():
             print(f"Worker {name} already stopped.")
    # shutdown_event.clear() # Optional: uncomment if needed to allow restart via UI
    return {"message": f"Stop signal sent via shared event. Active workers ({', '.join(active_workers)}) should stop on next cycle check."}

app.mount("/control", control_router)


# --- Health Check ---
@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check():
    """Basic health check endpoint."""
    return schemas.HealthCheck(name=settings.PROJECT_NAME, status="OK", version="2.0.0") # Version Bump!

# --- App Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    print(f"Starting up {settings.PROJECT_NAME}...")
    print("Connecting to database...")
    try:
        async with async_engine.connect() as connection:
            from sqlalchemy import text
            await connection.execute(text("SELECT 1"))
            print("Database connection successful.")
    except Exception as e:
        print(f"FATAL: Database connection failed on startup check: {e}")
    print(f"API Ready. Acumenis Interface at {settings.AGENCY_BASE_URL}/. Control Panel at /ui.")


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
                try: await task
                except asyncio.CancelledError: print(f"   - Task {task_name} cancelled.")
                except Exception as e: print(f"   - Error during task {task_name} cancellation/cleanup: {e}")

    print("Disposing database engine...")
    await async_engine.dispose()
    print("Shutdown complete.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    print(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    uvicorn.run(
        "Acumenis.app.main:app", # Correct path to the app instance
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )