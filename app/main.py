
# autonomous_agency/app/main.py
from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import signal
import os
import logging
from sqlalchemy.sql import text # For DB check

# Corrected imports for new structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.api.endpoints import payments as api_payments
    from Acumenis.app.db.base import engine as async_engine, Base as db_base, get_worker_session
    from Acumenis.app.db import models, crud
    from Acumenis.app.api import schemas
    from Acumenis.app.agents.agent_utils import load_and_update_api_keys, start_key_refresh_task, _key_refresh_task # Import key utils and task handle
except ImportError:
    print("[MainApp] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.api.endpoints import payments as api_payments
    from app.db.base import engine as async_engine, Base as db_base, get_worker_session
    from app.db import models, crud
    from app.api import schemas
    from app.agents.agent_utils import load_and_update_api_keys, start_key_refresh_task, _key_refresh_task

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Worker Control ---
worker_tasks: Dict[str, asyncio.Task] = {}
shutdown_event = asyncio.Event()

# --- FastAPI App Setup ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# --- Setup Templates ---
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
if not os.path.isdir(templates_dir):
    logger.warning(f"Templates directory not found at {templates_dir}, UI might not work.")
    templates = None
else:
    templates = Jinja2Templates(directory=templates_dir)

# --- API Routers ---
api_v1_router = FastAPI()
api_v1_router.include_router(api_payments.router, prefix="/payments", tags=["Payments & Orders"])
app.mount(settings.API_V1_STR, api_v1_router)


# --- Static Files ---
STATIC_DIR = "/app/static_website" # Served from container root
if not os.path.isdir(STATIC_DIR):
    logger.warning(f"Static website directory not found at {STATIC_DIR}. Website will not be served.")
else:
    os.makedirs(STATIC_DIR, exist_ok=True)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static_assets")

    # --- Serve Static HTML Pages ---
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def serve_index():
        path = os.path.join(STATIC_DIR, "index.html")
        if not os.path.exists(path): raise HTTPException(status_code=404, detail="index.html not found")
        return FileResponse(path)

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
        path = os.path.join(STATIC_DIR, "tos.html")
        if not os.path.exists(path): raise HTTPException(status_code=404, detail="tos.html not found")
        return FileResponse(path)

    @app.get("/order-success", response_class=FileResponse, include_in_schema=False)
    async def serve_order_success():
        path = os.path.join(STATIC_DIR, "order_success.html") # Assumes this file exists
        if not os.path.exists(path):
            path = os.path.join(STATIC_DIR, "index.html") # Fallback to index
            if not os.path.exists(path): raise HTTPException(status_code=404, detail="Fallback index.html not found")
        return FileResponse(path)

# --- Control Panel UI Endpoint ---
@app.get("/ui", response_class=HTMLResponse, tags=["Control UI"])
async def get_control_panel(request: Request):
    """Serves the HTML control panel UI."""
    if not templates:
        return HTMLResponse("<html><body><h1>Error</h1><p>UI Templates not configured.</p></body></html>", status_code=500)

    template_path = os.path.join(templates_dir, "control_panel.html")
    if not os.path.exists(template_path):
         logger.error(f"Control panel template not found at {template_path}")
         return HTMLResponse("<html><body><h1>Error</h1><p>Control panel template not found.</p></body></html>", status_code=500)

    worker_status_info = {name: ("Running" if task and not task.done() else "Stopped") for name, task in worker_tasks.items()}
    return templates.TemplateResponse("control_panel.html", {"request": request, "workers": worker_status_info})

# --- Worker Control API Endpoints (Mount under /control) ---
control_router = FastAPI()

# Dictionary mapping worker names to their runner functions
WORKER_RUNNERS = {}
def _import_runners():
    global WORKER_RUNNERS
    try:
        from Acumenis.app.workers.run_report_worker import run_report_generator_worker
        from Acumenis.app.workers.run_research_worker import run_prospect_researcher_worker
        from Acumenis.app.workers.run_email_worker import run_email_marketer_worker
        from Acumenis.app.workers.run_mcol_worker import run_mcol_worker
        from Acumenis.app.workers.run_key_acquirer_worker import run_key_acquirer_worker
        WORKER_RUNNERS = {
            "report_generator": run_report_generator_worker,
            "prospect_researcher": run_prospect_researcher_worker,
            "email_marketer": run_email_marketer_worker,
            "mcol": run_mcol_worker,
            "key_acquirer": run_key_acquirer_worker,
        }
    except ImportError:
        logger.error("Failed to import worker runner functions. Worker control API will be limited.", exc_info=True)
        # Fallback imports if needed
        try:
            from app.workers.run_report_worker import run_report_generator_worker
            from app.workers.run_research_worker import run_prospect_researcher_worker
            from app.workers.run_email_worker import run_email_marketer_worker
            from app.workers.run_mcol_worker import run_mcol_worker
            from app.workers.run_key_acquirer_worker import run_key_acquirer_worker
            WORKER_RUNNERS = {
                "report_generator": run_report_generator_worker,
                "prospect_researcher": run_prospect_researcher_worker,
                "email_marketer": run_email_marketer_worker,
                "mcol": run_mcol_worker,
                "key_acquirer": run_key_acquirer_worker,
            }
        except ImportError:
             logger.critical("CRITICAL: Failed even fallback imports for worker runners.")


@control_router.post("/start/{worker_name}", status_code=status.HTTP_200_OK)
async def start_worker(worker_name: str):
    """Starts a specific background worker."""
    if worker_name in worker_tasks and worker_tasks[worker_name] and not worker_tasks[worker_name].done():
        logger.info(f"Worker '{worker_name}' is already running.")
        return {"message": f"Worker '{worker_name}' is already running."}

    if worker_name not in WORKER_RUNNERS:
        logger.error(f"Worker '{worker_name}' not found in WORKER_RUNNERS.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    # Special check for KeyAcquirer
    if worker_name == "key_acquirer" and not getattr(settings, 'KEY_ACQUIRER_RUN_ON_STARTUP', False):
         logger.warning("Attempted to start Key Acquirer, but it's disabled in settings (KEY_ACQUIRER_RUN_ON_STARTUP=false).")
         raise HTTPException(status_code=400, detail="Key Acquirer is disabled in settings.")

    try:
        runner_func = WORKER_RUNNERS[worker_name]
        task = asyncio.create_task(runner_func(shutdown_event), name=f"worker_{worker_name}")
        worker_tasks[worker_name] = task
        logger.info(f"Started worker: {worker_name}")
        return {"message": f"Worker '{worker_name}' started."}
    except Exception as e:
        logger.error(f"Error starting worker {worker_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting worker {worker_name}: {str(e)}")


@control_router.post("/stop_all", status_code=status.HTTP_200_OK)
async def stop_all_workers():
    """Signals all running background workers to stop."""
    logger.info("Signaling all workers to stop via shared shutdown_event...")
    shutdown_event.set()
    await asyncio.sleep(0.1) # Give event loop a chance to propagate
    active_workers = [name for name, task in worker_tasks.items() if task and not task.done()]
    logger.info(f"Stop signal sent. Active workers ({len(active_workers)}) should stop on next cycle check.")
    return {"message": f"Stop signal sent. Active workers ({', '.join(active_workers)}) should stop on next cycle check."}

@control_router.get("/status", status_code=status.HTTP_200_OK)
async def get_worker_status():
    """Returns the status of each known worker."""
    status_info = {}
    # Ensure we report status for all possible workers, even if not currently running
    possible_workers = list(WORKER_RUNNERS.keys())
    for name in possible_workers:
        task = worker_tasks.get(name)
        if task:
            if task.done():
                try:
                    exception = task.exception()
                    status_info[name] = f"Stopped (Error: {type(exception).__name__})" if exception else "Stopped (Completed)"
                except asyncio.CancelledError:
                    status_info[name] = "Stopped (Cancelled)"
                except asyncio.InvalidStateError:
                     status_info[name] = "Stopped (Invalid State)"
            else:
                status_info[name] = "Running"
        else:
            status_info[name] = "Not Started"
    return status_info

app.mount("/control", control_router)


# --- Health Check ---
@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check():
    """Basic health check endpoint, including DB connectivity."""
    db_status = "Unknown"
    session = None
    try:
         session = await get_worker_session()
         await session.execute(text("SELECT 1"))
         db_status = "OK"
    except Exception as e:
         logger.error(f"Health check DB connection failed: {e}")
         db_status = f"DB Error" # Keep error details internal
    finally:
        if session: await session.close()

    # Check API key status (simple check if any keys loaded)
    key_status = "OK" if _keys_loaded and AVAILABLE_API_KEYS else ("No Keys Loaded" if _keys_loaded else "Keys Not Initialized")

    return schemas.HealthCheck(
        name=settings.PROJECT_NAME,
        status=f"API OK, DB: {db_status}, Keys: {key_status}",
        version="2.2.0-Prime" # Version Bump
    )

# --- App Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up {settings.PROJECT_NAME} - Acumenis Prime...")
    logger.info("Verifying database connection...")
    db_ready = False
    for i in range(5): # Retry DB connection check
        try:
            async with async_engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
            db_ready = True
            logger.info("Database connection successful.")
            break
        except Exception as e:
            logger.warning(f"DB connection attempt {i+1} failed: {e}. Retrying in 2s...")
            await asyncio.sleep(2)

    if not db_ready:
        logger.critical("FATAL: Database connection failed after multiple attempts. Check DB settings and availability.")
        # Optionally raise an exception or exit here in a real deployment
        # raise RuntimeError("Database connection failed on startup")

    logger.info("Importing worker runners...")
    _import_runners() # Populate WORKER_RUNNERS dict

    logger.info("Initial loading of API keys...")
    await load_and_update_api_keys() # Load keys into memory immediately

    logger.info("Starting background API key refresh task...")
    start_key_refresh_task() # Start the periodic refresh

    logger.info(f"API Ready. Acumenis Interface at {settings.AGENCY_BASE_URL}/. Control Panel at /ui.")
    logger.warning("Operating in high-risk, aggressive 'Victory Mandate' mode.")


@app.on_event("shutdown")
async def shutdown_app_event():
    logger.info("Shutdown signal received, initiating graceful shutdown...")
    if not shutdown_event.is_set():
        shutdown_event.set()

    # Cancel the key refresh task
    global _key_refresh_task # Ensure we access the global task handle
    if _key_refresh_task and not _key_refresh_task.done():
        logger.info("Cancelling background key refresh task...")
        _key_refresh_task.cancel()
        try:
            await _key_refresh_task
        except asyncio.CancelledError:
            logger.info("Key refresh task cancelled.")
        except Exception as e:
            logger.error(f"Error during key refresh task cancellation: {e}", exc_info=True)

    # Wait for worker tasks
    tasks_to_await = [task for task in worker_tasks.values() if task and not task.done()]
    if tasks_to_await:
        logger.info(f"Waiting for {len(tasks_to_await)} background worker tasks to complete (timeout: 10s)...")
        _, pending = await asyncio.wait(tasks_to_await, timeout=10.0)
        if pending:
            logger.warning(f"{len(pending)} tasks did not finish gracefully within timeout:")
            for task in pending:
                task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown_task'
                logger.warning(f" - Task: {task_name} - Force Cancelling...")
                task.cancel()
                try: await task # Allow cancellation to propagate
                except asyncio.CancelledError: logger.info(f"   - Task {task_name} force cancelled.")
                except Exception as e: logger.error(f"   - Error during task {task_name} force cancellation/cleanup: {e}", exc_info=True)

    logger.info("Disposing database engine...")
    await async_engine.dispose()
    logger.info("Shutdown complete.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    # Load .env if running directly (though docker-compose is standard)
    # from dotenv import load_dotenv
    # load_dotenv()
    uvicorn.run(
        "app.main:app", # Use relative path suitable for uvicorn
        host="0.0.0.0",
        port=8000,
        reload=False, # Production: False
        workers=1 # Production: Consider increasing based on load, but manage shared state carefully
    )

