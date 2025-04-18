# autonomous_agency/app/main.py
from fastapi import (
    FastAPI, Depends, Request, HTTPException, status, APIRouter, BackgroundTasks
)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import asyncio
import signal
import os
import logging
from sqlalchemy.sql import text
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, List, Optional, Callable, Awaitable

# --- Core Application Imports ---
try:
    # Assuming running from project root or PYTHONPATH is set
    from app.core.config import settings
    from app.api.endpoints import payments as api_payments
    from app.db.base import (
        engine as async_engine, Base as db_base,
        get_worker_session, get_db_session # Ensure get_db_session is imported
    )
    from app.db import models, crud # Import crud
    from app.api import schemas # Import base schemas
    # MODIFIED: Removed KeyAcquirer imports, kept agent_utils imports
    from app.agents.agent_utils import (
        # load_and_update_api_keys, # REMOVED - No DB keys to load
        # start_key_refresh_task, _key_refresh_task, # REMOVED - No refresh task
        # AVAILABLE_API_KEYS, _keys_loaded, # REMOVED - Using single key
        shutdown_event # Import shared shutdown_event
    )
except ImportError as e:
    print(f"[MainApp] CRITICAL IMPORT ERROR: {e}. Check PYTHONPATH or package structure.")
    # Attempt fallback - less reliable
    try:
        from Acumenis.app.core.config import settings
        from Acumenis.app.api.endpoints import payments as api_payments
        from Acumenis.app.db.base import engine as async_engine, Base as db_base, get_worker_session, get_db_session
        from Acumenis.app.db import models, crud
        from Acumenis.app.api import schemas
        from Acumenis.app.agents.agent_utils import (
            # load_and_update_api_keys, # REMOVED
            # start_key_refresh_task, _key_refresh_task, # REMOVED
            # AVAILABLE_API_KEYS, _keys_loaded, # REMOVED
            shutdown_event
        )
        print("[MainApp] WARNING: Using Acumenis fallback imports.")
    except ImportError:
         print("[MainApp] FATAL: Fallback imports also failed. Cannot proceed.")
         raise SystemExit("Failed to import critical application modules.")


# --- Logging Configuration ---
# Apply consistent formatting and levels
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING) # Reduce access log noise
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING) # Reduce SQLAlchemy noise


# --- Worker Control Globals ---
worker_tasks: Dict[str, asyncio.Task] = {}
# shutdown_event is imported from agent_utils
WORKER_RUNNERS: Dict[str, Callable[[asyncio.Event], Awaitable[None]]] = {} # Type hint for runner functions

# --- FastAPI App Setup ---
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous AI Agency for Market Research Reports",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc"
)

# --- Template Engine Setup ---
# Determine templates directory relative to this main.py file
script_dir = os.path.dirname(__file__)
templates_dir = os.path.join(script_dir, "templates")
if not os.path.isdir(templates_dir):
    logger.error(f"Templates directory not found at {templates_dir}. Control Panel UI will fail.")
    templates = None
else:
    templates = Jinja2Templates(directory=templates_dir)

# --- API Routers ---
api_v1_router = APIRouter()
api_v1_router.include_router(api_payments.router, prefix="/payments", tags=["Payments & Orders"])
# Add other functional API routers here as the application grows
# Example: api_v1_router.include_router(admin_api.router, prefix="/admin", tags=["Admin"])

app.include_router(api_v1_router, prefix=settings.API_V1_STR)


# --- Static Files Setup ---
# Determine static files directory relative to the project root
project_root = os.path.dirname(script_dir) # Assumes main.py is in 'app' directory
STATIC_DIR = os.path.join(project_root, "static_website")

if not os.path.isdir(STATIC_DIR):
    # Fallback check for path relative to container root if running in Docker
    STATIC_DIR_ALT = "/app/static_website"
    if os.path.isdir(STATIC_DIR_ALT):
        STATIC_DIR = STATIC_DIR_ALT
    else:
        logger.error(f"Static website directory not found at {STATIC_DIR} or {STATIC_DIR_ALT}. Website frontend will not be served.")
        STATIC_DIR = None

if STATIC_DIR:
    logger.info(f"Serving static files from: {STATIC_DIR}")
    # Mount at /assets to avoid conflict with API/UI routes - ensures clean separation
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="static_assets")

    # Serve primary HTML pages from the root path
    static_pages = ["index.html", "pricing.html", "order.html", "privacy.html", "tos.html", "order_success.html"]
    fallback_page = os.path.join(STATIC_DIR, "index.html") # Default fallback

    async def serve_static_page(page_name: str):
        path = os.path.join(STATIC_DIR, page_name)
        if not os.path.exists(path):
             logger.warning(f"Static page '{page_name}' not found at {path}. Serving fallback index.html.")
             if os.path.exists(fallback_page):
                 return FileResponse(fallback_page)
             else:
                  logger.error(f"FATAL: Fallback page index.html not found at {fallback_page}")
                  raise HTTPException(status_code=404, detail="Requested page not found.")
        return FileResponse(path)

    # Define routes for serving static pages
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def serve_index(): return await serve_static_page("index.html")

    @app.get("/pricing", response_class=FileResponse, include_in_schema=False)
    async def serve_pricing(): return await serve_static_page("pricing.html")

    @app.get("/order", response_class=FileResponse, include_in_schema=False)
    async def serve_order(): return await serve_static_page("order.html")

    @app.get("/privacy", response_class=FileResponse, include_in_schema=False)
    async def serve_privacy(): return await serve_static_page("privacy.html")

    @app.get("/terms", response_class=FileResponse, include_in_schema=False)
    async def serve_terms(): return await serve_static_page("tos.html") # Alias for consistency

    @app.get("/tos", response_class=FileResponse, include_in_schema=False)
    async def serve_tos(): return await serve_static_page("tos.html")

    @app.get("/order-success", response_class=FileResponse, include_in_schema=False)
    async def serve_order_success(): return await serve_static_page("order_success.html")

# --- Internal Worker/System Control Logic ---

def _import_runners():
    """Dynamically imports worker runner functions and populates WORKER_RUNNERS."""
    global WORKER_RUNNERS
    runners = {}
    # MODIFIED: Removed KeyAcquirer from the list
    worker_modules = [
        ("report_generator", "Acumenis.app.workers.run_report_worker", "run_report_generator_worker"),
        ("prospect_researcher", "Acumenis.app.workers.run_research_worker", "run_prospect_researcher_worker"),
        ("email_marketer", "Acumenis.app.workers.run_email_worker", "run_email_marketer_worker"),
        ("mcol", "Acumenis.app.workers.run_mcol_worker", "run_mcol_worker"),
        # ("key_acquirer", "Acumenis.app.workers.run_key_acquirer_worker", "run_key_acquirer_worker"), # REMOVED
    ]
    # Fallback path if primary fails
    fallback_worker_modules = [
        ("report_generator", "app.workers.run_report_worker", "run_report_generator_worker"),
        ("prospect_researcher", "app.workers.run_research_worker", "run_prospect_researcher_worker"),
        ("email_marketer", "app.workers.run_email_worker", "run_email_marketer_worker"),
        ("mcol", "app.workers.run_mcol_worker", "run_mcol_worker"),
        # ("key_acquirer", "app.workers.run_key_acquirer_worker", "run_key_acquirer_worker"), # REMOVED
    ]

    import importlib
    for name, module_path, func_name in worker_modules:
        try:
            module = importlib.import_module(module_path)
            runners[name] = getattr(module, func_name)
        except ImportError:
            logger.warning(f"Could not import primary worker runner: {module_path}.{func_name}. Trying fallback...")
            try:
                 fb_module_path = module_path.replace("Acumenis.app.", "app.")
                 module = importlib.import_module(fb_module_path)
                 runners[name] = getattr(module, func_name)
                 logger.info(f"Successfully imported fallback runner for {name}.")
            except ImportError:
                 logger.error(f"Could not import runner for worker '{name}' from primary or fallback path.")

    WORKER_RUNNERS = runners
    logger.info(f"Worker runners loaded for: {list(WORKER_RUNNERS.keys())}")


async def _start_worker_task(worker_name: str) -> bool:
    """Creates and stores worker task if runner exists. Returns True on success."""
    if worker_name not in WORKER_RUNNERS:
        logger.error(f"Runner function for worker '{worker_name}' not found. Cannot start.")
        return False

    # Prevent starting if already running
    if worker_name in worker_tasks and not worker_tasks[worker_name].done():
        logger.warning(f"Worker '{worker_name}' is already running. Start request ignored.")
        return True # Indicate it's running

    runner_func = WORKER_RUNNERS[worker_name]
    logger.info(f"Attempting to start worker: {worker_name}")
    try:
        # Pass the shared shutdown_event to the worker runner
        task = asyncio.create_task(runner_func(shutdown_event), name=f"worker_{worker_name}")
        worker_tasks[worker_name] = task
        # Add a callback to log completion or errors
        task.add_done_callback(lambda t: _worker_task_done_callback(worker_name, t))
        logger.info(f"Successfully started worker task: {worker_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create task for worker {worker_name}: {e}", exc_info=True)
        return False

def _worker_task_done_callback(worker_name: str, task: asyncio.Task):
    """Callback function when a worker task finishes."""
    try:
        exc = task.exception()
        if exc:
            logger.error(f"Worker '{worker_name}' task completed with error: {exc}", exc_info=exc)
        elif task.cancelled():
            logger.info(f"Worker '{worker_name}' task was cancelled.")
        else:
            logger.info(f"Worker '{worker_name}' task completed normally.")
    except asyncio.CancelledError:
        logger.info(f"Worker '{worker_name}' task was cancelled (callback).")
    except Exception as e:
        logger.error(f"Error in worker task done callback for '{worker_name}': {e}", exc_info=True)
    # Optionally remove from worker_tasks dict upon completion?
    # if worker_name in worker_tasks:
    #     del worker_tasks[worker_name]


async def stop_all_workers():
     """Internal function to signal workers to stop."""
     # MODIFIED: Removed key refresh task stop
     if not shutdown_event.is_set():
        logger.info("Signaling all workers to stop via shared shutdown_event...")
        shutdown_event.set()
        await asyncio.sleep(0.1) # Give event loop a chance to propagate signal
        logger.info("Shutdown signal sent.")

async def get_worker_status_internal() -> Dict[str, str]:
    """Internal function to get worker statuses."""
    status_info = {}
    possible_workers = list(WORKER_RUNNERS.keys()) # Base status on imported runners
    for name in possible_workers:
        task = worker_tasks.get(name)
        if task:
            if task.done():
                # Check for exceptions after task is done
                try:
                    exception = task.exception()
                    status_info[name] = f"Stopped (Error: {type(exception).__name__})" if exception else "Stopped (Completed)"
                except asyncio.CancelledError:
                    status_info[name] = "Stopped (Cancelled)"
                except asyncio.InvalidStateError:
                     status_info[name] = "Unknown (Task Done, Invalid State)"
            else:
                status_info[name] = "Running"
        else:
            status_info[name] = "Not Started"
    return status_info

# --- Control Panel UI Endpoint ---
@app.get("/ui", response_class=HTMLResponse, tags=["Control UI"], include_in_schema=False)
async def get_control_panel(request: Request):
    """Serves the HTML control panel UI with real-time worker status."""
    if not templates:
        logger.error("UI Templates directory not found or not configured.")
        return HTMLResponse("<html><body><h1>Internal Error</h1><p>UI cannot be displayed.</p></body></html>", status_code=500)

    template_path = os.path.join(templates_dir, "control_panel.html")
    if not os.path.exists(template_path):
         logger.error(f"Control panel template not found at {template_path}")
         return HTMLResponse("<html><body><h1>Internal Error</h1><p>Control panel template missing.</p></body></html>", status_code=500)

    # Fetch real-time worker status
    worker_status_info = await get_worker_status_internal()
    # Prepare data for the template (ensure all known workers are listed)
    template_data = {
        "request": request,
        # MODIFIED: Filter workers based on WORKER_RUNNERS keys
        "workers": [{"name": name, "status": worker_status_info.get(name, "Unknown")} for name in WORKER_RUNNERS.keys()]
    }
    return templates.TemplateResponse("control_panel.html", template_data)

# --- Worker Control API ---
control_router = APIRouter(tags=["Worker Control"])

@control_router.post("/start/{worker_name}", status_code=status.HTTP_200_OK, response_model=dict)
async def start_worker_endpoint(worker_name: str):
    """Starts a specific background worker via API."""
    if worker_name not in WORKER_RUNNERS:
        logger.error(f"API request to start unknown worker '{worker_name}'.")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Worker '{worker_name}' not found.")

    success = await _start_worker_task(worker_name)
    if success:
        # Re-fetch status after attempting start
        current_status = await get_worker_status_internal()
        return {"message": f"Worker '{worker_name}' start initiated.", "status": current_status.get(worker_name, "Unknown")}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start worker '{worker_name}'. Check logs.")


@control_router.post("/stop_all", status_code=status.HTTP_200_OK, response_model=dict)
async def stop_all_workers_endpoint():
    """Signals all running background workers to stop gracefully via API."""
    logger.info("Received API request to stop all workers...")
    await stop_all_workers() # Call internal stop function
    # Allow a moment for signal propagation before checking status
    await asyncio.sleep(0.5)
    current_status = await get_worker_status_internal()
    active_workers = [name for name, status_str in current_status.items() if status_str == "Running"]
    return {"message": f"Stop signal sent. Active workers ({len(active_workers)}) should stop on next cycle check.", "current_status": current_status}

@control_router.get("/status", status_code=status.HTTP_200_OK, response_model=Dict[str, str])
async def get_worker_status_endpoint():
    """Returns the current status of each known worker via API."""
    return await get_worker_status_internal()

app.include_router(control_router, prefix="/control") # Mount control routes under /control

# --- Health Check ---
@app.get("/health", response_model=schemas.HealthCheck, tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db_session)): # Use managed session
    """Provides a health check: API status, DB connectivity, API key status."""
    db_status = "OK"
    try:
         await db.execute(text("SELECT 1"))
    except Exception as e:
         logger.error(f"Health check DB connection failed: {e}", exc_info=False) # Keep log less verbose
         db_status = "DB Error"

    # MODIFIED: Check if the single API key is configured
    key_status = "OK"
    if not settings.OPENROUTER_API_KEY:
        key_status = "API Key Missing"
        logger.warning("Health Check: OPENROUTER_API_KEY is not configured.")

    overall_status = "OK" if db_status == "OK" and key_status == "OK" else "ERROR" if key_status == "API Key Missing" else "WARN"

    return schemas.HealthCheck(
        name=settings.PROJECT_NAME,
        status=f"API: {overall_status}, DB: {db_status}, Key: {key_status}",
        version=settings.VERSION
    )

# --- Application Lifecycle Events ---

async def check_db_connection(retries=5, delay=3):
     """Checks DB connection with exponential backoff."""
     for i in range(retries):
        session = None
        try:
            logger.info(f"DB connection check attempt {i+1}/{retries}...")
            session = await get_worker_session() # Use worker session for startup check
            await session.execute(text("SELECT 1"))
            logger.info("Database connection successful.")
            await session.close()
            return True
        except Exception as e:
            logger.warning(f"DB connection attempt {i+1} failed: {e}. Retrying in {delay}s...")
            if session: await session.close()
            if i < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logger.critical("FATAL: Database connection failed after multiple attempts. Check DB settings and availability.")
                return False
     return False # Should not be reached, but satisfy linter

@app.on_event("startup")
async def startup_event():
    logger.info(f"--- {settings.PROJECT_NAME} v{settings.VERSION} Startup Sequence Initiated ---")

    # 1. Verify Database Connection
    if not await check_db_connection():
        logger.critical("Startup halted due to database connection failure.")
        raise SystemExit("Database connection failed on startup")

    # 2. Import Worker Runners
    logger.info("Importing worker runners...")
    _import_runners() # This now excludes KeyAcquirer runner

    # 3. Verify Single API Key Presence
    logger.info("Verifying API key configuration...")
    if not settings.OPENROUTER_API_KEY:
         logger.critical("FATAL: OPENROUTER_API_KEY is not set in environment. LLM calls will fail.")
         # Consider stopping startup if key is essential
         raise SystemExit("Missing required OPENROUTER_API_KEY")
    else:
         logger.info(f"Single API key found (...{settings.OPENROUTER_API_KEY[-4:]}).")

    # 4. Start Background Key Refresh Task - REMOVED

    # 5. Conditionally Start KeyAcquirer - REMOVED

    # 6. Start Core Workers Automatically
    # MODIFIED: Removed key_acquirer from core_workers list
    core_workers = ["report_generator", "prospect_researcher", "email_marketer", "mcol"]
    logger.info(f"Starting core workers: {', '.join(core_workers)}...")
    start_results = await asyncio.gather(*[_start_worker_task(name) for name in core_workers if name in WORKER_RUNNERS])
    failed_starts = [core_workers[i] for i, success in enumerate(start_results) if not success]
    if failed_starts:
        logger.error(f"Failed to start the following core workers: {', '.join(failed_starts)}")

    logger.info(f"--- {settings.PROJECT_NAME} Startup Complete ---")
    logger.info(f"Access API at {settings.AGENCY_BASE_URL}{settings.API_V1_STR}")
    if STATIC_DIR: logger.info(f"Access Frontend at {settings.AGENCY_BASE_URL}/")
    if templates: logger.info(f"Access Control Panel at {settings.AGENCY_BASE_URL}/ui")
    logger.warning("Acumenis Prime Operational. Velocity Protocol Engaged.")


@app.on_event("shutdown")
async def shutdown_app_event():
    logger.info(f"--- {settings.PROJECT_NAME} Shutdown Sequence Initiated ---")
    await stop_all_workers() # Signal shared event first

    # Cancel the key refresh task - REMOVED

    # Wait for worker tasks with timeout
    tasks_to_await = [task for task in worker_tasks.values() if task and not task.done()]
    if tasks_to_await:
        logger.info(f"Waiting up to 20s for {len(tasks_to_await)} background worker tasks to complete...")
        done, pending = await asyncio.wait(tasks_to_await, timeout=20.0)

        for task in pending:
            task_name = task.get_name() if hasattr(task, 'get_name') else 'unknown_task'
            logger.warning(f"Task {task_name} did not finish gracefully within timeout. Force Cancelling...")
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=2.0) # Short wait for forced cancel
            except asyncio.CancelledError: logger.info(f"   - Task {task_name} force cancelled.")
            except asyncio.TimeoutError: logger.error(f"   - Task {task_name} did not respond to force cancellation.")
            except Exception as e: logger.error(f"   - Error during task {task_name} force cancellation: {e}")
        if done:
             logger.info(f"{len(done)} worker tasks completed gracefully.")

    # Dispose DB engine
    logger.info("Disposing database engine...")
    await async_engine.dispose()

    logger.info(f"--- {settings.PROJECT_NAME} Shutdown Complete ---")


# --- Main Execution Guard ---
if __name__ == "__main__":
    logger.info(f"Starting Uvicorn server for {settings.PROJECT_NAME}...")
    # .env loading should be handled by docker-compose or external process

    # Graceful shutdown handling for direct uvicorn run
    # Note: Uvicorn itself handles SIGINT/SIGTERM to trigger shutdown events
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False, # Never use reload in production
        workers=int(os.getenv("WEB_CONCURRENCY", 4)), # Default to 4 workers for speed
        loop="uvloop" if "uvloop" in globals() else "asyncio", # Use uvloop if available
        http="httptools" if "httptools" in globals() else "auto", # Use httptools if available
        # Add proxy headers if running behind a reverse proxy like Nginx/Traefik
        # proxy_headers=True,
        # forwarders_allow_ips='*' # Be careful with this in production
    )