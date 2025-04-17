# autonomous_agency/app/api/endpoints/payments.py
import hmac
import hashlib
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from typing import Optional

# Corrected relative imports for package structure
try:
    from Acumenis.app.core.config import settings
    from Acumenis.app.db import crud, models
    from Acumenis.app.db.base import get_db_session
    from Acumenis.app.api.schemas import CreateCheckoutRequest, CreateCheckoutResponse # Correct schema import path
except ImportError:
    # Fallback for potential direct execution or different structure
    print("[PaymentsAPI] WARNING: Using fallback imports. Ensure package structure is correct for deployment.")
    from app.core.config import settings
    from app.db import crud, models
    from app.db.base import get_db_session
    from app.api.schemas import CreateCheckoutRequest, CreateCheckoutResponse

router = APIRouter()
logger = logging.getLogger(__name__) # Setup logger

LEMONSQUEEZY_API_URL = "https://api.lemonsqueezy.com/v1"

async def get_ls_client() -> httpx.AsyncClient:
    """Creates an httpx client for Lemon Squeezy API calls with robust error handling."""
    if not settings.LEMONSQUEEZY_API_KEY:
         logger.critical("CRITICAL CONFIGURATION ERROR: Lemon Squeezy API Key (LEMONSQUEEZY_API_KEY) is not set.")
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Payment provider configuration incomplete.")
    headers = {
        "Authorization": f"Bearer {settings.LEMONSQUEEZY_API_KEY}",
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
        "User-Agent": f"{settings.PROJECT_NAME}/{settings.VERSION}",
    }
    # Increased timeout for payment provider interaction
    timeout = httpx.Timeout(45.0, connect=15.0)
    # Consider adding retries for transient network errors if needed (using Tenacity with httpx)
    return httpx.AsyncClient(base_url=LEMONSQUEEZY_API_URL, headers=headers, timeout=timeout, follow_redirects=True)

@router.post("/create-checkout", response_model=CreateCheckoutResponse, status_code=status.HTTP_201_CREATED)
async def create_lemon_squeezy_checkout(
    payload: CreateCheckoutRequest,
    ls_client: httpx.AsyncClient = Depends(get_ls_client)
):
    """
    Creates a Lemon Squeezy checkout session.
    Validates configuration and handles API errors robustly.
    """
    # Validate essential configuration for this endpoint
    required_configs = [
        settings.LEMONSQUEEZY_STORE_ID,
        settings.LEMONSQUEEZY_VARIANT_STANDARD,
        settings.LEMONSQUEEZY_VARIANT_PREMIUM,
        settings.AGENCY_BASE_URL
    ]
    if not all(required_configs):
        logger.critical("CRITICAL CONFIGURATION ERROR: Lemon Squeezy store/variant IDs or AGENCY_BASE_URL not fully configured.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Payment provider/agency configuration incomplete.")

    # Basic check for localhost in production-like environments
    if "localhost" in str(settings.AGENCY_BASE_URL) or "127.0.0.1" in str(settings.AGENCY_BASE_URL):
         # Allow for local testing, but WARN loudly. Production *needs* the real URL.
         logger.warning(f"AGENCY_BASE_URL ('{settings.AGENCY_BASE_URL}') appears to be localhost. Redirects/Webhooks will likely fail in production.")

    variant_id = None
    if payload.report_type == "standard_499":
        variant_id = settings.LEMONSQUEEZY_VARIANT_STANDARD
    elif payload.report_type == "premium_999":
        variant_id = settings.LEMONSQUEEZY_VARIANT_PREMIUM
    else:
        logger.warning(f"Invalid report_type received for checkout: {payload.report_type}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid report type specified.")

    # Ensure AGENCY_BASE_URL is correctly formatted (Pydantic AnyHttpUrl handles this)
    base_url = str(settings.AGENCY_BASE_URL).rstrip('/')

    # Data passed to webhook via Lemon Squeezy custom data
    # Keys here MUST align with what crud.create_initial_report_request expects from custom_data
    custom_data_payload = {
        "research_topic": payload.request_details,
        "company_name": payload.company_name,
    }

    checkout_data = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_options": {
                    "embed": False, # Use redirect flow for simplicity and robustness
                    "button_color": "#22d3ee" # Acumenis accent cyan
                },
                "checkout_data": {
                    "email": payload.client_email, # Pre-fill email
                    "name": payload.client_name, # Pre-fill name
                    "custom": custom_data_payload,
                },
                "redirect_url": f"{base_url}/order-success?session_id={{CHECKOUT_SESSION_ID}}", # order-success.html must exist
                # Optionally configure expires_at if needed
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": settings.LEMONSQUEEZY_STORE_ID}},
                "variant": {"data": {"type": "variants", "id": variant_id}},
            },
        }
    }

    try:
        async with ls_client: # Context manager ensures client closure
            response = await ls_client.post("/checkouts", json=checkout_data)
            response.raise_for_status() # Raises exception for 4xx/5xx
            ls_response_data = response.json()

        checkout_url = ls_response_data.get("data", {}).get("attributes", {}).get("url")
        checkout_id = ls_response_data.get("data", {}).get("id")

        if not checkout_url:
             logger.error("Lemon Squeezy API response missing checkout URL.")
             raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Payment provider response invalid (missing URL).")

        logger.info(f"Successfully created Lemon Squeezy checkout {checkout_id} for {payload.client_email}. URL: {checkout_url}")
        return CreateCheckoutResponse(checkout_url=checkout_url)

    except httpx.HTTPStatusError as e:
        error_detail = f"Payment provider API error (Status: {e.response.status_code})"
        try:
            # Attempt to parse Lemon Squeezy's specific error format
            ls_error = e.response.json().get('errors', [{}])[0]
            error_detail = ls_error.get('detail', ls_error.get('title', error_detail))
        except Exception:
            # Fallback if parsing fails
            error_detail += f": {e.response.text[:200]}"
        logger.error(f"Lemon Squeezy API Error creating checkout: {error_detail}", exc_info=False) # Don't need full trace for API errors usually
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Payment provider error: {error_detail}")
    except httpx.RequestError as e:
         logger.error(f"Network error creating Lemon Squeezy checkout: {e}", exc_info=True)
         raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail="Could not connect to payment provider.")
    except Exception as e:
        logger.error(f"Unexpected error creating Lemon Squeezy checkout: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not initiate payment process due to an internal error.")


# --- Webhook Handler ---
@router.post("/webhook", status_code=status.HTTP_200_OK, include_in_schema=False)
async def lemon_squeezy_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session), # Use FastAPI managed session
    background_tasks: BackgroundTasks = BackgroundTasks() # For potential post-processing
):
    """
    Handles incoming webhooks from Lemon Squeezy (e.g., order_created, order_refunded).
    CRITICAL: Verifies webhook signature before processing.
    Processes order_created atomically, creating ReportRequest and AgentTask.
    """
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
        logger.critical("CRITICAL CONFIGURATION ERROR: Lemon Squeezy Webhook secret (LEMONSQUEEZY_WEBHOOK_SECRET) is not set. Cannot process webhooks.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Webhook processing not configured.")

    raw_body = await request.body()

    # --- MANDATORY: Verify Webhook Signature ---
    try:
        computed_hash = hmac.new(
            settings.LEMONSQUEEZY_WEBHOOK_SECRET.encode('utf-8'),
            raw_body,
            hashlib.sha256
        )
        computed_signature = computed_hash.hexdigest()

        if not x_signature or not hmac.compare_digest(computed_signature, x_signature):
            logger.error(f"Invalid webhook signature received. Provided: '{x_signature}', Computed: '{computed_signature}'")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature.") # Use 401 Unauthorized
        logger.info("Webhook signature verified successfully.")
    except Exception as e:
         logger.error(f"Error during webhook signature verification: {e}", exc_info=True)
         # Use 500 as it's an internal server error during verification logic
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Webhook signature verification failed.")
    # --- End Signature Verification ---

    try:
        event_data = json.loads(raw_body)
        meta = event_data.get("meta", {})
        event_name = meta.get("event_name")
        custom_data_webhook = meta.get("custom_data", {}) # Custom data is in meta for webhooks
        order_data = event_data.get("data")
        is_test_mode = meta.get('test_mode', False)

        logger.info(f"Received Lemon Squeezy webhook event: '{event_name}' (Test Mode: {is_test_mode})")

        # --- Process 'order_created' ---
        if event_name == "order_created":
            if not order_data or order_data.get("type") != "orders":
                 logger.warning(f"Ignoring {event_name} webhook - missing or invalid order data.")
                 return {"message": "Webhook received, invalid data."} # Acknowledge receipt but note invalid data

            ls_order_id = order_data.get("id")
            logger.info(f"Processing 'order_created' webhook for LS Order ID: {ls_order_id}")
            # --- Transactional Update ---
            # get_db_session dependency handles commit/rollback
            try:
                report_request = await crud.create_initial_report_request(db, order_data, custom_data_webhook)

                if report_request:
                    # Determine priority based on report type
                    priority = 10 if report_request.report_type == 'premium_999' else 5

                    agent_task = await crud.create_agent_task(
                        db=db,
                        agent_name='ReportGenerator', # Must match the agent/worker name
                        goal=f'Generate {report_request.report_type} report for request {report_request.request_id}',
                        parameters={'report_request_id': report_request.request_id},
                        priority=priority
                    )
                    # Commit happens automatically via get_db_session context manager if no exceptions occur
                    logger.info(f"Successfully processed LS Order {ls_order_id}. Created ReportRequest {report_request.request_id} and AgentTask {agent_task.task_id}")
                    # Optionally trigger background task for post-processing (e.g., welcome email)
                    # background_tasks.add_task(send_order_confirmation, report_request.client_email, report_request.request_id)
                else:
                    # Duplicate order or creation failed, CRUD function logged this.
                    logger.info(f"Webhook for order {ls_order_id} skipped (duplicate or creation failed).")
                    # No commit needed as no changes were made or rollback occurred in CRUD

            except Exception as e:
                # Rollback handled by get_db_session dependency on exception
                logger.error(f"CRITICAL Error processing 'order_created' webhook DB operations for order {ls_order_id}: {e}", exc_info=True)
                # Return 500 to signal Lemon Squeezy to potentially retry (if configured)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal database error during webhook processing.")

        # --- Process 'order_refunded' ---
        elif event_name == "order_refunded":
             if not order_data or order_data.get("type") != "orders":
                 logger.warning(f"Ignoring {event_name} webhook - missing or invalid order data.")
                 return {"message": "Webhook received, invalid data."}

             ls_order_id_refund = order_data.get("id")
             logger.info(f"Processing 'order_refunded' webhook for LS Order ID: {ls_order_id_refund}")
             try:
                 # Find the original report request by LS Order ID
                 stmt = select(models.ReportRequest).where(models.ReportRequest.lemonsqueezy_order_id == str(ls_order_id_refund))
                 report_req_to_refund = await db.scalar(stmt)

                 if report_req_to_refund:
                     # Update status only if not already refunded
                     if report_req_to_refund.payment_status != "refunded":
                         logger.warning(f"Order {ls_order_id_refund} (ReportRequest {report_req_to_refund.request_id}) was refunded. Updating status to REFUNDED.")
                         report_req_to_refund.status = "REFUNDED" # Ensure REFUNDED is a valid status if needed
                         report_req_to_refund.payment_status = "refunded"
                         report_req_to_refund.error_message = f"Order refunded on {datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d')}"
                         # report_req_to_refund.updated_at = func.now() # Handled by model's onupdate

                         # TODO: Implement logic to cancel associated AgentTask if it's still PENDING or IN_PROGRESS
                         # task_to_cancel = await db.scalar(select(models.AgentTask).where(models.AgentTask.parameters['report_request_id'] == report_req_to_refund.request_id).where(models.AgentTask.status.in_(['PENDING', 'IN_PROGRESS'])))
                         # if task_to_cancel:
                         #     logger.info(f"Cancelling AgentTask {task_to_cancel.task_id} due to refund.")
                         #     task_to_cancel.status = "CANCELLED" # Add CANCELLED status? Or just FAILED?
                         #     task_to_cancel.result = "Cancelled due to order refund."

                         await db.commit() # Commit refund status update
                     else:
                         logger.info(f"Order {ls_order_id_refund} (ReportRequest {report_req_to_refund.request_id}) already marked as refunded.")
                 else:
                     logger.warning(f"Received refund webhook for LS Order ID {ls_order_id_refund}, but no matching ReportRequest found.")
             except Exception as e:
                 logger.error(f"Error processing 'order_refunded' webhook DB operations for order {ls_order_id_refund}: {e}", exc_info=True)
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal database error during refund webhook processing.")

        # --- Handle other potential events if needed ---
        # elif event_name == "subscription_payment_success":
        #     # Handle recurring payments if subscriptions are added
        #     pass
        else:
            logger.debug(f"Ignoring unhandled webhook event: {event_name}")

        # Return 200 OK to acknowledge receipt to Lemon Squeezy for successfully processed or ignored events
        return {"message": "Webhook received successfully."}

    except json.JSONDecodeError:
        logger.error("Webhook received invalid JSON payload.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.")
    except Exception as e:
        # Catch-all for unexpected errors during processing (after signature check)
        logger.error(f"Generic error processing webhook: {e}", exc_info=True)
        # Return 500 to indicate server error, Lemon Squeezy might retry
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error processing webhook.")