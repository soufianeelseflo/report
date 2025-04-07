# autonomous_agency/app/api/endpoints/payments.py
import hmac
import hashlib
import json
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from typing import Optional

from app.core.config import settings
from app.db import crud, models
from app.db.base import get_db_session
from pydantic import BaseModel, Field

router = APIRouter()

LEMONSQUEEZY_API_URL = "https://api.lemonsqueezy.com/v1"

class CreateCheckoutRequest(BaseModel):
    # Removed report_request_id - we create the request *after* payment now
    report_type: str = Field(..., description="Type of report: 'standard_499' or 'premium_999'")
    client_email: str
    client_name: Optional[str] = None
    company_name: Optional[str] = None # Optional: Collect company name
    request_details: str = Field(..., description="Specific details or topic for the report")

class CreateCheckoutResponse(BaseModel):
    checkout_url: str

async def get_ls_client() -> httpx.AsyncClient:
    """Creates an httpx client for Lemon Squeezy API calls."""
    if not settings.LEMONSQUEEZY_API_KEY:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy API Key not configured.")
    headers = {
        "Authorization": f"Bearer {settings.LEMONSQUEEZY_API_KEY}",
        "Accept": "application/vnd.api+json",
        "Content-Type": "application/vnd.api+json",
    }
    return httpx.AsyncClient(base_url=LEMONSQUEEZY_API_URL, headers=headers, timeout=20.0)

@router.post("/create-checkout", response_model=CreateCheckoutResponse, status_code=status.HTTP_201_CREATED)
async def create_lemon_squeezy_checkout(
    payload: CreateCheckoutRequest,
    ls_client: httpx.AsyncClient = Depends(get_ls_client)
    # No DB session needed here initially
):
    """Creates a Lemon Squeezy checkout session."""
    if not settings.LEMONSQUEEZY_STORE_ID or not settings.LEMONSQUEEZY_VARIANT_STANDARD or not settings.LEMONSQUEEZY_VARIANT_PREMIUM:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy store/variant IDs not configured.")

    # 1. Determine variant ID based on report type
    variant_id = None
    if payload.report_type == "standard_499":
        variant_id = settings.LEMONSQUEEZY_VARIANT_STANDARD
    elif payload.report_type == "premium_999":
        variant_id = settings.LEMONSQUEEZY_VARIANT_PREMIUM
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid report type for checkout.")

    # 2. Prepare checkout data for Lemon Squeezy API
    #    Pass necessary info via custom_data to be retrieved by webhook
    custom_data_payload = {
        "report_type": payload.report_type,
        "request_details": payload.request_details,
        "client_name": payload.client_name,
        "company_name": payload.company_name,
        # Add any other info needed post-payment
    }

    checkout_data = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_options": {
                    "embed": False,
                },
                "checkout_data": {
                    "email": payload.client_email,
                    "name": payload.client_name,
                    # Pass our internal data via custom fields
                    "custom": custom_data_payload,
                },
                # Redirect URLs (Use your actual deployed URLs from settings)
                "redirect_url": f"{settings.AGENCY_BASE_URL}/order-success?session_id={{CHECKOUT_SESSION_ID}}", # Example success page
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": settings.LEMONSQUEEZY_STORE_ID}},
                "variant": {"data": {"type": "variants", "id": variant_id}},
            },
        }
    }

    # 3. Call Lemon Squeezy API
    try:
        async with ls_client:
            response = await ls_client.post("/checkouts", json=checkout_data)
            response.raise_for_status() # Raise exception for 4xx/5xx errors
            ls_response_data = response.json()
            checkout_url = ls_response_data.get("data", {}).get("attributes", {}).get("url")
            checkout_id = ls_response_data.get("data", {}).get("id") # Get checkout ID

            if not checkout_url:
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy did not return a checkout URL.")

            print(f"Created Lemon Squeezy checkout {checkout_id} for {payload.client_email}")
            # We don't create the ReportRequest here; webhook does upon successful payment.

            return CreateCheckoutResponse(checkout_url=checkout_url)

    except httpx.HTTPStatusError as e:
        error_detail = "Unknown payment provider error"
        try:
            error_payload = e.response.json()
            error_detail = error_payload.get('errors', [{}])[0].get('detail', 'Unknown error')
        except Exception:
            pass # Ignore JSON parsing errors on error response
        print(f"Lemon Squeezy API Error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Payment provider error: {error_detail}")
    except Exception as e:
        print(f"Error creating Lemon Squeezy checkout: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not initiate payment process.")


# --- Webhook Handler ---
@router.post("/webhook", status_code=status.HTTP_200_OK, include_in_schema=False) # Hide from OpenAPI docs
async def lemon_squeezy_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session)
):
    """Handles incoming webhooks from Lemon Squeezy."""
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
        print("CRITICAL: Lemon Squeezy Webhook secret not set. Cannot process webhooks.")
        # Return 200 OK to prevent retries, but log critical error
        return {"message": "Webhook ignored, secret not configured."}

    # 1. Verify Signature
    raw_body = await request.body()
    try:
        computed_signature = hmac.new(
            settings.LEMONSQUEEZY_WEBHOOK_SECRET.encode('utf-8'),
            raw_body,
            hashlib.sha256
        ).hexdigest()

        if not x_signature or not hmac.compare_digest(computed_signature, x_signature):
            print("Webhook signature mismatch!")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature.")
    except Exception as e:
         print(f"Error verifying webhook signature: {e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Signature verification failed.")

    # 2. Process Event
    try:
        event_data = json.loads(raw_body) # Use json.loads on raw body
        meta = event_data.get("meta", {})
        event_name = meta.get("event_name")
        custom_data = meta.get("custom_data", {}) # Custom data passed during checkout creation
        order_data = event_data.get("data") # The actual order object

        print(f"Received Lemon Squeezy webhook: {event_name}")

        # --- Process successful order ---
        if event_name == "order_created":
            if order_data and order_data.get("type") == "orders":
                try:
                    # Use the custom data passed during checkout creation
                    # to get the necessary details for the report request
                    report_request = await crud.create_report_request_from_webhook(db, order_data)
                    if report_request:
                        await db.commit()
                        print(f"Successfully processed order {order_data.get('id')} and created ReportRequest {report_request.request_id}")
                    else:
                        # Order might already be processed or missing data
                        await db.rollback() # Rollback if creation failed or skipped

                except Exception as e:
                     print(f"Error processing 'order_created' webhook for order {order_data.get('id')}: {e}")
                     await db.rollback() # Rollback DB changes on error
                     # Return 500 to potentially trigger retry from Lemon Squeezy? Check docs.
                     # raise HTTPException(status_code=500, detail="Internal error processing webhook")
                     # For now, return 200 but log error to avoid infinite retries if issue persists
                     return {"message": "Webhook received but processing failed internally."}
            else:
                 print(f"Ignoring {event_name} webhook - missing or invalid order data.")

        # --- Handle other events if needed (e.g., refunds) ---
        # elif event_name == "order_refunded":
        #     # Find original order, update status?
        #     pass

        else:
            print(f"Ignoring irrelevant webhook event: {event_name}")

        return {"message": "Webhook received"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Generic error processing webhook: {e}")
        # Return 200 OK even on processing error so Lemon Squeezy doesn't retry indefinitely,
        # but log the error thoroughly.
        return {"message": "Webhook received but processing failed internally."}