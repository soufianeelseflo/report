# autonomous_agency/app/api/endpoints/payments.py
import hmac
import hashlib
import json
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
import httpx
from typing import Optional

# Corrected relative imports for package structure
from autonomous_agency.app.core.config import settings
from autonomous_agency.app.db import crud, models
from autonomous_agency.app.db.base import get_db_session
from pydantic import BaseModel, Field

router = APIRouter()

LEMONSQUEEZY_API_URL = "https://api.lemonsqueezy.com/v1"

class CreateCheckoutRequest(BaseModel):
    report_type: str = Field(..., description="Type of report: 'standard_499' or 'premium_999'")
    client_email: str
    client_name: Optional[str] = None
    company_name: Optional[str] = None
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
):
    """Creates a Lemon Squeezy checkout session."""
    if not settings.LEMONSQUEEZY_STORE_ID or not settings.LEMONSQUEEZY_VARIANT_STANDARD or not settings.LEMONSQUEEZY_VARIANT_PREMIUM:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy store/variant IDs not configured.")
    if not settings.AGENCY_BASE_URL or "localhost" in settings.AGENCY_BASE_URL:
         print("WARNING: AGENCY_BASE_URL is not set or is localhost. Redirects/Webhooks might fail in production.")
         # Allow proceeding for local testing, but production needs the real URL.

    variant_id = None
    if payload.report_type == "standard_499":
        variant_id = settings.LEMONSQUEEZY_VARIANT_STANDARD
    elif payload.report_type == "premium_999":
        variant_id = settings.LEMONSQUEEZY_VARIANT_PREMIUM
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid report type for checkout.")

    custom_data_payload = {
        # Use keys expected by webhook/crud function
        "research_topic": payload.request_details,
        "company_name": payload.company_name,
        # Pass client name/email again in custom data for redundancy? Optional.
        # "client_name": payload.client_name,
        # "client_email": payload.client_email,
    }

    checkout_data = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_options": {"embed": False},
                "checkout_data": {
                    "email": payload.client_email,
                    "name": payload.client_name,
                    "custom": custom_data_payload,
                },
                # Use AGENCY_BASE_URL from settings for redirects
                "redirect_url": f"{settings.AGENCY_BASE_URL}/order-success?session_id={{CHECKOUT_SESSION_ID}}", # Define this success page later
                # Add product specific redirects if needed
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": settings.LEMONSQUEEZY_STORE_ID}},
                "variant": {"data": {"type": "variants", "id": variant_id}},
            },
        }
    }

    try:
        async with ls_client:
            response = await ls_client.post("/checkouts", json=checkout_data)
            response.raise_for_status()
            ls_response_data = response.json()
            checkout_url = ls_response_data.get("data", {}).get("attributes", {}).get("url")
            checkout_id = ls_response_data.get("data", {}).get("id")

            if not checkout_url:
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy did not return a checkout URL.")

            print(f"Created Lemon Squeezy checkout {checkout_id} for {payload.client_email}")
            return CreateCheckoutResponse(checkout_url=checkout_url)

    except httpx.HTTPStatusError as e:
        error_detail = "Unknown payment provider error"
        try: error_detail = e.response.json().get('errors', [{}])[0].get('detail', 'Unknown error')
        except Exception: pass
        print(f"Lemon Squeezy API Error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Payment provider error: {error_detail}")
    except Exception as e:
        print(f"Error creating Lemon Squeezy checkout: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not initiate payment process.")


# --- Webhook Handler ---
@router.post("/webhook", status_code=status.HTTP_200_OK, include_in_schema=False)
async def lemon_squeezy_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session)
):
    """Handles incoming webhooks from Lemon Squeezy (e.g., order_created)."""
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
        print("CRITICAL: Lemon Squeezy Webhook secret not set. Cannot process webhooks.")
        return {"message": "Webhook ignored, secret not configured."}

    raw_body = await request.body()
    try:
        computed_signature = hmac.new(settings.LEMONSQUEEZY_WEBHOOK_SECRET.encode('utf-8'), raw_body, hashlib.sha256).hexdigest()
        if not x_signature or not hmac.compare_digest(computed_signature, x_signature):
            print("Webhook signature mismatch!")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature.")
    except Exception as e:
         print(f"Error verifying webhook signature: {e}")
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Signature verification failed.")

    try:
        event_data = json.loads(raw_body)
        meta = event_data.get("meta", {})
        event_name = meta.get("event_name")
        order_data = event_data.get("data")

        print(f"Received Lemon Squeezy webhook: {event_name}")

        if event_name == "order_created":
            if order_data and order_data.get("type") == "orders":
                try:
                    # Create report request using data from webhook payload
                    report_request = await crud.create_report_request_from_webhook(db, order_data)
                    if report_request:
                        await db.commit()
                        print(f"Successfully processed order {order_data.get('id')} and created ReportRequest {report_request.request_id}")
                    else:
                        await db.rollback() # Skipped (e.g., duplicate)
                except Exception as e:
                     print(f"Error processing 'order_created' webhook for order {order_data.get('id')}: {e}")
                     await db.rollback()
                     return {"message": "Webhook received but processing failed internally."} # Return 200 OK but log error
            else:
                 print(f"Ignoring {event_name} webhook - missing or invalid order data.")
        else:
            print(f"Ignoring irrelevant webhook event: {event_name}")

        return {"message": "Webhook received"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Generic error processing webhook: {e}")
        return {"message": "Webhook received but processing failed internally."}