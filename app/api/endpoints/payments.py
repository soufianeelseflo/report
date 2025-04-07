# autonomous_agency/app/api/endpoints/payments.py
import hmac
import hashlib
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
    report_request_id: int
    report_type: str # 'standard_499' or 'premium_999'
    client_email: str
    client_name: Optional[str] = None

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
    # Use a new client per request or manage a shared one carefully
    return httpx.AsyncClient(base_url=LEMONSQUEEZY_API_URL, headers=headers, timeout=20.0)

@router.post("/create-checkout", response_model=CreateCheckoutResponse, status_code=status.HTTP_201_CREATED)
async def create_lemon_squeezy_checkout(
    payload: CreateCheckoutRequest,
    db: AsyncSession = Depends(get_db_session),
    ls_client: httpx.AsyncClient = Depends(get_ls_client)
):
    """Creates a Lemon Squeezy checkout session for a report request."""
    if not settings.LEMONSQUEEZY_STORE_ID or not settings.LEMONSQUEEZY_VARIANT_STANDARD or not settings.LEMONSQUEEZY_VARIANT_PREMIUM:
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy store/variant IDs not configured.")

    # 1. Get the report request to confirm details (optional but good practice)
    report_request = await db.get(models.ReportRequest, payload.report_request_id)
    if not report_request:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report request not found.")
    if report_request.status != 'PENDING': # Prevent re-payment for processed requests
         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Report request already processed or paid.")

    # 2. Determine variant ID based on report type
    variant_id = None
    if payload.report_type == "standard_499":
        variant_id = settings.LEMONSQUEEZY_VARIANT_STANDARD
    elif payload.report_type == "premium_999":
        variant_id = settings.LEMONSQUEEZY_VARIANT_PREMIUM
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid report type for checkout.")

    # 3. Prepare checkout data for Lemon Squeezy API
    checkout_data = {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_options": {
                    "embed": False,
                    # "media": False, # Optional: disable media?
                    # "logo": False, # Optional: disable logo?
                },
                "checkout_data": {
                    "email": payload.client_email,
                    "name": payload.client_name,
                    "custom": {
                        # Store our internal request ID to link back on webhook
                        "report_request_id": str(payload.report_request_id),
                    },
                },
                # Redirect URLs (Use your actual deployed URLs)
                "redirect_url": f"{settings.AGENCY_BASE_URL}/order/success?session_id={{CHECKOUT_SESSION_ID}}", # Placeholder success page
            },
            "relationships": {
                "store": {"data": {"type": "stores", "id": settings.LEMONSQUEEZY_STORE_ID}},
                "variant": {"data": {"type": "variants", "id": variant_id}},
            },
        }
    }

    # 4. Call Lemon Squeezy API
    try:
        async with ls_client:
            response = await ls_client.post("/checkouts", json=checkout_data)
            response.raise_for_status() # Raise exception for 4xx/5xx errors
            ls_response_data = response.json()
            checkout_url = ls_response_data.get("data", {}).get("attributes", {}).get("url")

            if not checkout_url:
                 raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Lemon Squeezy did not return a checkout URL.")

            # Optionally: Update report_request status to 'AWAITING_PAYMENT' ?
            # await crud.update_report_request_status(db, payload.report_request_id, status="AWAITING_PAYMENT")
            # await db.commit() # Commit status change if needed

            return CreateCheckoutResponse(checkout_url=checkout_url)

    except httpx.HTTPStatusError as e:
        print(f"Lemon Squeezy API Error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Payment provider error: {e.response.json().get('errors', [{}])[0].get('detail', 'Unknown error')}")
    except Exception as e:
        print(f"Error creating Lemon Squeezy checkout: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not initiate payment process.")


# --- Webhook Handler ---
@router.post("/webhook", status_code=status.HTTP_200_OK)
async def lemon_squeezy_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db_session)
):
    """Handles incoming webhooks from Lemon Squeezy."""
    if not settings.LEMONSQUEEZY_WEBHOOK_SECRET:
        print("WARNING: Lemon Squeezy Webhook secret not set. Cannot verify signature.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Webhook secret not configured.")

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
        event_data = await request.json()
        event_name = event_data.get("meta", {}).get("event_name")
        custom_data = event_data.get("meta", {}).get("custom_data", {})
        report_request_id_str = custom_data.get("report_request_id")

        print(f"Received Lemon Squeezy webhook: {event_name} for request ID: {report_request_id_str}")

        if event_name == "order_created" or event_name == "subscription_created": # Handle one-time or subscription payments
            if report_request_id_str:
                try:
                    report_request_id = int(report_request_id_str)
                    # --- CRITICAL: Update report status to PAID/PENDING ---
                    # Find the corresponding report request
                    report_request = await db.get(models.ReportRequest, report_request_id)
                    if report_request and report_request.status != 'COMPLETED' and report_request.status != 'PROCESSING':
                        # Update status to PENDING to trigger the ReportGenerator
                        await crud.update_report_request_status(db, report_request_id, status="PENDING")
                        # Optionally store payment info (order ID, amount) if needed
                        # e.g., add columns to ReportRequest or a separate Payments table
                        # report_request.payment_status = "PAID"
                        # report_request.payment_order_id = event_data.get("data",{}).get("id")
                        await db.commit()
                        print(f"Payment confirmed for Report Request ID {report_request_id}. Status set to PENDING.")
                    elif report_request:
                         print(f"Payment webhook received for already processed/completed Report Request ID {report_request_id}. Ignoring status update.")
                    else:
                         print(f"Payment webhook received but Report Request ID {report_request_id} not found.")

                except ValueError:
                    print(f"Invalid report_request_id in webhook custom data: {report_request_id_str}")
                except Exception as e:
                     print(f"Error processing payment confirmation for request {report_request_id_str}: {e}")
                     await db.rollback() # Rollback if DB update fails
                     # Potentially raise 500 to signal retry to Lemon Squeezy? Check their docs.
            else:
                print("Webhook received without report_request_id in custom_data.")
        else:
            print(f"Ignoring irrelevant webhook event: {event_name}")

        return {"message": "Webhook received"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload.")
    except Exception as e:
        print(f"Error processing webhook: {e}")
        # Return 200 OK even on processing error so Lemon Squeezy doesn't retry indefinitely,
        # but log the error thoroughly.
        return {"message": "Webhook received but processing failed internally."}