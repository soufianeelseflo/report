# autonomous_agency/app/webhooks/lemonsqueezy.py
import hmac
import hashlib
import json
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from autonomous_agency.app.core.config import settings
from autonomous_agency.app.db import crud, models
from autonomous_agency.app.db.base import get_db_session

router = APIRouter()

@router.post("/lemonsqueezy", status_code=status.HTTP_200_OK, include_in_schema=False) # Hide from OpenAPI docs
async def lemon_squeezy_webhook(
    request: Request,
    x_signature: Optional[str] = Header(None, alias="X-Signature"), # Correct alias for Lemon Squeezy header
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
            print(f"Webhook signature mismatch! Received: {x_signature}, Computed: {computed_signature}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid signature.")
        print("Webhook signature verified successfully.")
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
                     import traceback
                     traceback.print_exc()
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
        import traceback
        traceback.print_exc()
        # Return 200 OK even on processing error so Lemon Squeezy doesn't retry indefinitely,
        # but log the error thoroughly.
        return {"message": "Webhook received but processing failed internally."}