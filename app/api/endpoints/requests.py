from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import schemas
from app.db import crud
from app.db.base import get_db_session

router = APIRouter()

@router.post("/", status_code=status.HTTP_202_ACCEPTED, response_model=schemas.ReportRequestResponse)
async def create_report_request_endpoint(
    request: schemas.ReportRequestCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Accepts a new report request and queues it for processing.
    """
    try:
        db_request = await crud.create_report_request(db=db, request=request)
        # Note: We commit in the get_db_session dependency handler
        return schemas.ReportRequestResponse(
            request_id=db_request.request_id,
            status=db_request.status,
            message="Report request accepted and queued for processing."
        )
    except Exception as e:
        # Log the exception e
        print(f"Error creating report request: {e}") # Replace with proper logging
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your request."
        )