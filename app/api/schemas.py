from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class CreateCheckoutRequest(BaseModel): # Renamed from ReportRequestCreate for clarity
    client_name: Optional[str] = Field(None, max_length=255)
    client_email: EmailStr
    company_name: Optional[str] = Field(None, max_length=255)
    report_type: str = Field(..., description="Type of report requested (e.g., 'standard_499', 'premium_999')")
    request_details: str = Field(..., description="Specific details or topic for the report")

class CreateCheckoutResponse(BaseModel): # Renamed for clarity
    checkout_url: str

class ReportRequestResponse(BaseModel): # Kept for potential future use
    request_id: int
    status: str
    message: str

    class Config:
        # orm_mode = True # Pydantic V1 compatibility
        from_attributes = True # Pydantic V2 equivalent

class HealthCheck(BaseModel):
    name: str
    status: str
    version: str # Placeholder for versioning