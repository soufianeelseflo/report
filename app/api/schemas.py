from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class ReportRequestCreate(BaseModel):
    client_name: Optional[str] = Field(None, max_length=255)
    client_email: EmailStr
    company_name: Optional[str] = Field(None, max_length=255)
    report_type: str = Field(..., description="Type of report requested (e.g., 'standard_499', 'premium_999')")
    request_details: str = Field(..., description="Specific details or topic for the report")

class ReportRequestResponse(BaseModel):
    request_id: int
    status: str
    message: str

    class Config:
        orm_mode = True # Pydantic V1 compatibility, use from_attributes=True in V2 if needed directly

class HealthCheck(BaseModel):
    name: str
    status: str
    version: str # Placeholder for versioning