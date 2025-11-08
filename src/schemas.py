from pydantic import BaseModel

class ChatRequest(BaseModel):
    """API Request DTO"""
    query: str
    session_id: str

class ChatResponse(BaseModel):
    """API Response DTO"""
    answer: str