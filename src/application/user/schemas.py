from pydantic import BaseModel, Field
from typing import Optional

class UserInfoUpdateRequest(BaseModel):
    birth_year: str = Field(..., description="출생년도")
    gender: str = Field(..., description="성별")
    disease_info: Optional[str] = Field(None, description="기저질환 정보")

class UserInfoResponse(BaseModel):
    id: int
    nickname: Optional[str]
    birth_year: Optional[str]
    gender: Optional[str]
    disease_info: Optional[str]

    class Config:
        from_attributes = True
