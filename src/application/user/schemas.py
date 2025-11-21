from pydantic import BaseModel, Field
from typing import Optional

class UserInfoUpdateRequest(BaseModel):
    nickname: Optional[str] = Field(None, description="사용자 닉네임")
    birth_year: str = Field(..., description="출생년도")
    gender: str = Field(..., description="성별")
    disease_info: Optional[str] = Field(None, description="기저질환 정보")
    email: Optional[str] = Field(None, description="사용자 이메일")
    phone: Optional[str] = Field(None, description="전화번호")

class UserInfoResponse(BaseModel):
    id: int
    nickname: Optional[str]
    birth_year: Optional[str]
    gender: Optional[str]
    disease_info: Optional[str]
    email: Optional[str]
    phone: Optional[str]

    class Config:
        from_attributes = True
