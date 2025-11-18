from fastapi import APIRouter, Depends
from pydantic import BaseModel
from .schemas import KakaoLoginResponse
from .service import KakaoService


class KakaoCode(BaseModel):
    code: str

router = APIRouter(prefix="/kakao", tags=["kakao"])


@router.post(
    "/login",
    response_model=KakaoLoginResponse,
    description="카카오 로그인",
)
async def kakao_login(
    kakao_code: KakaoCode, 
    kakao_service: KakaoService = Depends()
):
    app_token, is_new_user, user = await kakao_service.login(kakao_code.code)

    response_data = {
        "app_token": app_token,
        "is_new_user": is_new_user,
        "id": user.id,
        "nickname": user.nickname,
        "birth_year": user.birth_year,
        "gender": user.gender,
        "disease_info": user.disease_info,
        "phone": user.phone,
        "email": user.email,
    }
    
    return KakaoLoginResponse(**response_data)
