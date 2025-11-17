from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.models import User
from src.application.auth.jwt import get_current_user
from .schemas import UserInfoUpdateRequest, UserInfoResponse
from .service import UserService

router = APIRouter(
    prefix="/user",
    tags=["user"],
)

@router.put("/info", response_model=UserInfoResponse)
def update_user_info(
    user_info: UserInfoUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    인증된 사용자의 추가 정보(생년, 성별, 기저질환)를 업데이트합니다.
    """
    service = UserService(db)
    updated_user = service.update_user_info(current_user.id, user_info)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    return updated_user
