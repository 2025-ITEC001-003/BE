from pydantic import BaseModel


class KakaoLoginResponse(BaseModel):
    app_token: str
    is_new_user: bool
    nickname: str | None
