from pydantic import BaseModel, ConfigDict


class KakaoLoginResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    app_token: str
    is_new_user: bool
    id: int
    nickname: str | None
    birth_year: str | None
    gender: str | None
    disease_info: str | None
    phone: str | None
    email: str | None
