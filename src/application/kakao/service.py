import httpx
import jwt
import json
from datetime import datetime, timedelta
from typing import Tuple
from fastapi import HTTPException, Depends
from sqlalchemy.orm import Session

from src.config import KAKAO_REST_API_KEY, KAKAO_REDIRECT_URI, JWT_SECRET_KEY, JWT_ALGORITHM
from src.database import get_db
from src.models import User


class KakaoService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db
        self.kakao_rest_api_key = KAKAO_REST_API_KEY
        self.kakao_redirect_uri = KAKAO_REDIRECT_URI
        self.kakao_token_uri = "https://kauth.kakao.com/oauth/token"
        self.kakao_user_info_uri = "https://kapi.kakao.com/v2/user/me"

    async def login(self, code: str) -> Tuple[str, bool, str | None]:
        access_token = await self._get_kakao_access_token(code)
        user_info = await self._get_kakao_user_info(access_token)

        kakao_id = user_info["id"]
        nickname = user_info.get("properties", {}).get("nickname")

        is_new_user = False
        user = self.db.query(User).filter(User.kakao_id == kakao_id).first()
        if not user:
            is_new_user = True
            user = User(kakao_id=kakao_id, nickname=nickname)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        elif user.nickname != nickname:
            user.nickname = nickname
            self.db.commit()

        payload = {
            "sub": str(user.id),
            "exp": datetime.utcnow() + timedelta(days=30),
            "iat": datetime.utcnow(),
            "provider": "kakao",
            "nickname": nickname,
        }

        app_token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return app_token, is_new_user, nickname

    async def _get_kakao_access_token(self, code: str) -> str:
        headers = {"Content-type": "application/x-www-form-urlencoded;charset=utf-8"}
        data = {
            "grant_type": "authorization_code",
            "client_id": self.kakao_rest_api_key,
            "redirect_uri": self.kakao_redirect_uri,
            "code": code,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.kakao_token_uri, headers=headers, data=data)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            return response.json()["access_token"]

    async def _get_kakao_user_info(self, access_token: str) -> dict:
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8",
        }
        property_keys = ["properties.nickname"]
        params = {"property_keys": json.dumps(property_keys)}
        async with httpx.AsyncClient() as client:
            response = await client.get(self.kakao_user_info_uri, headers=headers, params=params)
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.json())
            return response.json()
