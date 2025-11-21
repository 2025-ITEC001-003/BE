from sqlalchemy.orm import Session
from src.models import User
from .schemas import UserInfoUpdateRequest

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def update_user_info(self, user_id: int, user_info: UserInfoUpdateRequest) -> User:
        db_user = self.db.query(User).filter(User.id == user_id).first()
        if db_user:
            db_user.birth_year = user_info.birth_year
            db_user.gender = user_info.gender
            db_user.disease_info = user_info.disease_info
            db_user.email = user_info.email
            db_user.phone = user_info.phone
            self.db.commit()
            self.db.refresh(db_user)
        return db_user
