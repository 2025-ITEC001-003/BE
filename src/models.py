from sqlalchemy import Column, Integer, String, BigInteger
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    kakao_id = Column(BigInteger, unique=True, index=True, nullable=False)
    nickname = Column(String, nullable=True)
    birth_year = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    disease_info = Column(String, nullable=True)
