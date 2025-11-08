import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase

# .env 파일 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# OpenAI API 키를 환경변수에 명시적으로 설정 (LangChain이 사용)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY

# DB 엔진 및 LangChain DB 인스턴스
engine = create_engine(DATABASE_URL)
db_langchain = SQLDatabase(engine=engine, include_tables=['jeju_accidents'])