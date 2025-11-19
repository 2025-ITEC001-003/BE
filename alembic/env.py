import os
import sys
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# --- 설정 시작 ---

# .env 파일 로드 (프로젝트 루트에 있는 .env 파일을 찾음)
# 이렇게 하면 alembic.ini에 민감한 정보를 저장할 필요가 없습니다.
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Alembic Config 객체, .ini 파일의 값에 접근 가능
config = context.config

# .env 파일에서 읽어온 환경 변수로 DB URL을 동적으로 설정
# 이 방식은 보안적으로 더 안전합니다.
db_url = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', 5432)}"
    f"/{os.getenv('POSTGRES_DB')}"
)
config.set_main_option('sqlalchemy.url', db_url)


# Python 로깅 설정
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 기능을 위해 SQLAlchemy 모델을 인식하도록 경로와 메타데이터 설정
# 1. 프로젝트의 src 폴더를 시스템 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. src/database.py의 Base와 src/models.py의 모델들을 import
from src.database import Base
from src.models import User  # User 모델을 명시적으로 import

# 3. target_metadata를 프로젝트의 Base.metadata로 설정
target_metadata = Base.metadata

# --- 설정 끝 ---


def run_migrations_offline() -> None:
    """'offline' 모드에서 마이그레이션을 실행합니다."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """'online' 모드에서 마이그레이션을 실행합니다."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()