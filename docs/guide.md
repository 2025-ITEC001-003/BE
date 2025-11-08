# _제주관광안전서비스 (JeSafe) 실행 가이드_

본 문서는 `jesafe` 프로젝트의 데이터베이스와 API 서버를 실행하는 방법을 안내합니다.

## 1. ⚙️ 사전 준비

프로젝트 실행 전, 로컬 환경에 다음이 준비되어 있어야 합니다.

1.  **Docker Desktop**이 설치 및 실행 중이어야 합니다.
2.  `poetry`가 설치되어 있어야 합니다.
3.  프로젝트 루트(`jesafe/`)에 다음 파일들이 존재해야 합니다:
    * `docker-compose.yml`: PostgreSQL 컨테이너 설정 파일
    * `.env`: API 키 및 데이터베이스 접속 정보 파일
    * `data/kma_grid_jeju.csv`: 제주도 지역 좌표 데이터 파일
4.  터미널에서 `poetry`를 사용해 필요한 모든 Python 의존성을 설치합니다.

    ```bash
    poetry install
    ```

---

## 2. 🚀 프로젝트 실행

### 1단계: PostgreSQL 데이터베이스 실행

터미널에서 `docker-compose.yml`이 있는 프로젝트 루트로 이동한 후, 아래 명령어를 입력하여 Docker 컨테이너를 백그라운드에서 실행합니다.

```bash
docker-compose up -d
```

DB 중지 시: Docker 컨테이너를 정지하고 제거합니다.

```Bash
docker-compose down
```
### 2단계: FastAPI 애플리케이션 실행 (Uvicorn)
데이터베이스가 실행 중인 상태에서, poetry 가상 환경을 사용해 Uvicorn 서버를 실행합니다.

```Bash
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
- src.main:app: src/main.py 파일의 app 객체를 실행합니다.
- --reload: 코드 변경 시 서버를 자동으로 재시작합니다. (개발용)
- 서버 종료 시: uvicorn이 실행 중인 터미널에서 Ctrl + C를 누릅니다.
---
## 3. ✅ 실행 확인
서버가 정상적으로 실행되면, 웹 브라우저에서 다음 주소로 접속하여 확인할 수 있습니다.

- API 환영 메시지: http://localhost:8000
- API 자동 문서 (Swagger UI): http://localhost:8000/docs