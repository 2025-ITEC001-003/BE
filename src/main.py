from fastapi import FastAPI
import uvicorn
from src.schemas import ChatRequest, ChatResponse
from src.services import get_agent_response
from fastapi.responses import StreamingResponse
import json
from src.services import stream_agent_response
from src.application.kakao import router as kakao_router
from src.application.user import router as user_router # user 라우터 import

# DB 초기화 관련
from src.database import engine, Base
import src.models

# 애플리케이션 시작 시 DB 테이블 생성
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="JeSafe 챗봇 API",
    description="제주 관광 안전 사고 통계 챗봇",
)

# AI 관련 API
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇 질의응답 엔드포인트"""
    answer = await get_agent_response(request.query, request.session_id)
    return ChatResponse(answer=answer)

@app.post("/chat-stream")
async def stream_chat(request: ChatRequest):
    """
    챗봇 질의응답 스트리밍 엔드포인트 (SSE)
    """
    generator = sse_format_generator(request.query, request.session_id)
    return StreamingResponse(generator, media_type="text/event-stream")

async def sse_format_generator(query: str, session_id: str):
    """
    서비스 로직(stream_agent_response)을 감싸서
    SSE (Server-Sent Events) 형식으로 변환하는 생성기
    """
    try:
        async for chunk in stream_agent_response(query, session_id):
            data_payload = {"chunk": chunk}
            sse_data = f"data: {json.dumps(data_payload, ensure_ascii=False)}\n\n"

            yield sse_data

    except Exception as e:
        print(f"[SSE 생성기 오류] {e}")
        error_payload = {"error": str(e)}
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

@app.get("/")
def read_root():
    return {"message": "Welcome to JeSafe API"}

app.include_router(kakao_router.router, prefix="/api")
app.include_router(user_router.router, prefix="/api")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
