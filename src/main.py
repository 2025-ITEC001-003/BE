from fastapi import FastAPI
import uvicorn
from src.schemas import ChatRequest, ChatResponse
from src.services import get_agent_response

app = FastAPI(
    title="JeSafe 챗봇 API",
    description="제주 관광 안전 사고 통계 챗봇",
)

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """챗봇 질의응답 엔드포인트"""
    answer = await get_agent_response(request.query, request.session_id)
    return ChatResponse(answer=answer)

@app.get("/")
def read_root():
    return {"message": "Welcome to JeSafe API"}

# (선택) 개발용 uvicorn 직접 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)