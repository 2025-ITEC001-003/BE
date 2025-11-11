from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from src.core import OPENAI_API_KEY
from src.tools import all_tools

# 메인 LLM 초기화
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)

# 메인 Agent용 프롬프트
VALID_LOCATIONS_LIST = "['제주시', '서귀포시', '애월읍', '한림읍', '성산읍', '안덕면', '조천읍', '구좌읍']"
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "당신은 'JeSafe' 제주 관광 안전 챗봇입니다."
        "날씨, 안전 사고, 관광 정보 질문에 답할 수 있습니다."
        "필요한 도구를 사용하여 정확한 정보를 찾습니다."
        
        "### 도구 사용 규칙 (매우 중요) ###"
        
        "1. **날씨 도구 사용 규칙:**"
        "   - '오늘' 또는 '현재' 날씨 질문에는 **`today_weather_tool`**을 사용합니다."
        "   - '내일' 또는 '특정 날짜'의 예보 질문에는 **`future_weather_tool`**을 사용합니다."
        "   - `location` 인자는 **반드시** 아래 목록에 있는 유효한 이름 중 하나여야 합니다:"
        f"     - 유효한 위치 목록: {VALID_LOCATIONS_LIST}"
        "   - 만약 사용자가 '제주도'처럼 모호한 위치를 질문하면, **절대 '제주도'를 `location` 인자로 사용하지 말고, '제주시'를 기본값으로 사용해야 합니다.**"
        "   - 만약 사용자가 목록에 없는 지역(예: '한라산')을 질문하면, '제주시'의 날씨를 대신 알려주거나 가장 가까운 유효 위치(예: '서귀포시')의 날씨를 알려준다고 명시해야 합니다."
        "   - `location` 인자로 `KeyError`나 `ValueError`(좌표를 찾을 수 없음) 오류가 발생하면, 이는 당신이 유효하지 않은 `location` 이름을 사용했기 때문입니다. 즉시 '제주시' 또는 다른 유효한 위치로 다시 시도해야 합니다."
        
        "2. **안전 사고 통계 질문** (예: '낙상 사고 몇 건이야?', '겨울철 사고 유형'):"
        "   - **`safety_sql_tool`** 도구를 사용합니다."
        
        "3. **관광 정보 질문** (예: '오름 추천해줘', '제주 향토음식 뭐 있어?', '한류 촬영지 알려줘'):"
        "   - **`tourism_rag_tool`** 도구를 사용합니다."

        "4. 일반 대화나 인사에는 도구를 사용하지 마세요."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# router agent
router_agent = create_tool_calling_agent(
    llm=llm,
    tools=all_tools,
    prompt=prompt
)

# 메인 Agent Executor
main_agent_executor = AgentExecutor(
    agent=router_agent,
    tools=all_tools,
    verbose=True
)

# 대화 기록(Memory) 관리
# (세션 ID별로 대화 기록을 저장할 휘발성 인메모리 저장소)
chat_history_store = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """세션 ID를 기반으로 메모리에서 ChatMessageHistory 객체를 가져옵니다."""
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    return chat_history_store[session_id]

# Agent와 기록 관리자 래핑(Wrapping)
agent_with_history = RunnableWithMessageHistory(
    main_agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 비즈니스 로직 함수 (API가 호출)
async def get_agent_response(query: str, session_id: str) -> str:
    """사용자 질문과 세션ID를 받아 Agent를 실행하고 답변을 반환"""
    print(f"서비스: [세션: {session_id}] \"{query}\" 질문 처리 시작")

    # 해당 세션의 대화 기록 가져오기
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    chat_history = chat_history_store[session_id]

    try:
        # 메인 Agent 실행
        response = await main_agent_executor.ainvoke({
            "input": query,
            "chat_history": chat_history.messages
        })
        
        answer = response.get("output", "답변 생성 실패")

        # 대화 기록에 현재 질문과 답변 추가
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
        
    except Exception as e:
        print(f"[Main Agent 오류] {e}")
        answer = "죄송합니다. 내부 오류가 발생했습니다."

    return answer

async def stream_agent_response(query: str, session_id: str):
    """
    Agent의 응답을 비동기 생성기(async generator)로 스트리밍합니다.
    """
    print(f"스트리밍 서비스: [세션: {session_id}] \"{query}\" 처리 시작")

    # 대화 기록 가져오기
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    chat_history = chat_history_store[session_id]

    # 스트리밍 실행
    full_answer = ""
    try:
        # main_agent_executor.ainvoke 대신 .astream_events 사용
        async for event in agent_with_history.astream_events(
            {"input": query},
            # ⬇️ 'session_id'는 config로 전달
            config={"configurable": {"session_id": session_id}},
            version="v1" # 이벤트 스트림 버전 v1 사용
        ):
            kind = event["event"]

            # LLM이 생성하는 응답 스트림(토큰)만 필터링
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                # AIMessageChunk의 content만 추출
                if isinstance(chunk, AIMessage) and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        yield content # 청크를 즉시 반환(yield)
            
    except Exception as e:
        print(f"[Streaming Agent 오류] {e}")
        error_message = "죄송합니다. 내부 오류가 발생했습니다."
        yield error_message
