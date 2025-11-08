from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from src.core import OPENAI_API_KEY
from src.tools import all_tools

# 1. 메인 LLM 초기화
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, api_key=OPENAI_API_KEY)

# 2. 메인 Agent용 프롬프트
VALID_LOCATIONS_LIST = "['제주시', '서귀포시', '애월읍', '한림읍', '성산읍', '안덕면', '조천읍', '구좌읍']"

prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "당신은 'JeSafe' 제주 관광 안전 챗봇입니다."
        "날씨 질문이나 안전 사고 통계 질문에 답할 수 있습니다."
        "필요한 경우 도구(tools)를 사용하여 정보를 찾습니다."

        "### 날씨 도구 사용 규칙 (매우 중요) ###"
        "1. 날씨 도구(`get_current_weather`, `get_date_weather_summary`)를 호출할 때, `location` 인자는 **반드시** 아래 목록에 있는 유효한 이름 중 하나여야 합니다."
        f" - 유효한 위치 목록: {VALID_LOCATIONS_LIST}"
        "2. 만약 사용자가 '제주도'처럼 모호한 위치를 질문하면, **절대 '제주도'를 `location` 인자로 사용하지 말고, '제주시'를 기본값으로 사용해야 합니다.**"
        "3. 만약 사용자가 목록에 없는 지역(예: '한라산')을 질문하면, '제주시'의 날씨를 대신 알려주거나 가장 가까운 유효 위치(예: '서귀포시')의 날씨를 알려준다고 명시해야 합니다."
        "4. `location` 인자로 `KeyError`나 `ValueError`(좌표를 찾을 수 없음) 오류가 발생하면, 이는 당신이 유효하지 않은 `location` 이름을 사용했기 때문입니다. 즉시 '제주시' 또는 다른 유효한 위치로 다시 시도해야 합니다."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. router agent
router_agent = create_tool_calling_agent(
    llm=llm,
    tools=all_tools,
    prompt=prompt
)

# 4. 메인 Agent Executor
main_agent_executor = AgentExecutor(
    agent=router_agent,
    tools=all_tools,
    verbose=True
)

# 5. 대화 기록(Memory) 관리
# (세션 ID별로 대화 기록을 저장할 임시 저장소)
chat_history_store = {}

# 6. 비즈니스 로직 함수 (API가 호출)
async def get_agent_response(query: str, session_id: str) -> str:
    """사용자 질문과 세션ID를 받아 Agent를 실행하고 답변을 반환"""
    print(f"서비스: [세션: {session_id}] \"{query}\" 질문 처리 시작")

    # 1. 해당 세션의 대화 기록 가져오기
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ChatMessageHistory()
    
    chat_history = chat_history_store[session_id]

    try:
        # 2. 메인 Agent 실행
        response = await main_agent_executor.ainvoke({
            "input": query,
            "chat_history": chat_history.messages
        })
        
        answer = response.get("output", "답변 생성 실패")

        # 3. 대화 기록에 현재 질문과 답변 추가
        chat_history.add_user_message(query)
        chat_history.add_ai_message(answer)
        
    except Exception as e:
        print(f"[Main Agent 오류] {e}")
        answer = "죄송합니다. 내부 오류가 발생했습니다."

    return answer