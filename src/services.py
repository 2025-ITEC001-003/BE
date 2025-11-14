import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent

from src.core import OPENAI_API_KEY, llm
from src.tools import all_tools

# 메인 Agent용 프롬프트
VALID_LOCATIONS_LIST = "['제주시', '서귀포시', '애월읍', '한림읍', '성산읍', '안덕면', '조천읍', '구좌읍']"
prompt = ChatPromptTemplate.from_messages([
    ("system", (
    "당신은 'JeSafe' 제주 관광 안전 챗봇입니다. "
    "날씨, 안전 사고, 관광 정보 질문에 답할 수 있습니다. "
    "필요한 도구를 사용하여 정확한 정보를 찾습니다.\n\n"

    "### 도구별 사용 설명서 ###\n\n"

    "#### 1. 오늘 날씨 (`today_weather_tool`)\n"
    " - 목적: '오늘' 또는 '현재' 날씨 질문 처리\n"
    " - 매개변수: `{{ 'location': <string> }}`\n"
    " - 유효한 제주도 위치 목록: {VALID_LOCATIONS_LIST}\n"
    " - 예시 호출:\n"
    "   ```json\n"
    "   {{\"location\": \"애월읍\"}}\n"
    "   ```\n\n"

    "#### 2. 미래 날씨 (`future_weather_tool`)\n"
    " - 목적: '내일', '3일 뒤', '다음 주' 등 미래 날짜 날씨 질문 처리\n"
    " - 매개변수: `{{ 'location': <string>, 'date': <YYYY-MM-DD> }}`\n"
    " - **[중요]** 이 도구는 **반드시 `location`과 `date` 2개 인자가 모두 필요합니다.**\n"
    " - **[필수]** 사용자가 '거기', '그럼 내일은?'처럼 위치를 생략하면, **반드시 이전 대화 기록에서 `location`을 찾아 함께 전달해야 합니다.**\n"  # <-- 핵심 규칙 추가
    " - 날짜 계산 규칙:\n"
    "   - 오늘 날짜 (계산 기준): {today}\n"
    " - 예시 호출:\n"
    "   ```json\n"
    "   {{\"location\": \"애월읍\", \"date\": \"2025-11-15\"}}\n"
    "   ```\n\n"

    "#### 3. 안전 사고 통계 (`safety_sql_tool`)\n"
    " - 목적: 사고 건수, 유형, 시기 등 안전 관련 질문 처리\n"
    " - 매개변수: `{{ 'query': <string> }}`\n"
    " - 예시 호출:\n"
    "   ```json\n"
    "   {{\"query\": \"애월읍 최근 낙상 사고 통계\"}}\n"
    "   ```\n\n"

    "#### 4. 관광 정보 (`tourism_rag_tool`)\n"
    " - 목적: 관광지, 음식, 축제, 오름 등 관광 관련 질문 처리\n"
    " - 매개변수: `{{ 'query': <string> }}`\n"
    " - 예시 호출:\n"
    "   ```json\n"
    "   {{\"query\": \"애월읍 오름 추천\"}}\n"
    "   ```\n\n"

    "### 매개변수 기본 규칙 ###\n"
    "- location은 반드시 위의 유효한 위치 목록 중 하나여야 합니다.\n"
    "- '제주도' 또는 '한라산' 같이 모호한 표현은 자동으로 가장 가까운 유효 지역으로 대체하세요.\n"
    "- 복합 질문(예: '3일 뒤 애월 날씨와 조심할 사고')은 한 번에 하나의 도구만 호출하고, 결과를 활용해 다음 도구를 순차적으로 호출하세요.\n"
    "- **[최우선 원칙]** 도구 호출에 필요한 인자(특히 `location` 또는 `date`)가 사용자 질문이나 이전 대화 기록 어디에도 명확히 없다면, **절대 도구를 호출하지 말고, 사용자에게 해당 정보를 먼저 질문하세요.** (예: '어느 지역의 날씨가 궁금하신가요?')\n"
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

def get_today():
    return datetime.datetime.now().strftime("%Y-%m-%d")

prompt = prompt.partial(
    VALID_LOCATIONS_LIST=VALID_LOCATIONS_LIST,
    today=get_today
)

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

def get_session_history(session_id: str) -> ConversationSummaryBufferMemory:
    if session_id not in chat_history_store:
        chat_history_store[session_id] = ConversationSummaryBufferMemory(
            llm=llm, # 요약에 사용할 LLM
            max_token_limit=1000, # 요약 임계값
            return_messages=True, # 메시지 객체 리스트로 반환
            memory_key="chat_history",
            input_key="input"
        )
    return chat_history_store[session_id]

# Agent와 기록 관리자 래핑(Wrapping)

# 비즈니스 로직 함수 (API가 호출)
async def get_agent_response(query: str, session_id: str) -> str:
    """사용자 질문과 세션ID를 받아 Agent를 실행하고 답변을 반환"""
    print(f"서비스: [세션: {session_id}] \"{query}\" 질문 처리 시작")

    memory = get_session_history(session_id)

    # 현재 메모리에서 대화 기록 로드
    # (이 과정에서 필요시 '자동 요약'이 발생함)
    # .load_memory_variables()는 동기 함수이므로 await 제거
    memory_variables = memory.load_memory_variables({})
    recent_messages = memory_variables.get("chat_history", []) # memory_key="chat_history"

    answer = ""

    try:
        # 메인 Agent 실행
        response = await main_agent_executor.ainvoke(
            {
                "input": query,
                "chat_history": recent_messages
            },
            config={"configurable": {"session_id": session_id}} 
        )
        
        answer = response.get("output", "답변 생성 실패")
        
    except Exception as e:
        print(f"[Main Agent 오류] {e}")
        answer = "죄송합니다. 내부 오류가 발생했습니다."

    finally:
        # 대화 기록 수동 저장
        # (이 과정에서 필요시 '자동 요약'이 발생함)
        if answer:
            # .save_context()는 동기 함수이므로 await 제거
            memory.save_context(
                {"input": query}, 
                {"output": answer}
            )

    return answer

async def stream_agent_response(query: str, session_id: str):
    """
    Agent의 최종 응답만 스트리밍 (내부 추론 과정 제외)
    """
    print(f"스트리밍 서비스: [세션: {session_id}] \"{query}\" 처리 시작")

    memory = get_session_history(session_id)

    # 현재 메모리에서 대화 기록 로드
    # (이 과정에서 필요시 '자동 요약'이 발생함)
    # .load_memory_variables()는 동기 함수이므로 await 제거
    memory_variables = memory.load_memory_variables({})
    recent_messages = memory_variables.get("chat_history", []) # memory_key="chat_history"

    full_answer = ""

    # 추론 과정 필터링을 위한 플래그
    is_final_response = False
    tool_depth = 0  # 중첩된 도구 호출 깊이 추적
    
    try:
        async for event in main_agent_executor.astream_events(
            {
                "input": query,
                "chat_history": recent_messages
            },
            config={"configurable": {"session_id": session_id}},
            version="v1" 
        ):
            kind = event["event"]
            
            # 1. 도구 호출 깊이 추적 (중첩된 SQL Agent 등 감지)
            if kind == "on_tool_start":
                tool_depth += 1
            elif kind == "on_tool_end":
                tool_depth -= 1
            
            # 2. AgentExecutor가 최종 응답 생성을 시작하는 시점 감지
            if kind == "on_chain_start":
                name = event.get("name", "")
                # AgentExecutor의 최종 LLM 호출 시작점
                if name == "AgentExecutor" and tool_depth == 0:
                    is_final_response = True
            
            # 3. LLM 토큰 필터링
            if kind == "on_chat_model_stream":
                # 도구 실행 중이 아니고, 최종 응답 단계일 때만 전송
                if tool_depth == 0:
                    chunk = event["data"].get("chunk")
                    if isinstance(chunk, AIMessage) and chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            # Agent의 내부 사고 과정 제외
                            # (Thought:, Action:, Observation: 등이 포함된 청크 필터링)
                            if not any(keyword in content for keyword in 
                                      ["Thought:", "Action:", "Action Input:", 
                                       "Observation:", "Final Answer:"]):
                                full_answer += content
                                yield content

    except Exception as e:
        print(f"[Streaming Agent 오류] {e}")
        full_answer = "죄송합니다. 내부 오류가 발생했습니다."
        yield full_answer
    
    finally:
        # 대화 기록 수동 저장
        # (이 과정에서 필요시 '자동 요약'이 발생함)
        if full_answer:
            # .save_context()는 동기 함수이므로 await 제거
            memory.save_context(
                {"input": query}, 
                {"output": full_answer}
            )