import httpx
from langchain.tools import Tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.core import get_db_langchain, OPENWEATHER_API_KEY, llm, get_compression_retriever
from src.data_loader import get_jeju_coordinates

# --- 1. '오늘/현재' 날씨 도구 (Tool 1 - OWM) ---
def get_current_weather(location: str) -> str:
    """
    OpenWeatherMap 'Overview' API를 호출하여 
    특정 위치의 '현재/오늘' 날씨 요약을 반환합니다.
    """
    print(f"[Tool] '오늘 날씨 (OWM)' 도구 호출됨: {location}")
    if not OPENWEATHER_API_KEY:
        return "OpenWeatherMap API 키(OPENWEATHER_API_KEY)가 설정되지 않았습니다."
    
    try:
        # CSV에서 lat/lon 좌표 가져오기
        coords = get_jeju_coordinates(location)
        lat, lon = coords['lat'], coords['lon']
        
        # OWM 'overview' API 호출
        with httpx.Client() as client:
            weather_url = "https://api.openweathermap.org/data/3.0/onecall/overview"
            params = {
                "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY,
                "units": "metric", "lang": "kr"
            }
            response = client.get(weather_url, params=params)
            response.raise_for_status()
            data = response.json()
            overview_text = data.get("weather_overview")
            
            if not overview_text:
                return f"'{location}'의 OWM 요약 정보를 가져오는 데 실패했습니다."
            return f"'{location}'의 오늘 날씨 요약: {overview_text}"

    except ValueError as e: # (좌표 조회 실패)
        return str(e)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
             return "OpenWeatherMap API 키가 유효하지 않거나 플랜 구독이 필요합니다 (401)."
        return f"OWM API 오류: {e.response.status_code}"
    except Exception as e:
        return f"OWM (오늘 날씨) 오류: {str(e)}"

# '오늘 날씨'용 Tool 래핑
today_weather_tool = Tool.from_function(
    func=get_current_weather,
    name="get_current_weather",
    description=(
        "사용자가 '현재' 또는 '오늘'의 날씨를 물어볼 때 사용됩니다. "
        "위치(location) 인자만 필요합니다. (예: '오늘 제주시 날씨')"
    )
)


# --- 2. '특정 날짜' 예보 도구 (Tool 2 - OWM) ---
def get_date_weather_summary(location: str, date: str) -> str:
    """
    OpenWeatherMap 'Day Summary' API를 호출하여 
    '특정 날짜'(내일, 모레 등)의 날씨 요약을 반환합니다.
    """
    print(f"[Tool] '특정 날짜 예보 (OWM)' 도구 호출됨: {location}, {date}")
    if not OPENWEATHER_API_KEY:
        return "OpenWeatherMap API 키(OPENWEATHER_API_KEY)가 설정되지 않았습니다."
        
    try:
        # CSV에서 lat/lon 좌표 가져오기
        coords = get_jeju_coordinates(location)
        lat, lon = coords['lat'], coords['lon']
        
        # OWM 'day_summary' API 호출
        with httpx.Client() as client:
            weather_url = "https://api.openweathermap.org/data/3.0/onecall/day_summary"
            params = {
                "lat": lat, "lon": lon, "date": date, "appid": OPENWEATHER_API_KEY,
                "units": "metric", "lang": "kr"
            }
            response = client.get(weather_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # (OWM day_summary 응답 파싱)
            description = data.get('summary', '정보 없음')
            temp_min = data.get('temperature', {}).get('min', 'N/A')
            temp_max = data.get('temperature', {}).get('max', 'N/A')
            
            return (
                f"'{location}'의 '{date}' 날씨 예보: {description}, "
                f"예상 기온 {temp_min}°C ~ {temp_max}°C 입니다."
            )
            
    except ValueError as e: # (좌표 조회 실패)
        return str(e)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
             return "OpenWeatherMap API 키가 유효하지 않거나 플랜 구독이 필요합니다 (401)."
        return f"OWM API 오류: {e.response.status_code}"
    except Exception as e:
        return f"OWM (특정 날짜) 오류: {str(e)}"

# '특정 날짜 예보'용 Tool 래핑
future_weather_tool = Tool.from_function(
    func=get_date_weather_summary,
    name="get_date_weather_summary",
    description=(
        "사용자가 '내일', '모레' 또는 '특정 날짜'의 날씨 예보를 물어볼 때 사용됩니다. "
        "위치(location)와 날짜(date, 'YYYY-MM-DD' 형식) 인자가 모두 필요합니다. "
        "LLM은 '내일' 같은 상대적 날짜를 'YYYY-MM-DD'로 변환해야 합니다."
    )
)


# --- 3. 사고 통계 도구 (Tool 3) ---
db_instance = get_db_langchain()

sql_agent_executor = create_sql_agent(
    llm=llm,
    db=db_instance,
    verbose=True,
    handle_parsing_errors=True
)
def run_sql_agent(query: str) -> str:
    print(f"[Tool] SQL Agent 도구 호출됨: {query}")
    try:
        result = sql_agent_executor.invoke({"input": query})
        return result.get("output", "SQL 에이전트 실행 중 오류가 발생했습니다.")
    except Exception as e:
        print(f"[Tool] SQL Agent 오류: {str(e)}")
        return f"SQL 데이터베이스 조회 중 오류 발생: {str(e)}"

safety_sql_tool = Tool(
    name="jeju_safety_statistics_db",
    func=run_sql_agent,
    description=(
        "제주도 관광객 안전 사고 통계(DB)에 대한 질문에 답할 때 사용됩니다. "
        "예: '낙상 사고 건수', '제주시 사고 다발 장소', '겨울철 사고 유형' 등 "
        "질문 전체를 입력으로 전달해야 합니다."
    )
)


# --- 4. 관광정보 RAG 도구 (Tool 4) ---
rag_prompt_template = """
당신은 제주 관광 전문가입니다. 오직 다음 '컨텍스트' 정보만을 기반으로 사용자의 질문에 답변해야 합니다.
컨텍스트에 답변이 없다면 "죄송합니다, 요청하신 정보는 찾을 수 없습니다."라고 답변하세요.

[컨텍스트]
{context}

[질문]
{question}
"""
prompt_rag = ChatPromptTemplate.from_template(rag_prompt_template)

# 앙상블 리트리버 + 문서 압축기 리트리버
compression_retriever = get_compression_retriever()

# RAG 체인(Chain) 결합
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_rag
    | llm
    | StrOutputParser()
)

# RAG 체인을 Tool로 포장
tourism_rag_tool = Tool(
    name="jeju_tourism_rag_search",
    func=rag_chain.invoke, # RAG 체인을 직접 실행
    description=(
        "제주도 관광 명소, 오름, 향토 음식, 한류 컨텐츠 등 일반적인 관광 정보에 대한 질문에 답할 때 사용됩니다. "
        "(예: '제주도 오름 추천해줘', '제주 향토음식이 뭐야?') "
        "질문 전체를 입력으로 전달해야 합니다."
    )
)

# --- 도구 리스트 ---
all_tools = [today_weather_tool, future_weather_tool, safety_sql_tool, tourism_rag_tool]