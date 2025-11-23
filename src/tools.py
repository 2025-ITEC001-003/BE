import httpx
import os
from langchain.tools import tool
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ddgs import DDGS

from src.core import get_db_langchain, OPENWEATHER_API_KEY, llm_default, llm_rag, get_compression_retriever
from src.data_loader import get_jeju_coordinates

# --- 1. '오늘/현재' 날씨 도구 (Tool 1 - OWM) ---
@tool
def get_current_weather(location: str) -> str:
    """
    사용자가 '현재' 또는 '오늘'의 날씨를 물어볼 때 사용됩니다.
    
    Args:
        location (str): 날씨를 조회할 제주도 내의 특정 위치 (예: '제주시', '애월읍')
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

# --- 2. '특정 날짜' 예보 도구 (Tool 2 - OWM) ---
@tool
def get_date_weather_summary(location: str, date: str) -> str:
    """
    사용자가 '내일', '모레' 또는 '특정 날짜'의 날씨 예보를 물어볼 때 사용됩니다.
    [필수] 'location'과 'date' 인자 2개가 모두 필요합니다.
    만약 사용자가 질문에서 위치(location)를 생략했다면, **반드시 대화 기록(context)에서 위치를 찾아 함께 전달해야 합니다.**
    
    Args:
        location (str): 날씨를 조회할 제주도 내의 특정 위치 (예: '제주시', '애월읍')
        date (str): 조회할 미래 날짜 (YYYY-MM-DD 형식)
    """
    print(f"[Tool] '특정 날짜 예보 (OWM)' 도구 호출됨: {location}, {date}")
    if not OPENWEATHER_API_KEY:
        return "OpenWeatherMap API 키(OPENWEATHER_API_KEY)가 설정되지 않았습니다."
        
    try:
        # CSV에서 lat/lon 좌표 가져오기
        coords = get_jeju_coordinates(location)
        print(f"[coords] {coords}")
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
            temp_min = data.get('temperature', {}).get('min', 'N/A')
            temp_max = data.get('temperature', {}).get('max', 'N/A')
            precipitation = data.get('precipitation', {}).get('total', 0.0)
            # 오후 구름 양을 대표로 사용
            cloud_cover = data.get('cloud_cover', {}).get('afternoon', -1) 

            description_parts = []
            # 1. 강수 정보
            if precipitation > 0.0:
                description_parts.append(f"총 {precipitation}mm의 비가 예상됩니다.")
            # 2. 구름 정보 (비가 오지 않을 경우)
            elif cloud_cover != -1:
                if cloud_cover < 30:
                    description_parts.append("대체로 맑겠습니다.")
                elif cloud_cover < 70:
                    description_parts.append("구름이 다소 있겠습니다.")
                else:
                    description_parts.append("흐리겠습니다.")
            else:
                description_parts.append("날씨 정보를 가져올 수 없습니다.") # Fallback
                
            description = " ".join(description_parts)
            
            return (
                f"'{location}'의 '{date}' 날씨 예보: {description} "
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


# --- 3. 사고 통계 도구 (Tool 3) ---
db_instance = get_db_langchain()

sql_agent_executor = create_sql_agent(
    llm=llm_default,
    db=db_instance,
    verbose=True,
    handle_parsing_errors=True
)

@tool
def jeju_safety_statistics_db(query: str) -> str:
    """
    제주도 관광객 안전 사고 통계(DB)에 대한 질문에 답할 때 사용됩니다.
    예: '낙상 사고 건수', '제주시 사고 다발 장소', '겨울철 사고 유형' 등
    질문 전체를 입력으로 전달해야 합니다.
    
    Args:
        query (str): SQL 에이전트에게 전달할 원본 사용자 질문
    """
    print(f"[Tool] SQL Agent 도구 호출됨: {query}")
    try:
        result = sql_agent_executor.invoke({"input": query})
        return result.get("output", "SQL 에이전트 실행 중 오류가 발생했습니다.")
    except Exception as e:
        print(f"[Tool] SQL Agent 오류: {str(e)}")
        return f"SQL 데이터베이스 조회 중 오류 발생: {str(e)}"

# --- 4. 관광정보 RAG 도구 (Tool 4) ---
# 4-1 RAG 체인
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
PROMPT_FILE = os.path.join(PROJECT_ROOT, "prompts", "jeju_tourism_rag_prompt.yaml")
prompt_rag = load_prompt(PROMPT_FILE)

# 앙상블 리트리버 + 문서 압축기 리트리버
compression_retriever = get_compression_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

local_rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt_rag
    | llm_default
    | StrOutputParser()
)

# 4-2 웹 검색 체인
def run_ddgs_search(query: str) -> str:
    """
    ddgs 라이브러리를 사용해 웹 검색을 수행하고,
    상위 3개 결과의 요약(snippet)을 반환합니다.
    """
    print(f"[Tool] DDGS Web Search: {query}")
    try:
        # backend='api' 경고가 발생하지 않는 최신 방식 사용
        # 상위 3개 결과를 가져와 컨텍스트로 사용
        results = DDGS().text(query, max_results=3)
        if not results:
            return "웹에서 관련 정보를 찾을 수 없습니다."

        # 3개 결과의 요약본(body)을 결합하여 반환
        snippets = [r.get('body', '') for r in results]
        return "\n\n".join(snippets)

    except Exception as e:
        print(f"[Tool] DDGS Search Error: {e}")
        return "웹 검색 중 오류가 발생했습니다."

web_rag_prompt_template = """
당신은 유능한 검색 어시스턴트입니다. '로컬 문서'에 정보가 없어서 웹 검색을 수행했습니다.
오직 다음 '웹 검색 컨텍스트' 정보만을 기반으로 사용자의 질문에 답변해야 합니다.
웹 검색 결과에도 답변이 없다면 "웹에서도 관련 정보를 찾을 수 없습니다."라고 답변하세요.

[웹 검색 컨텍스트]
{context}

[질문]
{question}
"""
web_prompt_rag = ChatPromptTemplate.from_template(web_rag_prompt_template)

web_rag_chain = (
    {"context": run_ddgs_search, "question": RunnablePassthrough()}
    | web_prompt_rag
    | llm_default
    | StrOutputParser()
)

# RAG 체인을 Tool로 포장
@tool
def jeju_tourism_rag_search(query: str) -> str:
    """
    제주도 관광 명소, 오름, 향토 음식, 한류 컨텐츠 등 일반적인 관광 정보에 대한 질문에 답할 때 사용됩니다.
    이 도구는 먼저 로컬 문서를 검색하고, 정보가 없으면 자동으로 웹 검색을 시도합니다.
    (예: '제주도 오름 추천해줘', '애월읍 최신 축제 정보')
    
    Args:
        query (str): RAG 시스템에 전달할 원본 사용자 질문
    """
    print(f"[Tool] RAG 도구 호출됨: {query}")
    
    try:
        # 1. (시도) 압축 리트리버를 먼저 실행
        print("[RAG] 1. 로컬 문서 검색 시도...")
        # (Ensemble -> RedundantFilter -> RelevanceFilter -> LongContextReorder(긴 문맥 재정렬) 실행)
        docs = compression_retriever.invoke(query)

        # 2. (판단) 필터링된 문서가 있는지 확인
        if not docs:
            # RelevanceFilter가 모든 문서를 '관련 없음'으로 판단
            print("[RAG] 1-1. 로컬 문서에 유효한 정보 없음. 웹 검색으로 대체.")
            raise ValueError("No relevant documents found in local RAG.")
        
        # 3. (RAG 성공) 필터링된 문서로 답변 생성
        print(f"[RAG] 1-2. 로컬 문서 {len(docs)}개 청크로 답변 생성.")
        # local_rag_chain을 실행 (이미 리트리버가 실행되었으므로 수동 주입)
        generation_chain = (
            prompt_rag
            | llm_rag
            | StrOutputParser()
        )
        local_answer = generation_chain.invoke({
            "context": format_docs(docs),
            "question": query
        })

        # 사용중인 RAG 프롬프트에서는 컨텍스트에 답이 없을 경우
        not_found_phrase = "죄송합니다, 요청하신 정보는 찾을 수 없습니다."
        # 응답에 추가 공백이나 문장부호 여지가 있으므로 포함 여부로 판단
        if local_answer and not_found_phrase in local_answer:
            print("[RAG] 1-3. 주어진 컨텍스트로 답변할 수 없음. 웹 검색으로 대체.")
            raise ValueError("RAG answer indicates no information found.")

        return local_answer

    except Exception as e:
        # 4. (RAG 실패 / Fallback) 웹 검색 실행
        print(f"[RAG] 2. RAG 오류({e}) 발생. 웹 검색으로 대체 실행.")
        try:
            return web_rag_chain.invoke(query)
        except Exception as web_e:
            print(f"[Tool] Web Search 오류: {web_e}")
            return f"RAG 및 웹 검색 중 오류 발생: {web_e}"


# --- 도구 리스트 ---
all_tools = [
    get_current_weather, 
    get_date_weather_summary, 
    jeju_safety_statistics_db, 
    jeju_tourism_rag_search
]