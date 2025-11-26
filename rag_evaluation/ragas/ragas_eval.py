import os
import pandas as pd
import time
from datetime import datetime
from dotenv import load_dotenv
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas import evaluate
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.core import get_cached_embedder, create_query_processing_chain
from ragas.run_config import RunConfig
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_EVAL_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(RAG_EVAL_DIR)
DATASET_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
LANGSMITH_PROJECT = "Jeju_RAG_Evaluation_v1" 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EVAL_RESULT_PATH = os.path.join(RAG_EVAL_DIR, "result", f"ragas_results_{timestamp}.csv")
PROMPT_FILE = os.path.join(PROJECT_ROOT, "prompts", "jeju_tourism_rag_prompt.yaml")

# Rate Limit 완화 설정
eval_llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=1,
    max_retries=5,  # 재시도 횟수 증가
    timeout=60,
)

eval_embeddings = get_cached_embedder()

RAGAS_METRICS = [
    faithfulness, 
    answer_relevancy, 
    context_recall, 
    context_precision
]

# --- RAG Tool 내부 로직 재정의 (RAGAS 전용 래퍼) ---
def format_docs(docs: list[Document]) -> str:
    """Document 객체 리스트를 텍스트로 포맷"""
    return "\n\n".join(doc.page_content for doc in docs)

@retry(
    stop=stop_after_attempt(5),  # 최대 5회 재시도
    wait=wait_exponential(multiplier=2, min=4, max=60),  # 지수 백오프 (4s ~ 60s)
)
def get_ragas_evaluation_output(query: str) -> dict:
    """
    RAG Tool이 사용하는 리트리버와 LLM 체인을 재결합하여
    답변과 contexts를 수집하여 RAGAS 형식으로 반환하는 래퍼 함수.
    Rate Limit 오류 발생 시 자동 재시도.
    """

    try:
        prompt_rag = load_prompt(PROMPT_FILE) 
    except Exception as e:
        print(f"❌ 프롬프트 로드 오류: {e}")
        return {"answer": "프롬프트 로드 실패", "contexts": []}


    try:
        docs = create_query_processing_chain().invoke({"question": query})
        
        if not docs:
            return {"answer": "정보를 찾을 수 없습니다.", "contexts": []}
        
        generation_chain = (
            prompt_rag
            | eval_llm
            | StrOutputParser()
        )
        answer = generation_chain.invoke({
            "context": format_docs(docs),
            "question": query
        })

        # RAGAS 형식으로 반환: Contexts를 가로채서 포함
        return {"answer": answer, "contexts": docs}

    except Exception as e:
        print(f"[RAGAS Wrapper Error] {e}")
        return {"answer": f"평가용 RAG 실행 중 오류 발생: {e}", "contexts": []}

def run_evaluation():
    if not os.path.exists(DATASET_FILE):
        print(f"❌ 한국어 데이터셋을 찾을 수 없습니다: {DATASET_FILE}. translate_dataset.py를 먼저 실행하세요.")
        return

    # 1. 한국어 데이터셋 로드
    df = pd.read_csv(DATASET_FILE)
    print(f"1. 한국어 데이터셋 로드 완료 (총 {len(df)}개 질문)")
    
    questions = df['user_input'].tolist()
    ground_truths = df['reference'].tolist()
    
    print("2. RAG Tool 실행 및 결과 수집 시작...")
    
    results = []
    
    # 2. RAG Tool만 직접 호출하여 답변 및 Contexts 수집
    for i, q in enumerate(questions):
        print(f"   > 질문 {i+1}/{len(questions)} 처리 중: {q[:30]}...")

        tool_output = get_ragas_evaluation_output(q)
        
        # RAGAS Dataset 형식에 맞게 데이터 준비
        results.append({
            "question": q,
            "answer": tool_output.get("answer", "답변 생성 실패"),
            "contexts": [doc.page_content for doc in tool_output["contexts"]],
            "ground_truth": ground_truths[i]  # 필드명 변경: reference → ground_truth
        })
        
        # ⭐ Rate Limit 완화: 질문 간 1초 지연
        if i < len(questions) - 1:
            time.sleep(3)

    # 3. RAGAS 평가
    ragas_dataset = Dataset.from_list(results)
    
    print("3. RAGAS 평가 실행 중...")

    # Rate Limit을 고려한 RunConfig 설정
    # - max_workers를 1로 낮춰서 동시 요청 최소화
    # - timeout을 늘려서 재시도 시간 확보
    run_config = RunConfig(
        max_workers=2,      # ⭐ 동시 처리를 1개로 제한 (Rate Limit 완화)
        timeout=300,        # 타임아웃을 300초로 연장
        max_retries=5,      # 실패 시 재시도 횟수 증가
    )

    result = evaluate(
        ragas_dataset,
        metrics=RAGAS_METRICS,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=run_config
    )

    # 4. 평가 결과 출력
    print("4. 평가 결과")
    print("---------------------------------------")
    print(result) 
    print("---------------------------------------")
    
    # 5. 결과를 CSV로 저장
    result_df = result.to_pandas()
    result_df.to_csv(EVAL_RESULT_PATH, index=False)
    print(f"✅ RAGAS 평가 결과 CSV 저장 완료: {EVAL_RESULT_PATH}")

if __name__ == "__main__":
    # LangSmith tracing 비활성화
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    run_evaluation()