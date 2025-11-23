import os
import sys

# 절대 경로 기반 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_eval_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(rag_eval_dir)

# src 모듈 로드를 위해 project_root를 Python path에 추가
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from langchain.smith import RunEvalConfig, run_on_dataset
from langsmith import Client
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.core import get_compression_retriever, llm_default

load_dotenv()

# 1. 설정 및 리소스 로드
# 평가할 데이터셋 이름 (upload_dataset.py에서 지정한 이름과 동일해야 함)
DATASET_NAME = "Jeju_Tourism_QA_Set_KO"

PROMPT_FILE = os.path.join(project_root, "prompts", "jeju_tourism_rag_prompt.yaml")
try:
    rag_prompt = load_prompt(PROMPT_FILE)
    print(f"✅ 프롬프트 로드 성공: {PROMPT_FILE}")
except Exception as e:
    print(f"⚠️ 프롬프트 로드 실패. 기본 프롬프트를 사용합니다. ({e})")
    from langchain_core.prompts import ChatPromptTemplate
    rag_prompt = ChatPromptTemplate.from_template(
        """다음 정보를 바탕으로 질문에 답하세요:\n\n{context}\n\n질문: {question}"""
    )

retriever = get_compression_retriever()

# 2. 평가용 RAG 체인 구성 (Context 반환 필수)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def evaluation_target_chain(inputs):
    """
    LangSmith 평가를 위한 RAG 체인 래퍼 함수입니다.
    context_qa 평가를 위해 'answer'와 함께 'contexts'를 반환해야 합니다.
    """
    question = inputs["question"]
    
    # 1. 검색 (Retrieval)
    docs = retriever.invoke(question)
    formatted_context = format_docs(docs)
    
    # 2. 답변 생성 (Generation)
    chain = (
        rag_prompt 
        | llm_default 
        | StrOutputParser()
    )
    
    answer = chain.invoke({
        "context": formatted_context,
        "question": question
    })
    
    # 3. 결과 반환 (중요: contexts 키 포함)
    return {
        "answer": answer,           # 생성된 답변
        "contexts": [d.page_content for d in docs], # 검색된 문서 내용 리스트 (context_qa용)
        "retrieved_docs": docs      # (선택) 메타데이터 포함 원본 문서
    }

# 3. LangSmith 평가 설정
def run_evaluation():
    client = Client()
    
    # 평가자(Judge) 모델 설정 - 정확한 평가를 위해 gpt-4o 권장
    eval_llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0
    )

    # 평가 지표 설정
    eval_config = RunEvalConfig(
        evaluators=[
            # 1. QA (Correctness): 정답(Ground Truth)과 얼마나 유사한지 평가
            "qa", 
            
            # 2. CoT QA (Chain of Thought): 이유를 먼저 생각하고 채점 (더 정확함)
            "cot_qa",
            
            # 3. Context QA (Context Relevance): 
            # 답변이 검색된 문서(Context)에 기반했는지 평가 (Hallucination 체크)
            "context_qa", 
        ],
        eval_llm=eval_llm,
        # 예측값과 참조값(정답)의 키 매핑
        prediction_key="answer",  
        reference_key="answer",   # 데이터셋 업로드 시 ground_truth를 'answer'로 매핑했는지 확인 필요
        input_key="question"      # 데이터셋의 질문 컬럼
    )

    print(f"🚀 LangSmith 평가 시작: {DATASET_NAME}")
    print(f"   - Evaluators: qa, cot_qa, context_qa")
    
    try:
        results = run_on_dataset(
            client=client,
            dataset_name=DATASET_NAME,
            llm_or_chain_factory=evaluation_target_chain,
            evaluation=eval_config,
            project_name="jeju-rag-eval-experiment-v1", # 실험 프로젝트 이름 (버전 관리용)
        )
        print("✅ 평가 완료! LangSmith 대시보드에서 결과를 확인하세요.")
        print(f"🔗 프로젝트 링크: {results['project_url'] if 'project_url' in results else 'N/A'}")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        print("팁: 데이터셋 이름이 정확한지, LangSmith API Key가 설정되었는지 확인하세요.")

if __name__ == "__main__":
    run_evaluation()