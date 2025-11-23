import os
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from src.core import get_cached_embedder, load_documents_from_vectorstore

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# 1. Generator: 데이터 생성용 (DeepSeek V3 사용)
generator_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_retries=10,
    timeout=120
)

# 수정된 Critic 설정
critic_llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_retries=10,
    timeout=120
)
ragas_embeddings = LangchainEmbeddingsWrapper(get_cached_embedder())

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_EVAL_DIR = os.path.dirname(CURRENT_DIR)
OUTPUT_FILE_PATH = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
TEST_SIZE = 30 

def generate_testset():
    print(f"1. 벡터스토어에서 문서 로드 중...")
    documents = load_documents_from_vectorstore()
    
    if not documents:
        print("❌ 문서 로드 실패")
        return

    print(f"2. TestsetGenerator 초기화 중...")
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        ragas_embeddings
    )

    # 질문유형 분포 정의
    distributions = {
        simple: 0.7,
        reasoning: 0.1,
        multi_context: 0.1,
        conditional: 0.1
    }

    print(f"3. RAGAS 합성 데이터셋 생성 시작 (총 {TEST_SIZE}개 질문)...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=TEST_SIZE,
        distributions=distributions,
    )

    # CSV로 저장
    df = testset.to_pandas()
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig') 
    print(f"✅ 데이터셋 생성 완료: {OUTPUT_FILE_PATH} (총 {len(df)}개)")

if __name__ == "__main__":
    # LangSmith tracing 비활성화
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    generate_testset()