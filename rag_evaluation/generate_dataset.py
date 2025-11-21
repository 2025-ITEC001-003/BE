import os
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from src.core import get_cached_embedder

load_dotenv()

generator_llm = ChatOpenAI(model="gpt-4o-mini")
critic_llm = ChatOpenAI(model="gpt-4o-mini")
ragas_embeddings = LangchainEmbeddingsWrapper(get_cached_embedder())

CURRENT_FILE_PATH = os.path.abspath(__file__)
RAG_EVAL_DIR = os.path.dirname(CURRENT_FILE_PATH)
PROJECT_ROOT = os.path.dirname(RAG_EVAL_DIR)
DOC_DIR = os.path.join(PROJECT_ROOT, "data", "processed_md_originals") 
OUTPUT_FILE_PATH = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
TEST_SIZE = 30 

def load_and_split_documents(path: str):
    """
    processed_md_originals 폴더의 순수 .md 파일들만 로드하고 청크로 분할합니다.
    """
    print(f"LlamaParse Markdown 파일 로드 경로: {path}")
    
    loader = DirectoryLoader(
        path=path,
        glob="*.md", 
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        recursive=False 
    )
    documents = loader.load()
    
    if not documents:
        print("❌ 로드된 Markdown 파일이 없습니다. data/processed_md_originals를 확인하세요.")
        return []

    print(f"RAGAS 합성 테스트 데이터셋 생성 시 입력으로 사용할 원본 파일 개수: {len(documents)}개")

    # 청크 분할 사이즈 및 오버랩 설정 제주 관광 RAG와 동일하게 적용
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def generate_testset():
    print(f"1. LlamaParse Markdown 문서 로드 및 분할 중...")
    documents = load_and_split_documents(DOC_DIR)
    
    if not documents:
        return

    # 2. TestsetGenerator 초기화
    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        ragas_embeddings
    )

    # 3. 질문유형 분포 정의
    distributions = {
        simple: 0.4,
        reasoning: 0.2,
        multi_context: 0.2,
        conditional: 0.2
    }

    print(f"2. RAGAS 합성 데이터셋 생성 시작 (총 {TEST_SIZE}개 질문)...")
    testset = generator.generate_with_langchain_docs(
        documents,
        test_size=TEST_SIZE,
        distributions=distributions
    )

    # 4. CSV로 저장
    df = testset.to_pandas()
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"✅ 데이터셋 생성 완료: {OUTPUT_FILE_PATH} (총 {len(df)}개)")

if __name__ == "__main__":
    generate_testset()