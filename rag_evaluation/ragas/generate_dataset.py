import os
import glob
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.core import get_cached_embedder

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_md_originals")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "rag_evaluation/dataset", "english_testset.csv")
TEST_SIZE = 10 

generator_llm = ChatOpenAI(model="gpt-5.1", temperature=0, timeout=60)
critic_llm = ChatOpenAI(model="gpt-5.1", temperature=0, timeout=60)

generator_llm_wrapper = LangchainLLMWrapper(generator_llm)
critic_llm_wrapper = LangchainLLMWrapper(critic_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(get_cached_embedder())

def load_raw_markdown_files():
    """
    지정된 폴더(data/processed_md_originals) 내의 모든 .md 파일을 로드합니다.
    """
    print(f"📂 원본 데이터 폴더 탐색: {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ 폴더가 존재하지 않습니다: {DATA_DIR}")
        return []

    # 하위 폴더 포함(**) 모든 .md 파일 검색
    md_files = glob.glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True)
    
    if not md_files:
        print(f"❌ '{DATA_DIR}' 경로에서 .md 파일을 찾을 수 없습니다.")
        return []
    print(f"   -> 총 {len(md_files)}개의 마크다운 파일 발견")

    documents = []
    for file_path in md_files:
        print(f"   - 로드 중: {os.path.basename(file_path)}")
        try:
            # UnstructuredMarkdownLoader는 마크다운 구조(헤더 등)를 잘 파악합니다.
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            # 파일명 메타데이터 추가 (Ragas가 문서를 구분하는 데 중요)
            for doc in docs:
                doc.metadata['filename'] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"   ⚠️ 로드 실패 ({file_path}): {e}")

    return documents

def generate_testset():
    # 1. 원본 문서 로드
    documents = load_raw_markdown_files()
    if not documents: return
    print(f"✅ 로드 완료: 총 {len(documents)}개의 문서 객체 생성됨")

    # 2. Generator 초기화
    generator = TestsetGenerator(
        llm=generator_llm_wrapper,
        embedding_model=ragas_embeddings
    )

    # 3. 질문 분포 설정
    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm_wrapper), 0.7),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm_wrapper), 0.1),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm_wrapper), 0.2),
    ]

    # 4. 데이터셋 생성
    print(f"3. RAGAS 합성 데이터셋 생성 시작 (목표: {TEST_SIZE}개)...")
    print("   (Ragas 기본 파이프라인이 원본 MD 파일을 자동으로 분석합니다)")
    try:
        # transforms=None (기본값)을 사용하여 Ragas가 
        # HeadlineSplitter -> EmbeddingExtractor 등의 표준 과정을 수행하게 합니다.
        testset = generator.generate_with_langchain_docs(
            documents=documents,
            testset_size=TEST_SIZE,
            query_distribution=query_distribution,
            raise_exceptions=False 
        )

        # 5. 저장
        print("4. 데이터셋 생성 완료, CSV 저장 중...")
        df = testset.to_pandas()
        
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig') 
        print(f"✅ 데이터셋 저장 성공: {OUTPUT_FILE_PATH}")
        print(f"   -> 생성된 질문 수: {len(df)}")
        
        if not df.empty:
            print(df[['user_input', 'reference']].head(2))

    except Exception as e:
        print(f"❌ 데이터셋 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # LangSmith tracing 비활성화
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    generate_testset()