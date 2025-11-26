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
from langchain_core.rate_limiters import InMemoryRateLimiter
from src.core import get_cached_embedder

load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) 
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed_md_originals")
OUTPUT_FILE_PATH = os.path.join(PROJECT_ROOT, "rag_evaluation/dataset", "english_testset.csv")
TEST_SIZE = 30

# Rate Limit & Retry 적용된 LLM 생성
rate_limiter = InMemoryRateLimiter(
    requests_per_second=2.0,      # 초당 요청 2회 정도
    check_every_n_seconds=0.1,
    max_bucket_size=4            # 약간의 버스트 허용
)

generator_llm = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    timeout=60,
    max_retries=5,   # LLM 내부 재시도
    response_format={"type": "json_object"},
    rate_limiter=rate_limiter
)

generator_llm_wrapper = LangchainLLMWrapper(generator_llm)
ragas_embeddings = LangchainEmbeddingsWrapper(get_cached_embedder())

# Markdown 파일 로드
def load_raw_markdown_files():
    print(f"📂 원본 데이터 폴더 탐색: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"❌ 폴더가 존재하지 않습니다: {DATA_DIR}")
        return []

    md_files = glob.glob(os.path.join(DATA_DIR, "**/*.md"), recursive=True)

    if not md_files:
        print(f"❌ '{DATA_DIR}' 경로에서 .md 파일을 찾을 수 없습니다.")
        return []

    print(f"   -> 총 {len(md_files)}개의 마크다운 파일 발견")

    documents = []
    for file_path in md_files:
        print(f"   - 로드 중: {os.path.basename(file_path)}")
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["filename"] = os.path.basename(file_path)

            documents.extend(docs)

        except Exception as e:
            print(f"   ⚠️ 로드 실패 ({file_path}): {e}")

    return documents

# 전체 파이프라인 재시도 (핵심)
def generate_with_retry(generator, documents, test_size, query_distribution, max_attempts=5):
    for attempt in range(1, max_attempts + 1):
        print(f"\n=============================")
        print(f"  🔁 테스트셋 생성 시도 {attempt}/{max_attempts}")
        print(f"=============================\n")

        try:
            testset = generator.generate_with_langchain_docs(
                documents=documents,
                testset_size=test_size,
                query_distribution=query_distribution,
                raise_exceptions=False
            )

            df = testset.to_pandas()

            # 충분히 생성되면 성공
            if len(df) >= test_size:
                print(f"🎉 성공적으로 생성됨: {len(df)}개")
                return df

            print(f"⚠️ {len(df)}개밖에 생성되지 않음 → 재시도 필요")

        except Exception as e:
            print(f"⚠️ 생성 중 예외 발생 → 재시도: {e}")

    raise RuntimeError("❌ 테스트셋 생성 실패: 최대 재시도 횟수 초과")

# Testset 생성 메인 로직
def generate_testset():
    documents = load_raw_markdown_files()
    if not documents:
        return

    print(f"✅ 문서 로드 완료 (총 {len(documents)}개)")

    generator = TestsetGenerator(
        llm=generator_llm_wrapper,
        embedding_model=ragas_embeddings
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm_wrapper), 0.8),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm_wrapper), 0.1),
        (MultiHopAbstractQuerySynthesizer(llm=generator_llm_wrapper), 0.1),
    ]

    print(f"📌 목표 테스트셋 크기: {TEST_SIZE}\n")

    df = generate_with_retry(
        generator=generator,
        documents=documents,
        test_size=TEST_SIZE,
        query_distribution=query_distribution,
        max_attempts=5
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    df.to_csv(OUTPUT_FILE_PATH, index=False, encoding="utf-8-sig")

    print(f"\n✅ CSV 저장 완료: {OUTPUT_FILE_PATH}")
    print(f"   -> 생성된 질문 수: {len(df)}")

    if not df.empty:
        print(df[['user_input', 'reference']].head(2))


if __name__ == "__main__":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    generate_testset()
