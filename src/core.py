import os
from typing import List, Callable
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangChain 1.0+ 이동 모듈
from langchain_classic.storage import LocalFileStore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.retrievers import (
    ContextualCompressionRetriever, 
    EnsembleRetriever
)
from langchain_classic.retrievers.document_compressors import (
    DocumentCompressorPipeline, 
    EmbeddingsFilter
)
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter, 
    LongContextReorder
)

# Core 및 Postgres
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from util.stopwords import get_korean_stopwords
from src.kiwi_tokenizer import KiwiBM25Tokenizer
from langchain_cohere import CohereRerank
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# .env 파일 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(CURRENT_DIR)
PROMPT_FILE = os.path.join(CORE_DIR, "prompts", "search_query_translation.yaml")

# API 키 환경변수 설정
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENWEATHER_API_KEY:
    os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY
if DEEPSEEK_API_KEY:
    os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# LLM 설정
llm_default = ChatOpenAI(
    model="gpt-4.1",
    temperature=1
)

llm_rag = ChatOpenAI(
    model="gpt-5.1",
    temperature=0,
    max_retries=10,
    timeout=120
)

llm_streaming = ChatOpenAI(
    model="gpt-5.1", 
    temperature=0, 
    model_kwargs={
        "stream_options": {"include_usage": True}
    }
)

# --- PostgreDB 싱글톤 ---
engine = create_engine(DATABASE_URL or "")
_db_langchain= None

def get_db_langchain():
    global _db_langchain
    if _db_langchain is not None:
        return _db_langchain
    _db_langchain = SQLDatabase(engine=engine, include_tables=['jeju_accidents'])
    return _db_langchain

# --- Cached Embedder 싱글톤 ---
_cached_embedder_instance = None

def get_cached_embedder():
    global _cached_embedder_instance
    if _cached_embedder_instance is not None:
        return _cached_embedder_instance

    print("Initializing CacheBackedEmbeddings...")
    store = LocalFileStore(root_path="./.cache/embeddings")
    underlying_embeddings = OpenAIEmbeddings()
    _cached_embedder_instance = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, 
        store, 
        namespace="tourism_docs"
    )
    return _cached_embedder_instance

# --- PGVector Store 싱글톤 ---
_vector_store_instance = None
COLLECTION_NAME = "tourism_docs"

def get_vector_store():
    global _vector_store_instance
    if _vector_store_instance is not None:
        return _vector_store_instance

    print(f"Initializing PGVector connection to collection: {COLLECTION_NAME}...")
    _vector_store_instance = PGVector(
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        embeddings=get_cached_embedder()
    )
    return _vector_store_instance

# --- BM25 관련 유틸리티 ---
_bm25_retriever_instance = None
_korean_bm25_tokenizer = None 

def get_korean_bm25_tokenizer():
    global _korean_bm25_tokenizer
    if _korean_bm25_tokenizer is not None:
        return _korean_bm25_tokenizer

    korean_stopwords = get_korean_stopwords()
    _korean_bm25_tokenizer = KiwiBM25Tokenizer(stop_words=korean_stopwords)
    print(f"✅ BM25 Tokenizer 초기화 완료. 불용어 {len(korean_stopwords)}개 적용됨.")
    return _korean_bm25_tokenizer

def load_documents_from_vectorstore():
    print("벡터스토어에서 문서 로드 중...")
    sql_query = f"""
        SELECT document, cmetadata 
        FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'
        );
    """
    documents = []
    try:
        with engine.connect() as connection:
            results = connection.execute(text(sql_query))
            for row in results:
                documents.append(Document(page_content=row[0], metadata=row[1]))
        
        print(f"✅ 벡터스토어에서 {len(documents)}개 문서 로드 완료")
        if not documents:
            print("❌ 로드된 문서가 없습니다. 벡터스토어를 확인하세요.")
            return []
        return documents
    except Exception as e:
        print(f"❌ 벡터스토어 로드 실패: {e}")
        return []

def get_bm25_retriever():
    global _bm25_retriever_instance
    if _bm25_retriever_instance is not None:
        return _bm25_retriever_instance

    print("Initializing BM25Retriever (from PGVector's stored documents)...")
    splits_from_db = load_documents_from_vectorstore()

    if not splits_from_db:
        print("❌ BM25 Error: DB에 문서가 없어 'splits'가 비어있습니다.")
        _bm25_retriever_instance = BM25Retriever.from_documents([], k=5)
    else:
        _bm25_retriever_instance = BM25Retriever.from_documents(
            splits_from_db, 
            k=5,
            preprocess_func=get_korean_bm25_tokenizer()
        )
    print("BM25Retriever initialization complete.")
    return _bm25_retriever_instance


# --- 쿼리 처리 체인 ---
def get_query_rewrite_chain():
    llm = ChatOpenAI(model="gpt-5.1", temperature=0)
    
    try:
        prompt = load_prompt(PROMPT_FILE)
    except Exception as e:
        print(f"⚠️ 프롬프트 로드 실패, 기본 프롬프트를 사용합니다: {e}")
        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "Translate the following to natural Korean for search: {question}"
        )
    
    return prompt | llm | StrOutputParser()


# --- 최종 RAG 리트리버 싱글톤 ---
_compression_retriever_instance = None

def get_compression_retriever():
    """
    압축/필터링 기능이 포함된 최종 앙상블 리트리버 싱글톤 객체를 반환합니다.
    """
    global _compression_retriever_instance
    if _compression_retriever_instance is not None:
        return _compression_retriever_instance

    print("Initializing Compression Retriever (Ensemble + Filters)...")

    # 1. 의미(Vector) 리트리버: 자연어 쿼리 선호
    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": 5})
    
    # 2. 키워드(BM25) 리트리버: 키워드 쿼리 선호(bm25 내부에서 등록한 토크나이저를 통해 쿼리에서 키워드 추출 작업 수행)
    raw_bm25_retriever = get_bm25_retriever()
    
    # 3. 앙상블 리트리버 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[raw_bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. 압축 필터 생성
    cached_embedder = get_cached_embedder()
    
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.95
    )
    relevance_filter = EmbeddingsFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.7
    )

    reorder_transformer = LongContextReorder()

    reranker = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=5
    )
    
    # 5. 압축 파이프라인 생성
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevance_filter, reranker]
    )
    
    # 6. 최종 압축 리트리버 생성
    _compression_retriever_instance = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=ensemble_retriever
    )
    
    print("Compression Retriever initialization complete.")
    return _compression_retriever_instance

def create_query_processing_chain():
    """
    사용자 쿼리 → 영어/한국어 자연어 최적화 → 리트리버 호출
    (키워드 변환은 이제 리트리버 내부에서 수행됨)
    """
    
    query_rewriter = get_query_rewrite_chain()
    compression_retriever = get_compression_retriever()
    
    processing_chain = (
        {"question": RunnablePassthrough()}
        | query_rewriter                      # 1. 자연어 최적화 (번역 및 의도 파악)
        | compression_retriever               # 2. 최종 문서 검색
    )
    return processing_chain

# --- 디버깅용 체인 ---
def create_debug_query_chain():
    """
    각 단계의 중간 결과를 확인할 수 있는 디버깅 체인
    """
    query_rewriter = get_query_rewrite_chain()
    
    def log_optimized_query(query: str) -> str:
        print(f"🔹 한글 자연어 쿼리: {query}")
        return query
    
    def log_retrieval_results(docs):
        print(f"🔹 검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs[:5], 1):
            # [수정] f-string 내부 백슬래시 에러 방지를 위해 변수에 할당 후 출력
            content = doc.page_content[:60].replace('\n', ' ')
            print(f"   {i}. {content}...")
        return docs
    
    compression_retriever = get_compression_retriever()
    
    debug_chain = (
        {"question": RunnablePassthrough()}
        | query_rewriter
        | RunnableLambda(log_optimized_query)
        | compression_retriever
        | RunnableLambda(log_retrieval_results)
    )
    
    return debug_chain

def debug_retriever_pipeline(query: str):
    """
    리트리버 파이프라인 디버깅 함수
    """
    print("\n" + "="*60)
    print(f"🔍 디버깅: 원본 쿼리 = '{query}'")
    
    rewriter = get_query_rewrite_chain()
    optimized_query = rewriter.invoke({"question": query})
    print(f"📝 최적화된 쿼리 (자연어): {optimized_query}")
    print("="*60)

    # Stage 1: BM25 (Wrapped)
    print("\n[Stage 1] BM25 검색 (내부적으로 키워드 변환 수행)")
    raw_bm25_retriever = get_bm25_retriever()
    bm25_docs = raw_bm25_retriever.invoke(optimized_query)
    for i, doc in enumerate(bm25_docs[:3], 1):
        # [수정] 에러 방지용 변수 할당
        content = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content}...")
    
    # Stage 2: Vector
    print("\n[Stage 2] Vector 검색 (자연어 쿼리 사용)")
    vector_ret = get_vector_store().as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_ret.invoke(optimized_query)
    for i, doc in enumerate(vector_docs[:3], 1):
        content = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content}...")
    
    # Stage 3: Ensemble
    print("\n[Stage 3] Ensemble 통합 결과")
    ensemble_ret = EnsembleRetriever(
        retrievers=[raw_bm25_retriever, vector_ret],
        weights=[0.5, 0.5]
    )
    ensemble_docs = ensemble_ret.invoke(optimized_query)
    for i, doc in enumerate(ensemble_docs[:3], 1):
        content = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content}...")
    
    # Stage 4: Final
    print("\n[Stage 4] 필터링 최종 결과")
    compression_ret = get_compression_retriever()
    final_docs = compression_ret.invoke(optimized_query)
    for i, doc in enumerate(final_docs, 1):
        content = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i}. {content}...")
    
    print("\n" + "="*60)
    return final_docs