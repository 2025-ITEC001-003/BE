import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter, LongContextReorder
from util.stopwords import get_korean_stopwords
from src.kiwi_tokenizer import KiwiBM25Tokenizer

# .env 파일 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# OpenAI API 키를 환경변수에 명시적으로 설정
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

# llm_default : ConversationSummaryBufferMemory 요약, sql_agent
llm_default = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=1,
    api_key=OPENAI_API_KEY
)

# RAG용 llm (stream option 미포함)
llm_rag = ChatOpenAI(
    model="gpt-4.1",
    temperature=0,
    max_retries=10,
    timeout=120,
    api_key=OPENAI_API_KEY
)

# llm_streaming : 메인 스트리밍 Agent용 (토큰 추적 옵션 포함)
llm_streaming = ChatOpenAI(
    model="gpt-4.1", 
    temperature=0, 
    api_key=OPENAI_API_KEY,
    model_kwargs={
        "stream_options": {"include_usage": True}
    }
)

# # --- PostgreDB 싱글톤 (기존) ---
engine = create_engine(DATABASE_URL)
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
    # 캐시를 저장할 로컬 파일 저장소 설정
    store = LocalFileStore(root_path="./.cache/embeddings")
    
    # 기본 임베딩 모델 (OpenAI)
    underlying_embeddings = OpenAIEmbeddings()
    
    # 캐시 지원 임베딩 모델 생성
    _cached_embedder_instance = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings, 
        store, 
        namespace="tourism_docs" # 캐시 ID
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

# --- BM25 Retriever 싱글톤 ---
_bm25_retriever_instance = None
_korean_bm25_tokenizer = None 

def get_korean_bm25_tokenizer():
    """불용어를 로드하고 KiwiBM25Tokenizer를 초기화하는 싱글톤 함수"""
    global _korean_bm25_tokenizer
    if _korean_bm25_tokenizer is not None:
        return _korean_bm25_tokenizer

    korean_stopwords = get_korean_stopwords()
    _korean_bm25_tokenizer = KiwiBM25Tokenizer(stop_words=korean_stopwords)
    print(f"✅ BM25 Tokenizer 초기화 완료. 불용어 {len(korean_stopwords)}개 적용됨.")
    return _korean_bm25_tokenizer

def load_documents_from_vectorstore():
    """
    벡터스토어(DB)에서 문서를 로드합니다.
    BM25 리트리버 초기화 및 RAGAS 평가용 문서 생성에 사용됩니다.
    """
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
    """
    BM25Retriever 싱글톤 객체를 반환합니다.
    최초 호출 시 load_documents_from_vectorstore()를 사용하여
    DB(PGVector)에 저장된 원본 텍스트를 로드합니다.
    """
    global _bm25_retriever_instance
    if _bm25_retriever_instance is not None:
        return _bm25_retriever_instance

    print("Initializing BM25Retriever (from PGVector's stored documents)...")
    
    # DB에서 문서 로드
    splits_from_db = load_documents_from_vectorstore()

    # BM25 리트리버 생성
    if not splits_from_db:
        print("❌ BM25 Error: DB에 문서가 없어 'splits'가 비어있습니다.")
        _bm25_retriever_instance = BM25Retriever.from_documents([], k=2)
    else:
        _bm25_retriever_instance = BM25Retriever.from_documents(
            splits_from_db, 
            k=2,
            custom_tokenizer=get_korean_bm25_tokenizer()
        )
    
    print("BM25Retriever initialization complete.")
    return _bm25_retriever_instance

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
    
    # 1. 의미, 키워드 리트리버 가져오기
    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": 2})
    bm25_retriever = get_bm25_retriever()
    
    # 2. 앙상블 리트리버 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    
    # 3. 압축 필터 생성 (임베더 싱글톤 사용)
    cached_embedder = get_cached_embedder()
    
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.95
    )
    relevance_filter = EmbeddingsFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.80
    )

    reorder_transformer = LongContextReorder()
    
    # 4. 압축 파이프라인 생성
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevance_filter, reorder_transformer]
    )
    
    # 5. 최종 압축 리트리버 생성 및 저장
    _compression_retriever_instance = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=ensemble_retriever
    )
    
    print("Compression Retriever initialization complete.")
    return _compression_retriever_instance

# --- 디버깅용: 리트리버 각 단계별 결과 확인 함수 ---
def debug_retriever_pipeline(query: str):
    """
    리트리버 파이프라인의 각 단계를 분석하여 디버깅합니다.
    - Stage 1: BM25 결과
    - Stage 2: Vector 결과
    - Stage 3: Ensemble 결과
    - Stage 4: 필터링 후 최종 결과
    """
    print("\n" + "="*60)
    print(f"🔍 디버깅: 쿼리 = '{query}'")
    print("="*60)
    
    # Stage 1: BM25
    bm25_ret = get_bm25_retriever()
    bm25_docs = bm25_ret.invoke(query)
    print(f"\n[Stage 1] BM25 검색 결과: {len(bm25_docs)}개")
    for i, doc in enumerate(bm25_docs, 1):
        content_preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content_preview}...")
    
    # Stage 2: Vector
    vector_ret = get_vector_store().as_retriever(search_kwargs={"k": 5})
    vector_docs = vector_ret.invoke(query)
    print(f"\n[Stage 2] Vector 검색 결과: {len(vector_docs)}개")
    for i, doc in enumerate(vector_docs, 1):
        content_preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content_preview}...")
    
    # Stage 3: Ensemble
    ensemble_ret = EnsembleRetriever(
        retrievers=[bm25_ret, vector_ret],
        weights=[0.5, 0.5]
    )
    ensemble_docs = ensemble_ret.invoke(query)
    print(f"\n[Stage 3] Ensemble 통합 결과: {len(ensemble_docs)}개")
    for i, doc in enumerate(ensemble_docs, 1):
        content_preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {content_preview}...")
    
    # Stage 4: Final (Compression + Filters)
    compression_ret = get_compression_retriever()
    final_docs = compression_ret.invoke(query)
    print(f"\n[Stage 4] 필터링 최종 결과: {len(final_docs)}개")
    for i, doc in enumerate(final_docs, 1):
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i}. {content_preview}...")
    
    print("\n" + "="*60)
    return final_docs