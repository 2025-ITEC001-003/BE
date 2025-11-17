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

# .env 파일 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# OpenAI API 키를 환경변수에 명시적으로 설정 (LangChain이 사용)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY

# llm_default : ConversationSummaryBufferMemory 요약, sql_agent, RAG용 llm (stream option 미포함)
llm_default = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=1,
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
def get_bm25_retriever():
    """
    BM25Retriever 싱글톤 객체를 반환합니다.
    최초 호출 시 DB(PGVector)에 저장된 원본 텍스트를 로드하여
    인-메모리 키워드 인덱스를 생성합니다.
    """
    global _bm25_retriever_instance
    if _bm25_retriever_instance is not None:
        return _bm25_retriever_instance

    print("Initializing BM25Retriever (from PGVector's stored documents)...")
    
    # 1. PDF 로드 로직을 DB 쿼리 로직으로 대체
    sql_query = f"""
        SELECT document, cmetadata 
        FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}'
        );
    """
    
    splits_from_db = []
    try:
        # 1-2. SQLAlchemy engine을 사용하여 DB에서 직접 로드
        with engine.connect() as connection:
            results = connection.execute(text(sql_query))
            for row in results:
                # 1-3. (text, metadata)를 LangChain 'Document' 객체로 변환
                splits_from_db.append(Document(page_content=row[0], metadata=row[1]))
        
        print(f"BM25: Found and loaded {len(splits_from_db)} chunks from DB.")

    except Exception as e:
        print(f"❌ BM25 Error: DB에서 청크를 로드하는 데 실패했습니다: {e}")
        print("BM25 리트리버를 비활성화합니다.")
        _bm25_retriever_instance = BM25Retriever.from_documents([], k=3)
        return _bm25_retriever_instance

    # 2. DB에서 가져온 'splits_from_db' 리스트로 인덱스 생성
    if not splits_from_db:
        print("❌ BM25 Error: DB에 문서가 없어 'splits'가 비어있습니다.")
        _bm25_retriever_instance = BM25Retriever.from_documents([], k=3)
    else:
        _bm25_retriever_instance = BM25Retriever.from_documents(splits_from_db, k=3)
    
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
    vector_retriever = get_vector_store().as_retriever(search_kwargs={"k": 3})
    bm25_retriever = get_bm25_retriever()
    
    # 2. 앙상블 리트리버 생성
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    
    # 3. 압축 필터 생성 (임베더 싱글톤 사용)
    cached_embedder = get_cached_embedder()
    
    redundant_filter = EmbeddingsRedundantFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.95
    )
    relevance_filter = EmbeddingsFilter(
        embeddings=cached_embedder,
        similarity_threshold=0.85 #테스트 결과에 따른 임계값 조정
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