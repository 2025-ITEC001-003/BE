import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

# .env 파일 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# OpenAI API 키를 환경변수에 명시적으로 설정 (LangChain이 사용)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENWEATHER_API_KEY"] = OPENWEATHER_API_KEY

# llm
llm = ChatOpenAI(model="gpt-5-mini", temperature=1, api_key=OPENAI_API_KEY)

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
#TODO: 실제 서비스에서는 이 로드/분할 과정을 캐시해야 합니다.
_bm25_retriever_instance = None
def get_bm25_retriever():
    """
    BM25Retriever 싱글톤 객체를 반환합니다.
    최초 호출 시 PDF 문서를 로드하고 인-메모리 인덱스를 생성합니다.
    """
    global _bm25_retriever_instance
    if _bm25_retriever_instance is not None:
        return _bm25_retriever_instance

    print("Initializing BM25Retriever (Loading PDF docs)...")
    
    # 1. 'src/core.py' 파일 기준으로 'data' 폴더 절대 경로 계산
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "tourism_docs")

    doc_paths = [
        os.path.join(DOCS_DIR, "제주공식관광가이드북.pdf"),
        os.path.join(DOCS_DIR, "제주한류컨텐츠.pdf"),
        os.path.join(DOCS_DIR, "제주향토음식.pdf"),
        os.path.join(DOCS_DIR, "제주오름.pdf"),
    ]
    
    all_docs = []
    for path in doc_paths:
        if os.path.exists(path):
            loader = UnstructuredPDFLoader(path, languages=["kor"])
            all_docs.extend(loader.load())
        else:
            print(f"BM25 Warning: {path} 파일을 찾을 수 없습니다.")

    # 2. 청크 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)

    if not splits:
        print("❌ BM25 Error: 로드된 문서가 없어 'splits'가 비어있습니다.")
        # 빈 리스트로 임시 초기화 (오류 방지)
        _bm25_retriever_instance = BM25Retriever.from_documents([], k=3)
    else:
        # 3. 인-메모리 인덱스 생성 및 저장
        _bm25_retriever_instance = BM25Retriever.from_documents(splits, k=3)
    
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
        similarity_threshold=0.7 
    )
    
    # 4. 압축 파이프라인 생성
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, relevance_filter]
    )
    
    # 5. 최종 압축 리트리버 생성 및 저장
    _compression_retriever_instance = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=ensemble_retriever
    )
    
    print("Compression Retriever initialization complete.")
    return _compression_retriever_instance