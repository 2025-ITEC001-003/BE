import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core import DATABASE_URL, get_cached_embedder, COLLECTION_NAME

# 문서 로드
doc_paths = [
    "data/tourism_docs/제주공식관광가이드북.pdf",
    "data/tourism_docs/제주한류컨텐츠.pdf",
    "data/tourism_docs/제주향토음식.pdf",
    "data/tourism_docs/제주오름.pdf",
]

all_docs = []
for path in doc_paths:
    if os.path.exists(path):
        print(f"로딩 중: {path}")
        loader = UnstructuredPDFLoader(
            path, 
            languages=["kor"]
        )
        all_docs.extend(loader.load())
    else:
        print(f"경고: {path} 파일을 찾을 수 없습니다.")

print(f"총 {len(all_docs)}개 페이지 로드 완료.")

# 청크 나누기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
splits = text_splitter.split_documents(all_docs)

print(f"총 {len(splits)}개 청크로 분할 완료.")

# 임베딩 및 벡터스토어 저장
print("core로부터 캐시 임베더 가져오는 중...")
cached_embedder = get_cached_embedder()

print("임베딩 및 PostgreSQL(pgvector)에 저장 시작...")
PGVector.from_documents(
    documents=splits,
    embedding=cached_embedder,
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    pre_delete_collection=True # 기존 테이블을 삭제하고 새로 만듭니다.
)

print(f"성공: '{COLLECTION_NAME}' 컬렉션에 데이터 저장 완료.")