import os
import glob
from sqlalchemy import text
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core import DATABASE_URL, get_cached_embedder, COLLECTION_NAME, engine

# 1. 경로 설정
CURRENT_FILE_PATH = os.path.abspath(__file__)
UTIL_DIR = os.path.dirname(CURRENT_FILE_PATH)
PROJECT_ROOT = os.path.dirname(UTIL_DIR)
DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "tourism_docs")
PROCESSED_MD_DIR = os.path.join(PROJECT_ROOT, "data", "processed_md_originals")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "processed_chunks_results")
os.makedirs(PROCESSED_MD_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

def get_processed_files(collection_name):
    sql = text(f"""
        SELECT DISTINCT cmetadata->>'source' as source
        FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :name
        )
    """)
    processed_files = set()
    try:
        with engine.connect() as conn:
            result = conn.execute(sql, {"name": collection_name})
            for row in result:
                if row[0]: processed_files.add(row[0])
    except Exception as e:
        print(f"⚠️ DB 조회 중 오류 (첫 실행이면 무시): {e}")
    return processed_files

def delete_existing_file_data(file_path, collection_name):
    sql = text(f"""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :name
        )
        AND cmetadata->>'source' = :path
    """)
    with engine.connect() as conn:
        conn.execute(sql, {"name": collection_name, "path": file_path})
        conn.commit()
    print(f"  🗑️ 기존 데이터 삭제 완료: {os.path.basename(file_path)}")
    

# 1. 파일 필터링
all_pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
processed_files = get_processed_files(COLLECTION_NAME)

files_to_process = []
print(f"📊 총 파일: {len(all_pdf_files)}개 / DB 저장됨: {len(processed_files)}개")

for pdf_path in all_pdf_files:
    if pdf_path in processed_files:
        print(f"  - [건너뜀] 이미 최신: {os.path.basename(pdf_path)}")
    else:
        print(f"  - [대기열] 신규/변경: {os.path.basename(pdf_path)}")
        files_to_process.append(pdf_path)

if not files_to_process:
    print("✅ 처리할 신규 파일이 없습니다. 종료합니다.")
    exit()

# 2. LlamaParse 및 Reader 초기화
print(f"\n🚀 {len(files_to_process)}개 파일 처리를 시작합니다...")

parser = LlamaParse(
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    parse_mode="parse_page_with_agent",
    model="openai-gpt-4-1-mini",
    high_res_ocr=True,
    adaptive_long_table=True,
    outlined_table_extraction=True,
    output_tables_as_HTML=True,
    precise_bounding_box=True,
    result_type="markdown", # LlamaParse 결과 유형
    num_workers=8,
    verbose=True,
    language="ko"
)

file_extractor = {".pdf": parser}
reader = SimpleDirectoryReader(
    input_files=files_to_process,
    file_extractor=file_extractor
)

# 3. Recursive 분할기
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

# 4. 벡터 스토어 연결
vector_store = PGVector(
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    embeddings=get_cached_embedder(),
    pre_delete_collection=False
)

# 5. 파일 단위 처리 (병합 -> Recursive 분할 -> 저장 및 로컬 저장)
for i, docs_in_file in enumerate(reader.iter_data()):
    if not docs_in_file:
        continue

    first_doc_meta = docs_in_file[0].metadata
    file_path = first_doc_meta.get("file_path", "")
    raw_filename = first_doc_meta.get("file_name", "unknown")
    title = raw_filename.replace(".pdf", "")
    
    print(f"\n--- Processing file: {raw_filename} ({len(docs_in_file)} pages) ---")

    # 기존 데이터 삭제
    delete_existing_file_data(file_path, COLLECTION_NAME)

    # 텍스트 병합 (LlamaParse의 Markdown 결과)
    full_text = "\n\n".join([doc.text for doc in docs_in_file])

    # LlamaParse Markdown 전체 결과 로컬 저장
    md_save_path = os.path.join(PROCESSED_MD_DIR, f"{title}.md")
    try:
        with open(md_save_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"  📝 Markdown 원본 저장 완료: {md_save_path}")
    except Exception as e:
        print(f"  ❌ Markdown 원본 저장 실패: {e}")


    # 메타데이터 구성 및 LangChain Document 생성
    file_metadata = {
        "source": file_path,
        "title": title,
    }
    
    full_doc = Document(
        page_content=full_text,
        metadata=file_metadata
    )
    
    # 청크 분할
    final_splits = text_splitter.split_documents([full_doc])
    
    # 청크 분할 결과 로컬 저장
    chunks_save_path = os.path.join(CHUNKS_DIR, f"{title}_chunks.md")
    chunk_separator = "\n\n---\n\n"
    
    if final_splits:
        try:
            with open(chunks_save_path, "w", encoding="utf-8") as f:
                for chunk_index, chunk in enumerate(final_splits):
                    f.write(f"## CHUNK {chunk_index + 1} (Size: {len(chunk.page_content)} bytes)\n")
                    f.write(chunk.page_content)
                    if chunk_index < len(final_splits) - 1:
                        f.write(chunk_separator)
            print(f"  ✂️ 청크 결과 저장 완료: {chunks_save_path} ({len(final_splits)} chunks)")
        except Exception as e:
            print(f"  ❌ 청크 결과 저장 실패: {e}")
            
        # DB 저장
        vector_store.add_documents(final_splits)
        print(f"  ✅ DB 저장 완료 ({len(final_splits)} chunks) - Title: {title}")
    else:
        print("  ⚠️ 경고: 추출된 텍스트가 없습니다. DB 저장 건너뜀.")

print("\n🎉 모든 작업이 완료되었습니다.")