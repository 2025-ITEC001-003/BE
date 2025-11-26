import os
import glob
import tiktoken
from sqlalchemy import text
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from typing import cast, Dict, Any
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_cloud_services.parse.utils import ResultType

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

def token_length(text: str) -> int:
    enc = tiktoken.encoding_for_model("gpt-4o")  # 사용하는 모델로 설정
    return len(enc.encode(text))

def create_korean_text_splitter():
    """
    한국어에 최적화된 텍스트 분할기 생성
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        # 한국어 문장 경계 고려
        separators=[
            "\n\n",              # 문단 구분
            "\n",                # 줄바꿈
            ". ",                # 마침표 + 공백 (영어 문장)
            "。 ",               # 일본어/중국어 마침표
            "? ",                # 물음표
            "! ",                # 느낌표
            "； ",               # 세미콜론
            "，",                # 쉼표
            " ",                 # 공백
            "",                  # 최후의 수단
        ],
        # 길이 계산 함수 (토큰 수 기반으로 변경 가능)
        length_function=token_length,
        # 메타데이터에 청크 위치 정보 추가
        keep_separator=True,
    )

def validate_chunk_quality(chunks: list[Document]) -> tuple[bool, str]:
    """
    생성된 청크의 품질을 검증
    
    Returns:
        (is_valid, message)
    """
    if not chunks:
        return False, "청크가 생성되지 않았습니다"
    
    # 평균 청크 크기 확인
    avg_size = sum(token_length(c.page_content) for c in chunks) / len(chunks)
    
    if avg_size < 100:
        return False, f"청크 크기가 너무 작습니다 (평균: {avg_size:.0f}토큰)"
    
    if avg_size > 2000:
        return False, f"청크 크기가 너무 큽니다 (평균: {avg_size:.0f}토큰)"
    
    # 빈 청크 확인
    empty_chunks = sum(1 for c in chunks if not c.page_content.strip())
    if empty_chunks > 0:
        return False, f"{empty_chunks}개의 빈 청크가 발견되었습니다"
    
    return True, f"✅ 청크 품질 검증 완료 (총 {len(chunks)}개, 평균 {avg_size:.0f}토큰)"
    
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

# 텍스트 분할기
text_splitter = create_korean_text_splitter()

# 벡터 스토어 연결
vector_store = PGVector(
    collection_name=COLLECTION_NAME,
    connection=DATABASE_URL,
    embeddings=get_cached_embedder(),
    pre_delete_collection=False
)

# LlamaParse는 실제로 필요할 때만 초기화(처음 파싱이 필요한 파일 만)
parser = None
file_extractor = None

# 파일 단위 처리: 이미 처리된 Markdown이 있으면 그것을 사용하고, 없으면 LlamaParse로 PDF 파싱
for pdf_path in files_to_process:
    raw_filename = os.path.basename(pdf_path)
    title = raw_filename.replace('.pdf', '')
    print(f"\n--- Processing file: {raw_filename} ---")

    # 기존 데이터 삭제
    delete_existing_file_data(pdf_path, COLLECTION_NAME)

    # 먼저 이미 로컬에 저장된 Markdown이 있는지 확인
    md_save_path = os.path.join(PROCESSED_MD_DIR, f"{title}.md")
    full_text = None

    if os.path.exists(md_save_path):
        try:
            with open(md_save_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            print(f"  📄 기존 Markdown 사용: {md_save_path}")
        except Exception as e:
            print(f"  ❌ 기존 Markdown 읽기 실패, 파싱으로 대체: {e}")

    # Markdown이 없으면 LlamaParse로 PDF 파싱
    if not full_text:
        # lazy init parser
        if parser is None:
            parser = LlamaParse(
                api_key=os.getenv("LLAMA_CLOUD_API_KEY", ""),
                parse_mode="parse_page_with_agent",
                model="openai-gpt-4-1-mini",
                high_res_ocr=True,
                adaptive_long_table=True,
                outlined_table_extraction=True,
                precise_bounding_box=True,
                result_type=ResultType.MD,
                num_workers=8,
                verbose=True,
                language="ko"
            )
            file_extractor = cast(Dict[str, Any], {".pdf": parser})

        print(f"  🔎 LlamaParse로 PDF 파싱 시작: {raw_filename}")
        reader = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor)
        try:
            docs_iter = reader.iter_data()
            docs_in_file = next(docs_iter, [])
        except Exception as e:
            print(f"  ❌ LlamaParse 파싱 실패: {e}")
            docs_in_file = []

        if not docs_in_file:
            print("  ⚠️ 추출된 페이지가 없습니다. 해당 파일 건너뜀.")
            continue

        # LlamaParse로부터 반환된 각 페이지의 텍스트를 병합
        parts = []
        for doc in docs_in_file:
            # support different doc types (llama vs langchain)
            text_part = getattr(doc, 'text', None) or getattr(doc, 'page_content', '')
            parts.append(text_part)
        full_text = "\n\n".join(parts)

        # Markdown 원본을 로컬에 저장
        try:
            with open(md_save_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            print(f"  📝 Markdown 원본 저장 완료: {md_save_path}")
        except Exception as e:
            print(f"  ❌ Markdown 원본 저장 실패: {e}")

    # 메타데이터 구성 및 LangChain Document 생성
    file_metadata = {
        "source": pdf_path,
        "title": title,
    }

    full_doc = Document(
        page_content=full_text or "",
        metadata=file_metadata
    )

    # 청크 분할
    final_splits = text_splitter.split_documents([full_doc])

    # 청크 품질 검증
    is_valid, message = validate_chunk_quality(final_splits)
    print(f"  {message}")
    if not is_valid:
        print(f"  ❌ 청크 품질 검증 실패. 처리 중단.")
        continue

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