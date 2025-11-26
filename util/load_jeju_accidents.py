import pandas as pd
from dotenv import load_dotenv
import os
from src.core import engine

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL이 .env 파일에 설정되지 않았습니다.")

CSV_FILE_PATH = "data/Jeju_Safety_Accidents_2025.csv" 
DB_TABLE_NAME = "jeju_accidents"

#  DB에 저장할 최종 컬럼 리스트
# FINAL_COLUMNS = [
#     'RPTP_NO', 'PTN_OCRN_TYPE_NM' , 'PTN_SYM_SE_NM', 'SRIL_ONCR_NM', 'TRFC_ACDNT_SE_NM', 
#     'ACDNT_INJR_NM', 'DCLR_YMD', 'SEASN_NM', 'DCLR_DOW,PTN_GNDR_NM','ACDNT_OCRN_LOT', 
#     'ACDNT_OCRN_LAT', 'HR_UNIT_ARTMP', 'HR_UNIT_RN', 'HR_UNIT_WSPD,HR_UNIT_WNDRCT', 'HR_UNIT_HUM,HR_UNIT_SNWFL'
# ]
FINAL_COLUMNS = [
    'SEASN_NM', 'ACDNT_OCRN_LOT', 
    'ACDNT_OCRN_LAT', 'DCLR_MM',
    'PTN_SYM_SE_NM', 'ACDNT_INJR_NM', 'DCLR_YR'
]

# LLM이 쿼리하기 좋은 '카테고리형' 컬럼 (NaN을 '알수없음' 등으로 처리)
CATEGORICAL_COLUMNS = [
    'SEASN_NM', 'TRFC_ACDNT_SE_NM', 'ETC_TRFC_ACDNT_NM', 'SRIL_ONCR_NM', 
    'PTN_OCRN_TYPE_NM', 'PTN_SYM_SE_NM', 'ACDNT_OCRN_PLC_NM', 
    'GRNDS_SGG_NM', 'ACDNT_INJR_NM'
]

def load_data_to_postgres():
    """CSV를 읽어 필터링 후 PostgreSQL에 적재합니다."""
    print(f"'{CSV_FILE_PATH}' 파일 읽기 시도...")
    try:
        df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("utf-8 실패, cp949로 재시도...")
        df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')
    print("파일 읽기 성공.")

    # 컬럼 필터링
    df_filtered = df[FINAL_COLUMNS].copy()
    
    # 데이터 전처리: 카테고리 컬럼의 NaN 값을 '알수없음'으로 변경
    # (LLM이 'WHERE PTN_SYM_SE_NM IS NULL' 대신 'WHERE PTN_SYM_SE_NM = '알수없음''으로 쿼리 가능)
    for col in CATEGORICAL_COLUMNS:
        if col in df_filtered.columns:
            df_filtered[col] = df_filtered[col].fillna("알수없음")
            
    print(f"컬럼 필터링 및 전처리 완료. {len(df_filtered)}개 행")

    # PostgreSQL에 연결 및 데이터 적재
    try:
        print("DB 엔진 생성 완료. 데이터 적재 시작...")
        
        # 테이블을 통째로 덮어쓰기 (가장 간단한 업데이트 방식)
        df_filtered.to_sql(DB_TABLE_NAME, engine, if_exists='replace', index=False)
        
        print(f"성공: 데이터가 PostgreSQL '{DB_TABLE_NAME}' 테이블에 성공적으로 적재되었습니다.")
    
    except Exception as e:
        print(f"DB 적재 중 오류 발생: {e}")

if __name__ == "__main__":
    load_data_to_postgres()