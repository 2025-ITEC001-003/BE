import pandas as pd
import os

original_file_name = './data/기상청41_단기예보 조회서비스_오픈API활용가이드_격자_위경도(2411).xlsx'
output_dir = 'data'
output_file_name = 'kma_jeju_grid_info.csv'
output_file_path = os.path.join(output_dir, output_file_name)

COL_LEVEL_1 = '1단계' # '제주특별자치도'가 있는 컬럼
COL_LEVEL_2 = '2단계' # '제주시', '서귀포시'가 있는 컬럼
COL_LEVEL_3 = '3단계' # '애월읍', '한림읍' 등이 있는 컬럼
COL_LAT = '위도(시)' # 위도
COL_LON = '경도(시)' # 경도

FILTER_VALUE = '제주특별자치도'

try:
    print(f"'{original_file_name}' 엑셀 파일 읽기 시도...")
    
    df = pd.read_excel(original_file_name)
    
    print(f"파일 로드 성공. 총 {len(df)}개 행.")
    print("로드된 데이터 샘플 (처음 3개 행):")
    print(df.head(3))
    print("\n컬럼 목록:")
    print(df.columns.tolist())

    # --- '제주특별자치도' 데이터만 필터링 ---
    print(f"\n'{COL_LEVEL_1}' 컬럼에서 '{FILTER_VALUE}' 값으로 필터링 중...")
    
    if COL_LEVEL_1 not in df.columns:
        print(f"❌ [오류] '{COL_LEVEL_1}' 컬럼이 파일에 없습니다.")
        print(f"(코드 상단의 'COL_LEVEL_1' 변수를 실제 컬럼명으로 수정하세요.)")
        
    else:
        jeju_df = df[df[COL_LEVEL_1] == FILTER_VALUE].copy()

        if jeju_df.empty:
            print(f"❌ [오류] '{FILTER_VALUE}'에 해당하는 데이터를 찾을 수 없습니다.")
            print(f"(컬럼명 '{COL_LEVEL_1}' 또는 필터 값 '{FILTER_VALUE}'가 정확한지 확인하세요.)")
        else:
            print(f"필터링 완료. {len(jeju_df)}개의 제주도 데이터 추출.")

            # --- (선택) 필요한 컬럼만 선택 ---
            required_cols = [COL_LEVEL_1, COL_LEVEL_2, COL_LEVEL_3, COL_LAT, COL_LON]
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                 print(f"❌ [오류] 엑셀 파일에 필요한 컬럼이 없습니다: {missing_cols}")
                 print(f"(코드 상단의 'COL_' 변수들을 실제 컬럼명으로 수정하세요.)")
            else:
                jeju_df_final = jeju_df[required_cols]

                # --- 'data' 디렉토리 생성 ---
                if not os.path.exists(output_dir):
                    print(f"'{output_dir}' 디렉토리 생성...")
                    os.makedirs(output_dir)

                # --- 필터링된 파일 저장 (CSV로) ---
                jeju_df_final.to_csv(output_file_path, index=False, encoding='utf-8')
                
                print(f"\n✅ 성공! '{output_file_path}' (CSV)에 파일이 저장되었습니다.")

except FileNotFoundError:
    print(f"❌ [오류] 원본 파일 '{original_file_name}'을 찾을 수 없습니다.")
except ImportError:
    print(f"❌ [오류] 'openpyxl' 라이브러리가 필요합니다.")
    print(f"(가상 환경에 'pip install openpyxl'을 실행하거나,")
    print(f"'pyproject.toml'에 'openpyxl'을 추가하고 'poetry install'을 실행하세요.)")
except KeyError as e:
    print(f"❌ [오류] 코드에 정의된 컬럼명(예: {e})이 엑셀 파일에 없습니다.")
    print(f"(코드 상단의 'COL_' 변수들을 실제 파일의 컬럼명과 일치시키세요.)")
except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")