import pandas as pd
import os

# jesafe/src/data_loader.py
CURRENT_FILE_PATH = os.path.abspath(__file__)
# jesafe/src/
SRC_DIR = os.path.dirname(CURRENT_FILE_PATH)
# jesafe/
PROJECT_ROOT = os.path.dirname(SRC_DIR)
# 'data/kma_jeju_grid_info.csv' 절대 경로
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'kma_jeju_grid_info.csv')

_jeju_coords_db = None

def _load_jeju_coords():
    global _jeju_coords_db
    if _jeju_coords_db is not None:
        return

    try:
        df = pd.read_csv(
            CSV_PATH,
            encoding='utf-8', 
            usecols=['1단계', '2단계', '3단계', '위도(시)', '경도(시)'], 
        )
        
        df = df[df['1단계'] == '제주특별자치도'].copy()
        df['location_name'] = df['3단계'].fillna(df['2단계'])
        
        df.rename(columns={
            '위도(시)': 'lat',
            '경도(시)': 'lon'
        }, inplace=True)
        
        _jeju_coords_db = df.set_index('location_name')[['lat', 'lon']]
        print(f"✅ [data_loader] 제주도 (KMA 격자) DB 로드 완료 (경로: {CSV_PATH})")
        
    except FileNotFoundError:
        print(f"❌ [data_loader] 오류: 좌표 CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
        _jeju_coords_db = pd.DataFrame()
    except Exception as e:
        print(f"❌ [data_loader] 오류: 좌표 CSV 파일 로드 실패: {e}")
        _jeju_coords_db = pd.DataFrame()

def get_jeju_coordinates(location_name: str) -> dict:
    if _jeju_coords_db is None:
        _load_jeju_coords()

    if _jeju_coords_db.empty:
        raise ValueError("좌표 DB가 비어있거나 로드에 실패했습니다.")

    try:
        result = _jeju_coords_db.loc[location_name]
        # .item()을 사용하여 int/float으로 변환 (Pandas/Numpy 자료형 방지)
        return {
            "lat": result.lat.item(), 
            "lon": result.lon.item()
        }
    except KeyError:
        raise ValueError(f"'{location_name}'에 대한 좌표를 CSV에서 찾을 수 없습니다.")