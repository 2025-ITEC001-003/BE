import pandas as pd
import os
from dotenv import load_dotenv
from rag_evaluation.translator import Translator 

load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
RAG_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset_filtered.csv") # 파일명 확인 필요
OUTPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "korean_testset.csv")

def translate_dataset():
    if not DEEPL_API_KEY:
        print("❌ Deepl API 키(DEEPL_API_KEY)가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
        
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일이 없습니다: {INPUT_FILE}. generate_dataset.py를 먼저 실행하세요.")
        return
        
    print(f"1. 데이터셋 로드 ({INPUT_FILE})")
    df = pd.read_csv(INPUT_FILE)
    
    # DeepL Translator 초기화
    translator = Translator(DEEPL_API_KEY, source_lang='EN', target_lang='KO')
    
    cols_to_translate = ['question', 'ground_truth']
    
    for col in cols_to_translate:
        if col not in df.columns:
            print(f"⚠️ 경고: '{col}' 컬럼이 데이터셋에 없습니다. 건너뜁니다.")
            continue

        print(f"2. '{col}' 컬럼 번역 중 (Batch 처리)...")
        original_texts = df[col].tolist()
        
        try:
            translated_texts = translator(original_texts)
            df[col] = translated_texts
            print(f"   -> {len(translated_texts)}개 문장 번역 완료")
            
        except Exception as e:
            print(f"   ❌ '{col}' 컬럼 일괄 번역 실패: {e}")
            print("   -> (대안) 한 건씩 번역을 시도하거나 원본을 유지합니다.")
            # 실패 시 비상용으로 루프 방식 사용 가능 (여기선 생략)

    # 3. CSV로 저장
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print(f"✅ 한국어 데이터셋 번역 완료: {OUTPUT_FILE}")
    print("참고: 'contexts' 컬럼은 평가 정확도를 위해 영어 원문을 유지했습니다.")
    
if __name__ == "__main__":
    translate_dataset()