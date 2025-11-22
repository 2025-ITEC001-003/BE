import pandas as pd
import os
from dotenv import load_dotenv
from rag_evaluation.translator import Translator 

load_dotenv()

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
RAG_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
OUTPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "korean_testset.csv")

def translate_dataset():
    if not DEEPL_API_KEY:
        print("❌ Deepl API 키(DEEPL_API_KEY)가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return
        
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일이 없습니다: {INPUT_FILE}. generate_dataset.py를 먼저 실행하세요.")
        return
        
    print(f"1. 데이터셋 로드 및 번역 시작...")
    df = pd.read_csv(INPUT_FILE)
    translator = Translator(DEEPL_API_KEY, source_lang='EN', target_lang='KO')
    
    cols_to_translate = ['question', 'ground_truth']
    
    for col in cols_to_translate:
        print(f"2. '{col}' 컬럼 번역 중...")
        texts = df[col].tolist()
        translated_texts = []
        
        for i, text in enumerate(texts):
            try:
                translated_text = translator(text)
                translated_texts.append(translated_text)
            except Exception as e:
                print(f"   ⚠️  {i+1}번째 텍스트 번역 실패: {e}. 원문 사용.")
                translated_texts.append(text)  # 실패 시 원문 유지
        
        df[col] = translated_texts

    # 3. CSV로 저장
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ 한국어 데이터셋 번역 완료: {OUTPUT_FILE}")
    print("⚠️ 'contexts' 컬럼은 RAGAS의 Faithfulness 평가를 위해 영어 원문을 그대로 유지합니다.")
    
if __name__ == "__main__":
    translate_dataset()