import os
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, load_prompt
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# --- 설정 ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_EVAL_DIR = os.path.dirname(CURRENT_DIR)
INPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
OUTPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "korean_testset.csv")

# 번역을 위한 LLM (빠르고 저렴한 mini 모델 사용)
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 번역 프롬프트
try:
    translation_prompt = load_prompt("../../prompts/search_query_translation.yaml")
except Exception as e:
    print(f"⚠️ 프롬프트 로드 실패, 기본 프롬프트를 사용합니다: {e}")
    from langchain_core.prompts import ChatPromptTemplate
    translation_prompt = ChatPromptTemplate.from_template(
        "Translate the following to natural Korean: {question}"
    )

chain = translation_prompt | llm

def translate_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일을 찾을 수 없습니다: {INPUT_FILE}")
        return

    print("📂 영어 데이터셋 로드 중...")
    df = pd.read_csv(INPUT_FILE)
    
    # 결과 저장을 위한 리스트
    translated_rows = []
    
    print(f"🚀 총 {len(df)}개의 데이터 번역 시작...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. 번역 실행
            response = chain.invoke({
                "user_input": row['user_input'],
                "reference": row['reference']
            })
            
            # 2. JSON 파싱 (Markdown 코드블록 제거 처리)
            content = response.content.replace("```json", "").replace("```", "").strip()
            translated_data = json.loads(content)
            
            # 3. 기존 데이터에 번역된 내용 덮어쓰기
            new_row = row.copy()
            new_row['user_input'] = translated_data['user_input'] # 질문 번역
            new_row['reference'] = translated_data['reference']   # 정답 번역
            # corpus(참고 문단)는 원래 한국어였으므로 그대로 둠
            
            translated_rows.append(new_row)
            
        except Exception as e:
            print(f"⚠️ Row {index} 번역 실패: {e}")
            # 실패 시 원본 유지
            translated_rows.append(row)

    # 4. 저장
    translated_df = pd.DataFrame(translated_rows)
    translated_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n✅ 번역 완료!")
    print(f"💾 저장 경로: {OUTPUT_FILE}")
    print("\n[번역 결과 미리보기]")
    print(translated_df[['user_input', 'reference']].head(2))

if __name__ == "__main__":
    translate_dataset()