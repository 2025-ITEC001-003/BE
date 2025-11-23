import os
import time
import pandas as pd
from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

# 1. 설정
DATASET_NAME = f"Jeju_Tourism_QA_Set_-{time.strftime('%Y%m%d-%H%M%S')}"  # LangSmith에 저장될 데이터셋 이름
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_EVAL_DIR = os.path.dirname(CURRENT_DIR)
CSV_PATH = os.path.join(RAG_EVAL_DIR, "dataset", "korean_testset.csv")

def upload_dataset():
    client = Client()
    
    # CSV 읽기
    if not os.path.exists(CSV_PATH):
        print(f"❌ 파일이 없습니다: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    
    # 2. 데이터셋 생성 (이미 존재하면 건너뛰거나 덮어쓰기 로직 필요)
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"⚠️ 데이터셋 '{DATASET_NAME}'이 이미 존재합니다. 삭제 후 재생성하거나 이름을 변경하세요.")
        # client.delete_dataset(dataset_name=DATASET_NAME) # 필요시 주석 해제
        return

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="제주 관광 챗봇 RAG 평가용 데이터셋 (한국어)",
    )
    
    print(f"🚀 데이터셋 업로드 시작: {DATASET_NAME} (총 {len(df)}개)")

    # 3. 예제(Example) 추가
    inputs = []
    outputs = []
    
    for _, row in df.iterrows():
        # 입력 데이터 (질문)
        inputs.append({
            "question": row["question"]
        })
        
        # 출력 데이터 (정답, RAGAS에서 만든 ground_truth)
        # contexts도 평가에 필요하다면 inputs나 outputs에 포함시킬 수 있음
        outputs.append({
            "answer": row["ground_truth"] # LangChain 평가기는 보통 'answer' 키를 정답으로 봅니다.
        })

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id
    )
    
    print("✅ 업로드 완료! LangSmith 웹에서 데이터셋을 확인하세요.")

if __name__ == "__main__":
    upload_dataset()