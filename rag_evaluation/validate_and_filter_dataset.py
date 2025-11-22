import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

RAG_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset.csv")
OUTPUT_FILE_FILTERED = os.path.join(RAG_EVAL_DIR, "dataset", "english_testset_filtered.csv")

def validate_and_filter_dataset():
    """
    1. 생성된 데이터셋의 품질을 검증합니다.
    2. 답변이 없는 케이스를 필터링합니다.
    3. 필터링된 결과를 CSV로 저장합니다.
    """
    
    # 1. 데이터셋 로드
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 입력 파일이 없습니다: {INPUT_FILE}")
        print("   먼저 generate_dataset.py를 실행하세요.")
        return
    
    df = pd.read_csv(INPUT_FILE)
    print("=" * 60)
    print("📋 RAGAS 데이터셋 검증 및 필터링")
    print("=" * 60)
    
    # 2. 검증: 원본 데이터셋 분석
    print(f"\n📊 원본 데이터셋 분석")
    print("-" * 60)
    print(f"총 케이스: {len(df)}개")
    
    # 답변 없는 케이스 식별
    no_answer_mask = df['ground_truth'].str.contains(
        "The answer to given question is not present in context",
        case=False,
        na=False
    )
    no_answer_count = no_answer_mask.sum()
    no_answer_pct = (no_answer_count / len(df)) * 100
    
    print(f"❌ 답변 없는 케이스: {no_answer_count}개 ({no_answer_pct:.1f}%)")
    print(f"✅ 답변 있는 케이스: {len(df) - no_answer_count}개 ({100 - no_answer_pct:.1f}%)")
    
    # 진화 타입 분포
    print(f"\n📈 진화 타입별 분포:")
    evolution_dist = df['evolution_type'].value_counts()
    for evo_type, count in evolution_dist.items():
        pct = (count / len(df)) * 100
        print(f"  • {evo_type}: {count}개 ({pct:.1f}%)")
    
    # 답변 없는 케이스의 진화 타입 분포
    print(f"\n📉 답변 없는 케이스의 진화 타입 분포:")
    no_answer_evolution = df[no_answer_mask]['evolution_type'].value_counts()
    for evo_type, count in no_answer_evolution.items():
        pct = (count / no_answer_count) * 100
        print(f"  • {evo_type}: {count}개 ({pct:.1f}%)")
    
    # 3. 필터링: 답변 없는 케이스 제거
    print(f"\n🔄 필터링 진행 중...")
    df_filtered = df[~no_answer_mask].reset_index(drop=True)
    
    print(f"✅ 필터링 완료")
    print(f"  제거된 케이스: {no_answer_count}개")
    print(f"  남은 케이스: {len(df_filtered)}개")
    
    # 필터링된 데이터의 진화 타입 분포
    print(f"\n📊 필터링된 데이터셋의 진화 타입 분포:")
    filtered_evolution = df_filtered['evolution_type'].value_counts()
    for evo_type, count in filtered_evolution.items():
        pct = (count / len(df_filtered)) * 100
        print(f"  • {evo_type}: {count}개 ({pct:.1f}%)")
    
    # 4. 필터링된 데이터 저장
    print(f"\n💾 필터링된 데이터셋 저장 중...")
    df_filtered.to_csv(OUTPUT_FILE_FILTERED, index=False)
    print(f"✅ 저장 완료: {OUTPUT_FILE_FILTERED}")
    
    # 5. 권장사항
    print(f"\n💡 권장사항:")
    print("-" * 60)
    
    if len(df_filtered) < 10:
        print(f"⚠️  필터링 후 케이스가 {len(df_filtered)}개로 너무 적습니다.")
        print(f"   generate_dataset.py에서 TEST_SIZE를 더 늘려서 재생성하세요.")
    elif len(df_filtered) < 20:
        print(f"⚠️  필터링 후 케이스가 {len(df_filtered)}개입니다.")
        print(f"   더 많은 테스트 케이스를 원한다면 TEST_SIZE를 늘려 재생성하세요.")
    else:
        print(f"✅ 충분한 테스트 케이스가 확보되었습니다.")
        print(f"   다음 단계: translate_dataset.py를 실행하세요.")

if __name__ == "__main__":
    validate_and_filter_dataset()