import requests
from typing import List

def get_korean_stopwords() -> List[str]:
    """GitHub URL로부터 한국어 불용어 목록을 불러옵니다."""
    try:
        # GitHub URL로부터 'korean_stopwords.txt' 파일을 읽어 한국어 불용어를 불러옵니다.
        file_url = "https://raw.githubusercontent.com/teddylee777/langchain-teddynote/main/assets/korean_stopwords.txt"
        
        # 불용어 파일을 인터넷에서 가져옵니다.
        response = requests.get(file_url, timeout=5)
        response.raise_for_status() # HTTP 요청이 실패하면 예외를 발생시킵니다.
        
        # 응답으로부터 텍스트 데이터를 받아옵니다.
        stopwords_data = response.text

        # 텍스트 데이터를 줄 단위로 분리합니다.
        stopwords = stopwords_data.splitlines()

        # 각 줄에서 여분의 공백 문자(개행 문자 등)를 제거합니다.
        return [word.strip() for word in stopwords]
    except Exception as e:
        print(f"❌ 불용어 로드 오류: {e}. 기본 빈 리스트를 사용합니다.")
        return []