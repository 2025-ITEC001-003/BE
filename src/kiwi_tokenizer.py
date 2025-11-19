import string
from typing import List, Optional
from kiwipiepy import Kiwi
import nltk

class KiwiBM25Tokenizer:
    def __init__(self, stop_words: Optional[List[str]] = None):
        self._setup_nltk()
        self._stop_words = set(stop_words) if stop_words else set()
        self._punctuation = set(string.punctuation)
        self._tokenizer = self._initialize_tokenizer()

    @staticmethod
    def _initialize_tokenizer() -> Kiwi:
        # Kiwi 객체는 직렬화할 수 없으므로, 필요 시 다시 초기화합니다.
        # (BM25Retriever 직렬화/역직렬화에 대응)
        return Kiwi()

    @staticmethod
    def _tokenize(tokenizer: Kiwi, text: str) -> List[str]:
        # 형태소 분석 결과를 사용하며, 여기서는 '형태소 원형(form)'을 토큰으로 사용합니다.
        return [token.form for token in tokenizer.tokenize(text)]

    @staticmethod
    def _setup_nltk() -> None:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def __call__(self, text: str) -> List[str]:
        # 토큰화 및 필터링
        tokens = self._tokenize(self._tokenizer, text)
        return [
            word.lower()
            for word in tokens
            if word not in self._punctuation and word not in self._stop_words
        ]

    def __getstate__(self):
        # 직렬화 시 _tokenizer 객체를 제외
        state = self.__dict__.copy()
        del state["_tokenizer"]
        return state

    def __setstate__(self, state):
        # 역직렬화 시 _tokenizer 객체를 다시 초기화
        self.__dict__.update(state)
        self._tokenizer = self._initialize_tokenizer()
