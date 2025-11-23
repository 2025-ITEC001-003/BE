import deepl
from typing import Union, List

class Translator:
    def __init__(self, api_key, source_lang='EN', target_lang='KO'):
        self.translator = deepl.Translator(api_key)
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        # DeepL은 리스트 입력을 지원
        results = self.translator.translate_text(
            text, 
            source_lang=self.source_lang, 
            target_lang=self.target_lang
        )
        
        # 입력이 리스트였다면 결과 리스트 반환
        if isinstance(results, list):
            return [res.text for res in results]
        # 입력이 단일 문자열이었다면 단일 문자열 반환
        else:
            return results.text