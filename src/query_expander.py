import google.generativeai as genai
import os

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

def expand_query(query: str) -> str:
    if not query.strip():
        return query
    
    response = model.generate_content(f"""다음 카페 검색어를 CLIP 이미지 검색에 최적화된 영어 프롬프트로 확장해줘.
검색어: {query}
규칙:
- 영어로만 출력
- 카페 인테리어/분위기 관련 키워드 위주
- 쉼표로 구분된 10개 이내 키워드
- 다른 설명 없이 키워드만 출력""")
    
    return f"{query}, {response.text.strip()}"