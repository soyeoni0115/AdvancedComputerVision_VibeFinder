# llm을 통해 사용자가 입력한 검색어를 CLIP 이미지 검색에 최적화된 영어 프롬프트로 확장해주는 코드
import google.generativeai as genai
import streamlit as st
import os

API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# API 실패 시 최소한의 폴백용 매핑
FALLBACK_MAP = {
    "아늑한": "cozy cafe interior, warm lighting, wooden furniture, comfortable atmosphere",
    "조용한": "quiet cafe interior, calm atmosphere, minimal design, good for reading",
    "힙한": "trendy cafe interior, industrial design, urban atmosphere, stylish decor",
    "감성적인": "aesthetic cafe interior, artistic mood, soft lighting, cozy atmosphere",
    "공부": "study friendly cafe, quiet atmosphere, spacious seating, work friendly",
    "화이트": "bright white cafe interior, clean minimal design, natural lighting",
    "우드": "wooden furniture, warm cafe interior, natural wood texture, cozy",
    "채광": "large windows, natural lighting, bright sunny cafe interior",
    "모던": "modern cafe interior, minimalist design, clean aesthetic",
    "빈티지": "vintage cafe interior, retro furniture, antique decor, warm atmosphere",
    "어두운": "dark moody cafe interior, dim lighting, intimate atmosphere",
    "넓은": "spacious cafe interior, large seating area, open space",
}


def fallback_expand(query: str) -> str:
    expanded = [query]
    for key, prompt in FALLBACK_MAP.items():
        if key in query:
            expanded.append(prompt)
    if len(expanded) == 1:
        expanded.append("cafe interior, coffee shop atmosphere")
    return ", ".join(expanded)

@st.cache_data(ttl=86400, show_spinner=False)
def expand_query(query: str) -> str:
    query = query.strip()
    if not query:
        return query

    try:
        response = model.generate_content(f"""다음 카페 검색어를 CLIP 이미지 검색에 최적화된 영어 프롬프트로 확장해줘.
검색어: {query}
규칙:
- 영어로만 출력
- 카페 인테리어/분위기 관련 키워드 위주
- 쉼표로 구분된 10개 이내 키워드
- 다른 설명 없이 키워드만 출력""")

        expanded = response.text.strip()
        if not expanded:
            return fallback_expand(query)

        return f"{query}, {expanded}"

    except Exception as e:
        print(f"[query_expander] API 호출 실패, 폴백 매핑 사용: {e}")
        return fallback_expand(query)