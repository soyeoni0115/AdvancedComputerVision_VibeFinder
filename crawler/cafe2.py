import os
import sys
from urllib.parse import quote
from dotenv import load_dotenv
import requests

# =============================================================
# [경로 해결] 현재 crawler 폴더 위치를 기준으로 최상위 폴더를 경로에 추가
# =============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 상위 폴더 경로가 추가된 후 DB 모듈 임포트 진행
from src.database.postgres_new2 import insert_cafe

# =========================
# .env 로드 (상위 폴더의 .env 파일 타겟팅)
# =========================
load_dotenv(os.path.join(parent_dir, ".env"))

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
}


# =========================
# 네이버 지역 검색 API
# [안전성 수정] URL 결합 에러 방지를 위해 params 구조로 변경
# =========================
def search_cafe_places(query, display=20, start=1):
    url = "https://openapi.naver.com/v1/search/local.json"

    # 네이버 공식 문서 규격에 맞춰 정합성 확보 (sort=comment 제외)
    params = {"query": query, "display": display, "start": start, "sort": "sim"}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200:
        return response.json().get("items", [])

    print(
        f"검색 실패 | 상태 코드: {response.status_code} | 사유: {response.text}"
    )
    return []


# =========================
# 네이버 지도 링크 생성
# =========================
def generate_naver_map_link(cafe_name):
    encoded_name = quote(cafe_name)
    return f"https://map.naver.com/p/search/{encoded_name}"


# =========================
# 제외 키워드
# =========================
EXCLUDE_KEYWORDS = [
    "스터디",
    "프린트",
    "만화",
    "보드게임",
    "룸카페",
    "멀티방",
    "PC",
    "피시",
    "애견",
    "키즈",
    "메가커피",
    "컴포즈",
    "빽다방",
    "스타벅스",
    "집무실",
    "스파크플러스",
    "라운지",
    "공유오피스",
    "오피스",
    "렌탈",
    "스튜디오",
    "파티룸",
    "투썸플레이스",
    "메가",
    "베스킨라빈스",
    "파리바게뜨"
]


# =========================
# 메인 실행
# =========================
if __name__ == "__main__":
    print("성수 / 서울숲 / 뚝섬 카페 수집 시작")

    keywords = [
        
        "성수동 카이막",          # 요즘 감성 인테리어 카페 많음
        "성수동 츄러스",          # 유럽풍 인테리어 위주
        "성수동 휘낭시에",        # 구움과자 세분화
        "서울숲 도넛 카페",       # 키치하고 아기자기한 무드
        "뚝섬 크루아상 카페",
        
        # 2. 음료 종류 세분화
        "성수동 밀크티 맛집",
        "서울숲 드립커피",
        
        # 1. '어두운 / 바 분위기' 태그 정밀 타격
        "성수동 에스프레소바 위스키",
        "성수 위스키앤커피",
        "뚝섬 심야카페",
        
        # 2. '럭셔리 / 고급스러운' 태그 정밀 타격
        "성수동 플래그십 카페",
        "서울숲 디저트 부티크",
        "성수 파인다이닝 디저트",
        
        # 3. '이국적인 / 휴양지풍' 태그 정밀 타격
        "성수동 유럽풍 카페",
        "서울숲 라탄 인테리어 카페",
        "성수 에스프레소바 테라스",
        
        # 4. '빈티지 / 레트로' 및 고즈넉한 태그 정밀 타격
        "성수동 한옥 카페",
        "뚝섬 적벽돌 카페",
        
        # 5. '몽환적인 / 감각적인 / 키치한' 태그 정밀 타격
        "성수 미디어아트 카페",
        "성수동 와인바 겸 카페",
        "서울숲 쇼룸 카페"
    ]

    saved_cafes = set()
    total_saved = 0

    # 키워드 순회
    for keyword in keywords:
        print(f"\n====================\n[{keyword}]\n====================")

        # 페이지네이션 최대 80개 확보 시도
        for start in [1, 21, 41, 61]:
            print(f"\n페이지 시작: {start}")

            cafes = search_cafe_places(keyword, display=20, start=start)

            # 결과 없으면 해당 키워드 종료
            if not cafes:
                break

            for cafe in cafes:
                clean_name = (
                    cafe["title"]
                    .replace("<b>", "")
                    .replace("</b>", "")
                    .strip()
                )
                road_address = cafe.get("roadAddress", "")
                category = cafe.get("category", "")

                # 성동구 필터
                if "성동구" not in road_address:
                    continue

                # 카페 카테고리 필터
                if "카페" not in category:
                    continue

                # 제외 키워드 필터
                if any(ex in clean_name for ex in EXCLUDE_KEYWORDS):
                    continue

                # 중복 제거
                if clean_name in saved_cafes:
                    continue

                saved_cafes.add(clean_name)

                # 네이버 지도 URL 생성
                naver_map_link = generate_naver_map_link(clean_name)

                try:
                    # DB 저장
                    cafe_id = insert_cafe(
                        clean_name, road_address, naver_map_link
                    )

                    if cafe_id:
                        total_saved += 1
                        print(f"저장 완료 [{cafe_id}] {clean_name}")

                except Exception as e:
                    print(f"DB 저장 실패 {clean_name}")
                    print(e)

    print("\n====================")
    print(f"총 {total_saved}개 저장 완료")