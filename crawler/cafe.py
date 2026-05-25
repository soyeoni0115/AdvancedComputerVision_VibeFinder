import os
import requests
import urllib.request
from dotenv import load_dotenv
import json
from urllib.parse import quote

from database.postgres import insert_cafe

# =========================
# .env 로드
# =========================
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
}

# =========================
# 네이버 지역 검색 API
# =========================
def search_cafe_places(query, display=5):

    url = (
        f"https://openapi.naver.com/v1/search/local.json"
        f"?query={quote(query)}&display={display}&sort=comment"
    )

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:

        return response.json().get("items", [])

    else:

        print(f"지역 검색 실패: {response.status_code}")

        return []

# =========================
# 네이버 이미지 검색 API
# =========================
def search_cafe_images(query, display=3):

    url = (
        f"https://openapi.naver.com/v1/search/image.json"
        f"?query={quote(query)}&display={display}"
    )

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:

        items = response.json().get("items", [])

        print(f"이미지 검색 결과 수: {len(items)}")

        return [
            item["link"]
            for item in items
        ]

    else:

        print(f"이미지 검색 실패: {response.status_code}")

        return []

# =========================
# 이미지 다운로드
# =========================
def download_image(url, save_path):

    try:

        headers = {

            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )

        }

        req = urllib.request.Request(
            url,
            headers=headers
        )

        with urllib.request.urlopen(
            req,
            timeout=10
        ) as response:

            with open(save_path, "wb") as out_file:

                out_file.write(response.read())

        return True

    except Exception as e:

        print(f"이미지 다운로드 실패: {e}")

        return False

# =========================
# 네이버 지도 링크 생성
# =========================
def generate_naver_map_link(cafe_name):

    encoded_name = quote(cafe_name)

    return f"https://map.naver.com/p/search/{encoded_name}"

# =========================
# 메인 실행
# =========================
if __name__ == "__main__":

    print("성수 / 서울숲 / 뚝섬 카페 크롤링 시작")
    # 키워드 그때마다 바꿔가면서 해서 안 올림
    keywords= [

                
                ]

    # metadata 저장용
    db_metadata = {}

    # 이미지 저장 폴더 생성
    os.makedirs("data/raw", exist_ok=True)

    # 중복 방지 및 ID 추적용 딕셔너리 (이름 : cafe_id)
    # 이제 set 대신 dict를 사용해 중복 시 기존 등록된 ID를 찾기
    saved_cafes_registry = {}

    # =========================
    # 키워드 순회
    # =========================
    for keyword in keywords:

        print(f"\n[{keyword}] 검색 중...")

        cafes = search_cafe_places(
            keyword,
            display=5
        )

        for cafe in cafes:

            clean_name = (
                cafe["title"]
                .replace("<b>", "")
                .replace("</b>", "")
            )

            category = cafe.get("category", "")
            road_address = cafe["roadAddress"]

            # =========================
            # 제외 키워드 필터
            # =========================
            exclude_keywords = [
                "스터디", "프린트", "만화", "보드게임", "룸카페", "멀티방", "PC", "피시",
                "애견", "키즈", "메가커피", "컴포즈", "빽다방", "스타벅스", "집무실",
                "스파크플러스", "라운지", "공유오피스", "오피스", "렌탈", "스튜디오", "파티룸"
            ]

            if any(k in clean_name for k in exclude_keywords):
                print(f"제외된 카페: {clean_name}")
                continue

            # =========================
            # 성동구 필터
            # =========================
            if "성동구" not in road_address:
                continue

            # =========================
            # 중복 발생 처리
            # =========================
            if clean_name in saved_cafes_registry:
                existing_id = saved_cafes_registry[clean_name]
                cafe_key = f"cafe_{existing_id}"
                
                # [핵심 추가] 이미 등록된 카페라면, 새로운 검색 키워드만 리스트에 추가하고 패스
                if keyword not in db_metadata[cafe_key]["search_keywords"]:
                    db_metadata[cafe_key]["search_keywords"].append(keyword)
                    
                print(f"⚠️ 중복 패스: {clean_name} -> 키워드 '{keyword}' 추가 완료")
                continue

            # =========================
            # 신규 수집 처리
            # =========================
            print(f"수집 중: {clean_name}")
            naver_map_link = generate_naver_map_link(clean_name)

            # DB 저장 (수정했던 RETURNING id 함수)
            cafe_id = insert_cafe(clean_name, road_address, naver_map_link)

            if cafe_id is None:
                print(f" DB에서 cafe_id를 받지 못했습니다. 넘어갑니다.")
                continue

            # 레지스트리에 기록 (추후 중복 체크용)
            saved_cafes_registry[clean_name] = cafe_id

            # ==========================================
            # [변경] metadata 구조 설정 (search_keywords 추가)
            # ==========================================
            cafe_key = f"cafe_{cafe_id}"
            db_metadata[cafe_key] = {
                "cafe_id": cafe_id,
                "cafe_name": clean_name,
                "location": road_address,
                "map_url": naver_map_link,
                "search_keywords": [keyword],  # 첫 발견된 검색어 삽입
                "reviews": [],                  # 다음 단계에서 크롤링해 채울 빈 리스트
                "images": [],
                "mood_tags": []                 # LLM이 최종적으로 채워넣을 공간
            }

            # =========================
            # 이미지 검색 및 다운로드
            # =========================
            image_query = f"{clean_name} 성수 카페"
            print(f"이미지 검색어: {image_query}")
            img_urls = search_cafe_images(image_query, display=5)

            for img_idx, img_url in enumerate(img_urls):
                file_name = f"{cafe_id}_{img_idx}.jpg"
                save_path = f"data/raw/{file_name}"

                success = download_image(img_url, save_path)
                if success:
                    print(f"이미지 저장 완료: {save_path}")
                    db_metadata[cafe_key]["images"].append(file_name)

    # =========================
    # metadata.json 저장
    # =========================
    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(db_metadata, f, ensure_ascii=False, indent=2)

    print(f"\n크롤링 완료! 총 {len(saved_cafes_registry)}개의 카페 수집 및 metadata.json 저장 완료")