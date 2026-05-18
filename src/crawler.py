import os
import requests
import urllib.request
from dotenv import load_dotenv
import json
from utils import CLIPEmbedding  # 개발자님이 만드신 임베딩 파일 (CLIP 로드용 예시)
from search_engine import SimpleFaissDB
from PIL import Image

# .env 파일 로드
load_dotenv()

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
}
# 신분증 묶음(HEADERS)을 딕셔너리로 깔끔하게 정의해 두고, requests.get() 명령어 한 줄로 주소와 신분증을 동시에 실어서 네이버 서버에 던집니다. 가독성이 훨씬 좋습니다.

def search_cafe_places(query, display=5):
    """네이버 지역 검색 API로 카페 정보 가져오기"""
    url = f"https://openapi.naver.com/v1/search/local.json?query={query}&display={display}&sort=comment"
    # local.json은 지역 검색, 검색 개수와 리뷰 많은 순으로 정렬
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return response.json().get('items', [])
    else:
        print(f"지역 검색 실패: {response.status_code}")
        return []

def search_cafe_images(cafe_name, display=3):
    """네이버 이미지 검색 API로 카페 사진 URL 가져오기"""
    url = f"https://openapi.naver.com/v1/search/image?query={cafe_name}&display={display}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        return [item['link'] for item in response.json().get('items', [])]
    else:
        print(f"이미지 검색 실패: {response.status_code}")
        return []

def download_image(url, save_path):
    """이미지 URL에서 사진 다운로드하여 로컬에 저장"""
    try:
        # 네이버 이미지 검색 결과 중 간혹 다운로드가 막힌 주소가 있을 수 있어 예외처리 수행
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            with open(save_path, 'wb') as out_file:
                out_file.write(response.read())
        return True
    except Exception as e:
        print(f"이미지 다운로드 실패 ({url}): {e}")
        return False

if __name__ == "__main__":
    print("성수 & 서울숲 지역 집중 크롤링 시작...")
    
    # 1. 🎯 타게팅할 구체적인 키워드 리스트 정의
    keywords = [
        "성수동 분위기 카페",
        "성수역 감성 카페",
        "성수 카페",
        "성수 느좋 카페"
        "성수 카페 추천",
        "서울숲 카페",
        "서울숲 느좋 카페",
        "서울숲 카페 추천",
        "뚝섬역 카페 추천",
        "뚝섬역 느좋 카페"
    ]
    
    db_metadata = {}
    os.makedirs("data/raw", exist_ok=True)
    
    # 중복 저장 방지용 세트 (이미 뽑은 카페는 또 안 뽑게 차단)
    saved_cafe_names = set()
    global_idx = 0 # 이미지 파일명을 고유하게 만들기 위한 일련번호
    
    # 2. 키워드를 하나씩 순회하며 검색 수행
    for keyword in keywords:
        print(f"\n[{keyword}] 키워드로 검색 중...")
        
        # 각 키워드당 상위 5개씩 가져오기 (원하시면 display 숫자를 늘리셔도 됩니다)
        cafes = search_cafe_places(keyword, display=5)
        
        for cafe in cafes:
            clean_name = cafe['title'].replace("<b>", "").replace("</b>", "")
            road_address = cafe['roadAddress']
            naver_map_link = cafe.get('link', '')
            
            # 주소 검증 및 중복 체크 
            # (네이버가 간혹 다른 지역을 끼워 넣을 수 있으므로 주소에 '성동구'가 포함되어 있는지 확인)
            if "성동구" not in road_address:
                continue
                
            if clean_name in saved_cafe_names:
                print(f"중복 패스: {clean_name}")
                continue
                
            print(f"수집 중: {clean_name}")
            saved_cafe_names.add(clean_name)
            
            # 이미지 검색 및 다운로드
            img_urls = search_cafe_images(clean_name, display=1)
            if img_urls:
                file_name = f"cafe_{global_idx}.jpg"
                success = download_image(img_urls[0], f"data/raw/{file_name}")
                
                if success:
                    print(f"이미지 저장 완료: data/raw/{file_name}")
                    
                # JSON 데이터 적재
                db_metadata[str(global_idx)] = {
                    "image_path": file_name,
                    "tags": [],
                    "location": road_address,
                    "map_url": naver_map_link
                }
                global_idx += 1
                
    # 3. 최종 하나의 파일로 통합 저장
    with open("data/metadata.json", "w", encoding="utf-8") as f:
        json.dump(db_metadata, f, ensure_ascii=False, indent=2)
        
    print(f"\n성수/서울숲 특화 테스트 완료! 총 {len(saved_cafe_names)}개의 카페를 찾았습니다.")
    
    