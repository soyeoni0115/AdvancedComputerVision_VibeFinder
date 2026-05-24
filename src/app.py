import streamlit as st  # Streamlit UI 라이브러리
import numpy as np  # 벡터 연산용
import faiss  # 벡터 검색 라이브러리
import psycopg2  # PostgreSQL 연결 라이브러리
from pathlib import Path  # 파일 경로 처리
from utils import CLIPEmbedding  # CLIP 모델 (직접 만든 클래스)

# -------------------------------
# 1. 기본 설정
# -------------------------------

st.set_page_config(
    page_title="Vibe Finder",  # 페이지 제목
    layout="wide"  # 전체 화면 사용
)

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_DIR = BASE_DIR / "data" / "raw"

# -------------------------------
# 2. CLIP 모델 로드
# -------------------------------

clip_model = CLIPEmbedding()  # 텍스트 → 벡터 변환용

# -------------------------------
# 3. FAISS 인덱스 로드
# -------------------------------

index = faiss.read_index(
    str(BASE_DIR / "faiss_vibe.index")  # 미리 만들어둔 벡터 DB
)

# -------------------------------
# 4. PostgreSQL 연결
# -------------------------------

conn = psycopg2.connect(
    dbname="vibe",      # DB 이름
    user="user",        # 사용자
    password="pass",    # 비밀번호
    host="localhost",   # 로컬 서버
    port="5432"         # 포트
)

cur = conn.cursor()  # 쿼리 실행 객체

# -------------------------------
# 5. UI 구성
# -------------------------------

# 제목
st.title("Vibe Finder")

# 사용자 입력 받기
vibe = st.text_input(
    "원하는 분위기를 입력하세요",
    placeholder="예: 따뜻한 감성 카페, 공부하기 좋은 곳"
)

# -------------------------------
# 6. 검색 버튼 클릭 시 실행
# -------------------------------

if st.button("검색 시작"):

    # 입력이 비어있지 않을 경우
    if vibe.strip():

        # -------------------------------
        # (1) 텍스트 → 벡터 변환
        # -------------------------------

        query_embedding = clip_model.get_text_embedding(vibe)

        # FAISS 입력 형식 맞추기 (2차원 배열)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        # -------------------------------
        # (2) FAISS에서 유사 이미지 검색
        # -------------------------------

        D, I = index.search(query_embedding, k=5)  # Top-5 검색

        results = []  # 결과 저장 리스트

        # -------------------------------
        # (3) PostgreSQL에서 정보 가져오기
        # -------------------------------

        for idx in I[0]:  # 검색된 이미지 id 순회

            # DB에서 해당 id 데이터 조회
            cur.execute(
                """
                SELECT cafe_name, address, map_url, image_path
                FROM cafes
                WHERE id = %s
                """,
                (int(idx),)
            )

            row = cur.fetchone()  # 한 줄 가져오기

            # 결과가 존재하면 리스트에 추가
            if row:
                results.append({
                    "cafe_name": row[0],   # 카페 이름
                    "address": row[1],     # 주소
                    "map_url": row[2],     # 지도 링크
                    "image_path": row[3]   # 이미지 파일 경로
                })

        # -------------------------------
        # (4) 결과 출력
        # -------------------------------

        if results:
            st.write("### 추천 카페")

            for cafe in results:

                # 이미지 파일 경로
                image_file = RAW_IMAGE_DIR / cafe["image_path"]

                # 좌우 레이아웃
                col1, col2 = st.columns([1, 2])

                with col1:
                    if image_file.exists():
                        st.image(str(image_file), use_container_width=True)
                    else:
                        st.warning("이미지를 찾을 수 없음")

                with col2:
                    st.subheader(cafe["cafe_name"])
                    st.write(f"주소: {cafe['address']}")
                    st.write(f"[지도 보기]({cafe['map_url']})")

        else:
            st.warning("검색 결과가 없습니다.")

    else:
        st.warning("검색어를 입력하세요.")