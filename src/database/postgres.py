# SQLAlchemy로 Neon PostgreSQL에 테이블을 생성하고 카페 데이터를 저장/조회하는 DB 가장 초기 설정 코드
import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy import text


# =========================
# .env 파일 로드
# =========================
load_dotenv()


# =========================
# Neon PostgreSQL 연결 문자열
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")


# =========================
# DB 엔진 생성
# =========================
engine = create_engine(DATABASE_URL)


# =========================
# DB 연결 테스트
# =========================
def test_connection():

    try:
        with engine.connect() as conn:

            result = conn.execute(
                text("SELECT version();")
            )

            version = result.fetchone()

            print("PostgreSQL 연결 성공")
            print(version)

    except Exception as e:

        print("연결 실패")
        print(e)


# =========================
# 테이블 생성
# =========================
def create_tables():

    try:
        with engine.connect() as conn:

            # =========================
            # cafes 테이블
            # =========================
            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS cafes (

                    id SERIAL PRIMARY KEY,

                    cafe_name TEXT UNIQUE,

                    location TEXT,

                    map_url TEXT

                );

            """))


            # =========================
            # cafe_images 테이블
            # =========================
            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS cafe_images (

                    id SERIAL PRIMARY KEY,

                    cafe_id INTEGER REFERENCES cafes(id),

                    image_path TEXT

                );

            """))


            # =========================
            # reviews 테이블
            # =========================
            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS reviews (

                    id SERIAL PRIMARY KEY,

                    cafe_id INTEGER REFERENCES cafes(id),

                    review_text TEXT

                );

            """))


            # =========================
            # tags 테이블
            # =========================
            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS tags (

                    id SERIAL PRIMARY KEY,

                    cafe_id INTEGER REFERENCES cafes(id),

                    tag_name TEXT

                );

            """))


            conn.commit()

            print("테이블 생성 완료!")

    except Exception as e:

        print("테이블 생성 실패")
        print(e)


# =========================
# 카페 저장 (수정본)
# =========================
def insert_cafe(cafe_name, location, map_url):
    try:
        with engine.connect() as conn:
            # RETURNING id를 사용해 새로 저장되거나 변경된 행의 id를 가져옴
            # ON CONFLICT가 발생하면 아무것도 안 하는(DO NOTHING) 대신, 
            # 기존 데이터를 유지하며 id만 가져오도록 DO UPDATE를 사용
            result = conn.execute(
                text("""
                INSERT INTO cafes (
                    cafe_name,
                    location,
                    map_url
                )
                VALUES (
                    :cafe_name,
                    :location,
                    :map_url
                )
                ON CONFLICT (cafe_name)
                DO UPDATE SET cafe_name = EXCLUDED.cafe_name
                RETURNING id;
            """),
            {
                "cafe_name": cafe_name,
                "location": location,
                "map_url": map_url
            })
            
            # 반환된 결과에서 id 추출
            row = result.fetchone()
            conn.commit()

            if row:
                cafe_id = row[0]
                print(f"저장 완료: {cafe_name} (ID: {cafe_id})")
                return cafe_id  # ★ 메인 크롤러로 id를 넘겨줌
            else:
                print(f"저장 완료되었으나 id를 가져오지 못했습니다: {cafe_name}")
                return None

    except Exception as e:
        print(f"저장 실패: {e}")
        return None

# =========================
# 전체 카페 조회
# =========================
def get_all_cafes():

    try:
        with engine.connect() as conn:

            result = conn.execute(
                text("SELECT * FROM cafes")
            )

            cafes = result.fetchall()

            return cafes

    except Exception as e:

        print(f"조회 실패: {e}")

        return []


# =========================
# 실행 영역
# =========================
if __name__ == "__main__":

    # DB 연결 테스트
    test_connection()

    # 테이블 생성
    create_tables()