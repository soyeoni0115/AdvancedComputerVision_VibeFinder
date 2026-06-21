# SQLAlchemy로 Neon PostgreSQL에 테이블을 생성하고 카페 데이터를 저장/조회하는 DB 초기 설정 코드
import os
from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy import text

# .env 로드
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# DB 엔진 생성
engine = create_engine(DATABASE_URL)


# DB 연결 테스트
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


# 테이블 생성
def create_tables():

    try:

        with engine.connect() as conn:

            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS cafes (

                    id SERIAL PRIMARY KEY,

                    cafe_name TEXT UNIQUE,

                    location TEXT,

                    map_url TEXT,

                    -- 추가: 사진 수집 완료 여부
                    photo_crawled BOOLEAN DEFAULT FALSE

                );

            """))

            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS cafe_images (

                    id SERIAL PRIMARY KEY,

                    cafe_id INTEGER REFERENCES cafes(id),

                    image_path TEXT,

                    caption TEXT

                );

            """))

            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS reviews (

                    id SERIAL PRIMARY KEY,

                    cafe_id INTEGER REFERENCES cafes(id),

                    review_text TEXT

                );

            """))

            # 변경: 기존 tags 대신 CLIP 프롬프트 저장용
            conn.execute(text("""

                CREATE TABLE IF NOT EXISTS vibe_presets (

                    id SERIAL PRIMARY KEY,

                    tag_name TEXT UNIQUE,

                    clip_prompt TEXT,

                    is_active BOOLEAN DEFAULT TRUE

                );

            """))

            conn.commit()

            print("테이블 생성 완료!")

    except Exception as e:

        print("테이블 생성 실패")
        print(e)


# 카페 저장
def insert_cafe(cafe_name, location, map_url):

    try:

        with engine.connect() as conn:

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

                    DO UPDATE SET

                        location = EXCLUDED.location,
                        map_url = EXCLUDED.map_url

                    RETURNING id;

                """),
                {
                    "cafe_name": cafe_name,
                    "location": location,
                    "map_url": map_url
                }
            )

            row = result.fetchone()

            conn.commit()

            if row:

                cafe_id = row[0]

                print(
                    f"저장 완료: "
                    f"{cafe_name} "
                    f"(ID: {cafe_id})"
                )

                return cafe_id

            return None

    except Exception as e:

        print(f"저장 실패: {e}")

        return None


# 전체 카페 조회
def get_all_cafes():

    try:

        with engine.connect() as conn:

            result = conn.execute(
                text("SELECT * FROM cafes")
            )

            return result.fetchall()

    except Exception as e:

        print(f"조회 실패: {e}")

        return []

# 여기서부터 진짜 실행부인건가
if __name__ == "__main__":

    test_connection()

    create_tables()