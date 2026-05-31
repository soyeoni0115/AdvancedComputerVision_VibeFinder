import os
from pathlib import Path

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

                    image_path TEXT,
                    
                    caption TEXT

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


# ==================================================
# 🔥 [추가] 증강 이미지 자동 스캔 및 INSERT 함수
# ==================================================
def insert_augmented_images():
    """
    로컬의 train_aug, train_aug_2 폴더를 뒤져서 
    DB에 등록되지 않은 증강 이미지만 골라 cafe_images 테이블에 추가합니다.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # 1. 탐색할 증강 폴더 지정
    AUG_DIRS = [
        DATA_DIR / "train_aug",
        DATA_DIR / "train_aug_2"
    ]
    
    # 2. 현재 DB에 이미 등록되어 있는 모든 image_path를 싹 긁어와서 중복 방지 셋 만들기
    existing_paths = set()
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT image_path FROM cafe_images;"))
            for row in result.fetchall():
                existing_paths.add(row[0])
            print(f"📊 현재 DB에 등록된 이미지 경로 수: {len(existing_paths)}개")
    except Exception as e:
        print(f"🚨 기존 이미지 경로 조회 실패: {e}")
        return

    # 3. 로컬 폴더 스캔하여 인서트 데이터 빌드
    images_to_insert = []
    
    for aug_dir in AUG_DIRS:
        if not aug_dir.exists():
            print(f"⚠️ 폴더가 존재하지 않아 건너뜁니다: {aug_dir}")
            continue
            
        # 하위 폴더(예: 2, 3, 10...) 및 파일들 싹 다 뒤지기
        for file_path in aug_dir.rglob("*"):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # 팀원의 DB 포맷 형식인 '../data/train_aug/2/2_0aug0.jpg' 형태로 경로 가공
                # 만약 팀원의 프로젝트 실행 환경 주소가 다르면 수정을 위해 상대경로 구조화
                relative_path = f"../data/{file_path.relative_to(DATA_DIR)}".replace("\\", "/")
                
                # 중복 검사
                if relative_path in existing_paths:
                    continue
                    
                # 파일명(예: 11_0aug0.jpg)에서 언더바(_) 앞자리를 떼어내어 cafe_id 파싱
                filename = file_path.name
                if '_' in filename:
                    try:
                        cafe_id = int(filename.split('_')[0])
                        
                        # 인서트 대상 리스트에 추가 (caption은 우선 빈 값 처리, 필요시 수정 가능)
                        images_to_insert.append({
                            "cafe_id": cafe_id,
                            "image_path": relative_path,
                            "caption": "augmented image data" 
                        })
                    except ValueError:
                        print(f"❌ 파일명에서 cafe_id를 추출할 수 없습니다 (패스): {filename}")

    # 4. DB에 대량(Bulk) INSERT 실행
    if not images_to_insert:
        print("✅ DB에 새로 추가할 증강 이미지가 없습니다. 이미 최신 상태입니다.")
        return

    print(f"🚀 총 {len(images_to_insert)}개의 새로운 증강 이미지를 DB에 추가하는 중...")
    
    try:
        with engine.connect() as conn:
            # 팀원의 최신 테이블 스펙(caption 컬럼 포함)에 맞추어 실행
            conn.execute(
                text("""
                    INSERT INTO cafe_images (cafe_id, image_path, caption)
                    VALUES (:cafe_id, :image_path, :caption);
                """),
                images_to_insert
            )
            conn.commit()
            print(f"🎉 성공적으로 {len(images_to_insert)}개의 행이 cafe_images에 추가되었습니다!")
    except Exception as e:
        print(f"🚨 DB 대량 인서트 중 오류 발생: {e}")


# =========================
# 실행 영역
# =========================
if __name__ == "__main__":
    # 1. DB 연결 테스트
    test_connection()

    # 2. 테이블이 없다면 자동 생성
    create_tables()
    
    # 3. 🔥 증강 이미지 자동 스캔 및 DB 밀어넣기 실행
    insert_augmented_images()