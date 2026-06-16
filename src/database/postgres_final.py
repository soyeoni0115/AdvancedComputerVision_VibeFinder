import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text


# ==========================================
# 설정
# ==========================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

IMAGE_DIR = Path("data/all_raw")


# ==========================================
# 네온에 업로드
# ==========================================

def upload_images():

    files = sorted(
        [
            f
            for f in IMAGE_DIR.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    print(f"총 이미지 수: {len(files)}")

    with engine.begin() as conn:

        # 혹시 이전 데이터 있으면 삭제
        conn.execute(
            text("""
                DELETE FROM cafe_final_images;
            """)
        )

        inserted = 0

        for file in files:

            file_name = file.name

            cafe_id = int(
                file.stem.split("_")[0]
            )

            conn.execute(
                text("""
                    INSERT INTO cafe_final_images (

                        cafe_id,
                        image_path,
                        caption

                    )

                    VALUES (

                        :cafe_id,
                        :image_path,
                        NULL

                    );
                """),
                {
                    "cafe_id": cafe_id,
                    "image_path": file_name
                }
            )

            inserted += 1

            if inserted % 100 == 0:

                print(
                    f"{inserted}장 업로드 완료"
                )

    print()
    print("====================")
    print(f"업로드 완료: {inserted}장")
    print("====================")


# ==========================================
# 확인
# ==========================================

def check_count():

    with engine.connect() as conn:

        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM cafe_final_images;
            """)
        )

        count = result.scalar()

        print(
            f"DB 이미지 수: {count}"
        )


# ==========================================
# 실행
# ==========================================

if __name__ == "__main__":

    upload_images()

    check_count()