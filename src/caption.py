# 캡션 생성 코드
import os

from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from PIL import Image
from tqdm import tqdm

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

# =====================================
# 환경변수
# =====================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# =====================================
# 경로
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

IMAGE_DIR = BASE_DIR / "data" / "raw"

# =====================================
# DB 연결
# =====================================

conn = psycopg2.connect(
    DATABASE_URL
)

cur = conn.cursor()

# =====================================
# BLIP 모델
# =====================================

print("BLIP 로딩 중...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("BLIP 로딩 완료")

# =====================================
# 캡션 보강
# =====================================

def smart_caption(blip_caption):

    caption = blip_caption.lower()

    tags = []

    tags.extend([
        "cafe interior",
        "coffee shop",
        "seongsu cafe"
    ])

    if any(
        x in caption
        for x in [
            "window",
            "sunlight",
            "light"
        ]
    ):
        tags.extend([
            "bright atmosphere",
            "natural lighting",
            "airy space",
            "cozy cafe"
        ])

    if any(
        x in caption
        for x in [
            "wood",
            "wooden"
        ]
    ):
        tags.extend([
            "warm interior",
            "wooden furniture",
            "cozy atmosphere"
        ])

    if any(
        x in caption
        for x in [
            "plant",
            "green"
        ]
    ):
        tags.extend([
            "nature inspired",
            "green interior",
            "relaxing atmosphere"
        ])

    if any(
        x in caption
        for x in [
            "table",
            "desk"
        ]
    ):
        tags.extend([
            "good for studying",
            "work friendly",
            "quiet atmosphere"
        ])

    if any(
        x in caption
        for x in [
            "sofa",
            "couch"
        ]
    ):
        tags.extend([
            "comfortable seating",
            "relaxing place"
        ])

    if "white" in caption:

        tags.extend([
            "minimalist design",
            "clean aesthetic"
        ])

    if "brick" in caption:

        tags.extend([
            "industrial style",
            "urban atmosphere"
        ])

    if any(
        x in caption
        for x in [
            "lamp",
            "lighting"
        ]
    ):
        tags.extend([
            "warm lighting",
            "moody atmosphere"
        ])

    if any(
        x in caption
        for x in [
            "cake",
            "dessert",
            "pastry"
        ]
    ):
        tags.extend([
            "dessert cafe",
            "sweet atmosphere"
        ])

    tags = list(dict.fromkeys(tags))

    return (
        blip_caption
        + ", "
        + ", ".join(tags)
    )

# =====================================
# 아직 안된 것만 조회
# =====================================

cur.execute("""
    SELECT
        id,
        image_path
    FROM cafe_final_images
    WHERE caption IS NULL
    ORDER BY id
""")

rows = cur.fetchall()

print()
print(f"남은 이미지 수: {len(rows)}")

# =====================================
# 캡션 생성
# =====================================

for image_id, image_path in tqdm(rows):

    try:

        image_file = (
            IMAGE_DIR /
            image_path
        )

        if not image_file.exists():

            print(
                f"파일 없음: {image_path}"
            )

            continue

        image = Image.open(
            image_file
        ).convert("RGB")

        inputs = processor(
            image,
            return_tensors="pt"
        )

        output = model.generate(
            **inputs,
            max_new_tokens=40
        )

        blip_caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        final_caption = smart_caption(
            blip_caption
        )

        cur.execute(
            """
            UPDATE cafe_final_images
            SET caption = %s
            WHERE id = %s
            """,
            (
                final_caption,
                image_id
            )
        )

        conn.commit()

    except Exception as e:

        print()
        print(
            f"실패: {image_path}"
        )

        print(e)

        try:
            conn.rollback()
        except:
            pass

# =====================================
# 종료
# =====================================

cur.close()
conn.close()

print()
print("==========================")
print("캡션 생성 완료")
print("==========================")