import os
from pathlib import Path

import psycopg2

from database.postgres_new import DATABASE_URL


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS = ["train", "valid_seen", "valid_unseen"]
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".webp"]
TABLE_NAME = "cafe_final_images"


conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()

cur.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN IF NOT EXISTS split TEXT;")

updated = 0
not_found = 0

for split in SPLITS:
    split_dir = BASE_DIR / split

    if not split_dir.exists():
        print(f"Folder not found: {split_dir}")
        continue

    for img_path in split_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        file_name = img_path.name
        db_path = os.path.join("data", "processed", split, img_path.relative_to(split_dir))
        db_path = db_path.replace("\\", "/")

        cur.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET split = %s
            WHERE image_path = %s
               OR image_path = %s
               OR image_path LIKE %s
            """,
            (split, db_path, file_name, f"%/{file_name}"),
        )

        if cur.rowcount == 0:
            print(f"Not found in DB: {file_name}")
            not_found += 1
        else:
            updated += cur.rowcount

conn.commit()
cur.close()
conn.close()

print("\n===== Result =====")
print(f"Updated: {updated}")
print(f"Not found: {not_found}")
