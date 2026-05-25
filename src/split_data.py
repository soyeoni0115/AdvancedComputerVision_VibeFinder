import os
import random
import shutil
from collections import defaultdict

RAW_DIR = "../data/raw"
BASE_DIR = "../data/processed"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_SEEN_DIR = os.path.join(BASE_DIR, "valid_seen")
VALID_UNSEEN_DIR = os.path.join(BASE_DIR, "valid_unseen")

UNSEEN_RATIO = 0.2
SEEN_SPLIT = 0.8

random.seed(42)

# 폴더 생성
for d in [TRAIN_DIR, VALID_SEEN_DIR, VALID_UNSEEN_DIR]:
    os.makedirs(d, exist_ok=True)

# ===== 카페별 그룹핑 =====
cafe_dict = defaultdict(list)

for img in os.listdir(RAW_DIR):
    if not img.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    
    cafe_id = img.split("_")[0]
    cafe_dict[cafe_id].append(img)

cafes = list(cafe_dict.keys())
random.shuffle(cafes)

# ===== unseen 분리 =====
num_unseen = int(len(cafes) * UNSEEN_RATIO)
unseen_cafes = cafes[:num_unseen]
seen_cafes = cafes[num_unseen:]

# ===== unseen → valid_unseen =====
for cafe in unseen_cafes:
    os.makedirs(os.path.join(VALID_UNSEEN_DIR, cafe), exist_ok=True)
    
    for img in cafe_dict[cafe]:
        shutil.copy(
            os.path.join(RAW_DIR, img),
            os.path.join(VALID_UNSEEN_DIR, cafe, img)
        )

# ===== seen → train / valid_seen =====
for cafe in seen_cafes:
    imgs = cafe_dict[cafe]
    random.shuffle(imgs)

    split_idx = int(len(imgs) * SEEN_SPLIT)

    train_imgs = imgs[:split_idx]
    valid_imgs = imgs[split_idx:]

    os.makedirs(os.path.join(TRAIN_DIR, cafe), exist_ok=True)
    os.makedirs(os.path.join(VALID_SEEN_DIR, cafe), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(RAW_DIR, img),
            os.path.join(TRAIN_DIR, cafe, img)
        )

    for img in valid_imgs:
        shutil.copy(
            os.path.join(RAW_DIR, img),
            os.path.join(VALID_SEEN_DIR, cafe, img)
        )

print("✅ split 완료")