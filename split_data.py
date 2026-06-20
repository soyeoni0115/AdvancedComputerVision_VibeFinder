# 데이터를 train/valid로 분할했을 때 사용한 파일
import os
import shutil
import random

# ==========================================
# 경로
# ==========================================

source_dir = "data/all_raw"

output_train_dir = "data/train"
output_valid_dir = "data/valid"

os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_valid_dir, exist_ok=True)

# ==========================================
# all_raw 확인
# ==========================================

if not os.path.exists(source_dir):

    print(f"❌ {source_dir} 없음")
    exit()

all_files = [
    f
    for f in os.listdir(source_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"📷 전체 이미지 수: {len(all_files)}")

# ==========================================
# 카페 ID 추출
# ==========================================

cafe_ids = sorted(
    list(
        set(
            int(f.split("_")[0])
            for f in all_files
        )
    )
)

print(f"☕ 전체 카페 수: {len(cafe_ids)}")

# ==========================================
# 카페 단위 랜덤 분할
# ==========================================

random.seed(42)

random.shuffle(cafe_ids)

split_idx = int(len(cafe_ids) * 0.8)

train_ids = set(cafe_ids[:split_idx])
valid_ids = set(cafe_ids[split_idx:])

print()
print("Train 카페 수:", len(train_ids))
print("Valid 카페 수:", len(valid_ids))

# ==========================================
# 이미지 복사
# ==========================================

train_count = 0
valid_count = 0

for file_name in all_files:

    cafe_id = int(file_name.split("_")[0])

    src_path = os.path.join(
        source_dir,
        file_name
    )

    if cafe_id in train_ids:

        dst_path = os.path.join(
            output_train_dir,
            file_name
        )

        shutil.copy2(
            src_path,
            dst_path
        )

        train_count += 1

    else:

        dst_path = os.path.join(
            output_valid_dir,
            file_name
        )

        shutil.copy2(
            src_path,
            dst_path
        )

        valid_count += 1

# ==========================================
# 결과
# ==========================================

print()
print("=================================")
print("✨ Split 완료")
print("=================================")

print(
    f"📁 Train : {train_count}장 "
    f"({len(train_ids)}개 카페)"
)

print(
    f"📁 Valid : {valid_count}장 "
    f"({len(valid_ids)}개 카페)"
)

print()

print("Train IDs")
print(sorted(train_ids))

print()

print("Valid IDs")
print(sorted(valid_ids))