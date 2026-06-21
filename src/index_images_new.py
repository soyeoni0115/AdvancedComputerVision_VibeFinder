# 개선된 인덱싱/임베딩 생성하여 FAISS에 저장
import os
import numpy as np
import faiss
import psycopg2
import torch
from pathlib import Path
from PIL import Image
from model_utils import get_lora_clip_model
from dotenv import load_dotenv


# 경로 설정

SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"

INDEX_PATH = BASE_DIR / "faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"

LORA_PATH = BASE_DIR / "models" / "lora_weights2" 

ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIRS = [
    DATA_DIR / "raw"
]

print("=" * 50)
print("BASE_DIR =", BASE_DIR)
print("DATA_DIR =", DATA_DIR)
print("DEVICE =", DEVICE)
print("LoRA Path =", LORA_PATH)
print("LoRA Exists =", LORA_PATH.exists())
print("=" * 50)

# 모델 로드

print("파인튜닝된 CLIP 모델 및 LoRA 가중치 로드 중...")

model, processor = get_lora_clip_model()

if LORA_PATH.exists():
    model.load_adapter(str(LORA_PATH), adapter_name="default")
    model.set_adapter("default")
    print("LoRA 로드 완료")
else:
    print("LoRA 폴더를 찾지 못했습니다.")

model.to(DEVICE)
model.eval()

# DB 연결

print("Neon 데이터베이스 연결 중...")

try:

    db_url = os.getenv("DATABASE_URL")

    if not db_url:
        raise ValueError(
            f"DATABASE_URL이 없습니다. (.env 위치: {ENV_PATH})"
        )

    conn = psycopg2.connect(db_url)

    with conn.cursor() as cur:

        cur.execute("""
            SELECT
                id,
                image_path
            FROM cafe_final_images
            ORDER BY id ASC
        """)

        db_rows = cur.fetchall()

    conn.close()

except Exception as e:

    print(f"DB 연결 실패: {e}")
    exit()

print(f"DB에서 {len(db_rows)}개 이미지 조회 완료")

for row in db_rows[:5]:
    print(row)

# 파일 찾기

def find_local_file(db_img_path):

    filename = os.path.basename(db_img_path)

    for target_dir in IMAGE_DIRS:

        # all_raw/10_1.jpg
        path1 = target_dir / filename

        if path1.exists():
            return path1

        # all_raw/10/10_1.jpg
        cafe_id = filename.split("_")[0]

        path2 = target_dir / cafe_id / filename

        if path2.exists():
            return path2

    return None

# 임베딩 생성

ordered_paths = []
embeddings = []

success_count = 0
missing_count = 0

print("이미지 임베딩 생성 시작...")

for vector_id, db_img_path in db_rows:

    local_file_path = find_local_file(db_img_path)

    if local_file_path is None:

        print(f"파일 없음: {db_img_path}")

        embeddings.append(
            np.zeros(512, dtype=np.float32)
        )

        ordered_paths.append("")
        missing_count += 1

        continue

    try:

        image = Image.open(
            local_file_path
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }

        with torch.no_grad():

            try:

                img_features = model.get_image_features(
                    **inputs
                )

                if not isinstance(
                    img_features,
                    torch.Tensor
                ):

                    if hasattr(
                        img_features,
                        "image_embeds"
                    ):
                        img_features = (
                            img_features.image_embeds
                        )

                    elif hasattr(
                        img_features,
                        "pooler_output"
                    ):
                        img_features = (
                            img_features.pooler_output
                        )

                    else:
                        raise ValueError(
                            f"알 수 없는 출력 타입: {type(img_features)}"
                        )

            except Exception:

                # PEFT 환경 fallback
                vision_outputs = model.vision_model(
                    pixel_values=inputs[
                        "pixel_values"
                    ]
                )

                img_features = (
                    vision_outputs.pooler_output
                )

        img_features = (
            img_features
            / img_features.norm(
                dim=-1,
                keepdim=True
            )
        )

        emb = (
            img_features
            .cpu()
            .numpy()
            .flatten()
            .astype("float32")
        )

        embeddings.append(emb)
        ordered_paths.append(db_img_path)

        success_count += 1

    except Exception as e:

        print(
            f"임베딩 실패 "
            f"(ID={vector_id}, 파일={local_file_path})"
        )

        print(e)

        embeddings.append(
            np.zeros(512, dtype=np.float32)
        )

        ordered_paths.append("")

print("=" * 50)
print(f"성공: {success_count}")
print(f"파일없음: {missing_count}")
print("=" * 50)

# FAISS 생성

print("FAISS IndexFlatIP 생성 중...")

embeddings_array = np.array(
    embeddings,
    dtype=np.float32
)

print("Embedding Shape:", embeddings_array.shape)

dimension = embeddings_array.shape[1]

index = faiss.IndexFlatIP(dimension)

faiss.normalize_L2(
    embeddings_array
)

index.add(
    embeddings_array
)

# 저장

if INDEX_PATH.exists():
    os.remove(INDEX_PATH)

if PATHS_PATH.exists():
    os.remove(PATHS_PATH)

faiss.write_index(
    index,
    str(INDEX_PATH)
)

np.save(
    str(PATHS_PATH),
    np.array(
        ordered_paths,
        dtype=object
    )
)

print("완료")
print(f"총 {len(ordered_paths)}개 저장")
print("FAISS:", INDEX_PATH)
print("PATHS:", PATHS_PATH)