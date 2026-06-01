import os
import numpy as np
import faiss
import psycopg2
import torch
from pathlib import Path
from model_utils import get_lora_clip_model
from dotenv import load_dotenv

load_dotenv()

# 1. 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = BASE_DIR / "faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"
LORA_PATH = BASE_DIR / "models" / "lora_weights"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 이미지 탐색 우선순위 폴더들
IMAGE_DIRS = [
    DATA_DIR / "train_aug",
    DATA_DIR / "train_aug_2",
    DATA_DIR / "raw"
]

print("파인튜닝된 CLIP 모델 및 LoRA 가중치 로드 중...")
model, processor = get_lora_clip_model()
if LORA_PATH.exists():
    model.load_adapter(str(LORA_PATH), adapter_name="default")
    model.set_adapter("default")
model.to(DEVICE)
model.eval()

# Neon DB 접속 정보를 그대로 활용
print("Neon 데이터베이스에 연결하여 이미지 매핑용 ID 리스트를 가져오는 중...")
try:
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    with conn.cursor() as cur:
        # DB에 등록된 이미지 데이터의 ID와 실제 경로/파일명 정보를 가져옵니다.
        cur.execute("SELECT id, image_path FROM cafe_images ORDER BY id ASC;")
        db_rows = cur.fetchall()
    conn.close()
except Exception as e:
    print(f"DB 연결 실패! 확인이 필요합니다: {e}")
    exit()

print(f" DB에서 총 {len(db_rows)}개의 이미지 정보를 성공적으로 조회했습니다.")

# 로컬 컴퓨터에 존재하는 진짜 이미지 파일 매핑 함수
def find_local_file(db_img_path):
    filename = os.path.basename(db_img_path) # 예: 2_0.jpg 또는 2_0aug0.jpg
    cafe_id_dir = filename.split('_')[0] if '_' in filename else ""

    for target_dir in IMAGE_DIRS:
        if not target_dir.exists():
            continue
        
        # 후보 1: data/.../2/2_0.jpg (하위 ID 폴더가 있는 경우)
        if cafe_id_dir:
            prob1 = target_dir / cafe_id_dir / filename
            if prob1.exists():
                return prob1
        
        # 후보 2: data/.../2_0.jpg (폴더 구분 없이 바로 들어있는 경우)
        prob2 = target_dir / filename
        if prob2.exists():
            return prob2
            
    return None

ordered_paths = []
embeddings = []

print("📸 DB ID 순서에 맞춰 로컬 이미지 CLIP 임베딩 추출 중...")
# ⚠️ 아주 중요: DB의 id 순서(0, 1, 2...)와 FAISS 인덱스 순서가 완벽하게 일치해야 합니다!
for vector_id, db_img_path in db_rows:
    local_file_path = find_local_file(db_img_path)
    
    if local_file_path and local_file_path.exists():
        try:
            # 팀원의 모델 포맷에 맞춰 전처리 및 임베딩 생성
            from PIL import Image
            image = Image.open(local_file_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                img_features = model.get_image_features(**inputs)
            
            if hasattr(img_features, "image_embeds") and img_features.image_embeds is not None:
                img_features = img_features.image_embeds
            elif hasattr(img_features, "pooler_output") and img_features.pooler_output is not None:
                img_features = img_features.pooler_output
                
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            emb = img_features.cpu().numpy().flatten().astype("float32")
            
            embeddings.append(emb)
            ordered_paths.append(str(local_file_path))
        except Exception as e:
            print(f"❌ 임베딩 추출 실패 (ID {vector_id}, 파일 {local_file_path}): {e}")
            # 순서가 뒤틀리는 것을 막기 위해 에러 발생 시 더미(0 벡터)를 채웁니다.
            embeddings.append(np.zeros(512, dtype="float32"))
            ordered_paths.append("")
    else:
        print(f"⚠️ 로컬에서 파일을 찾지 못함 (DB 경로: {db_img_path}) -> 공백 처리")
        embeddings.append(np.zeros(512, dtype="float32"))
        ordered_paths.append("")

# 2. FAISS 인덱스 생성 및 저장
print("FAISS Index FlatIP 빌드 중...")
embeddings_array = np.array(embeddings).astype('float32')
dimension = embeddings_array.shape[1]

index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings_array)
index.add(embeddings_array)

# 기존 파일 지우고 새로 덮어쓰기
if INDEX_PATH.exists(): os.remove(INDEX_PATH)
if PATHS_PATH.exists(): os.remove(PATHS_PATH)

faiss.write_index(index, str(INDEX_PATH))
np.save(str(PATHS_PATH), np.array(ordered_paths, dtype=object))

print(f"완벽하게 동기화 완료! 총 {len(ordered_paths)}개의 이미지가 인덱스에 매핑되었습니다.")