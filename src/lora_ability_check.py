import torch
import numpy as np
import faiss
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from model_utils import get_lora_clip_model

# ========================
# 설정
# ========================
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "src" /"faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"
LORA_PATH = BASE_DIR / "src" / "models" / "lora_weights5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================
# 모델 로드
# ========================
def load_model():
    model, processor = get_lora_clip_model()

    if LORA_PATH.exists():
        model.load_adapter(str(LORA_PATH), adapter_name="lora")
        model.set_adapter("lora")

    model.to(DEVICE)
    model.eval()
    return model, processor


# ========================
# 데이터 로드
# ========================
def load_image_paths():
    return [str(p) for p in np.load(PATHS_PATH, allow_pickle=True)]


def extract_labels(image_paths):
    labels = []
    class_names = []

    for path in image_paths:
        label = Path(path).parent.name  # 폴더 이름

        labels.append(label)
        if label not in class_names:
            class_names.append(label)

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    labels_idx = [label_to_idx[l] for l in labels]

    return labels_idx, class_names


def load_images(image_paths, max_samples=300):
    images = []
    valid_labels = []

    for path, label in zip(image_paths, labels):
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
            valid_labels.append(label)
        except:
            continue

        if len(images) >= max_samples:
            break

    return images, valid_labels


# ========================
# Zero-shot Accuracy
# ========================
def zero_shot_accuracy(model, processor, images, labels, class_names):
    correct = 0

    text_inputs = processor(text=class_names, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_outputs = model.get_text_features(**text_inputs)

        # 🔥 여기 추가
        if hasattr(text_outputs, "text_embeds"):
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    for img, label in zip(images, labels):
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            image_outputs = model.get_image_features(**inputs)

            # 🔥 여기 추가
            if hasattr(image_outputs, "image_embeds"):
                image_features = image_outputs.image_embeds
            elif hasattr(image_outputs, "pooler_output"):
                image_features = image_outputs.pooler_output
            else:
                image_features = image_outputs

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity = image_features @ text_features.T
        pred = similarity.argmax().item()

        if pred == label:
            correct += 1

    return correct / len(images)

# ========================
# Retrieval Recall@K
# ========================
def compute_recall_at_k(index, labels, k=1):
    correct = 0
    n = len(labels)

    for i in tqdm(range(n)):
        query_vec = index.reconstruct(i)

        D, I = index.search(np.array([query_vec]), k+1)
        neighbors = I[0][1:]  # 자기 자신 제외

        if any(labels[i] == labels[j] for j in neighbors):
            correct += 1

    return correct / n


# ========================
# 실행
# ========================
if __name__ == "__main__":
    print("모델 로딩 중...")
    model, processor = load_model()

    print("데이터 로딩 중...")
    image_paths = load_image_paths()
    labels, class_names = extract_labels(image_paths)

    print("이미지 로딩 중...")
    images, valid_labels = load_images(image_paths)

    print("Zero-shot Accuracy 계산 중...")
    acc = zero_shot_accuracy(model, processor, images, valid_labels, class_names)

    print("FAISS 인덱스 로딩 중...")
    index = faiss.read_index(str(INDEX_PATH))

    print("Recall@1 계산 중...")
    r1 = compute_recall_at_k(index, labels, k=1)

    print("Recall@5 계산 중...")
    r5 = compute_recall_at_k(index, labels, k=5)

    print("\n===== 결과 =====")
    print(f"Zero-shot Accuracy: {acc:.4f}")
    print(f"Recall@1: {r1:.4f}")
    print(f"Recall@5: {r5:.4f}")