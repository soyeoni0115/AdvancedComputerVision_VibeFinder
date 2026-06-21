import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


BASE_MODEL = "openai/clip-vit-base-patch32"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
LORA_PATH = BASE_DIR / "models" / "lora_weights"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(lora_path: Path = LORA_PATH):
    print(f"Device: {DEVICE}")
    print(f"LoRA path: {lora_path}")

    base_model = CLIPModel.from_pretrained(BASE_MODEL)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)

    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_path}")

    model = PeftModel.from_pretrained(base_model, str(lora_path))
    model.to(DEVICE)
    model.eval()
    return model, processor

def load_base_model():
    print("Loading BASE CLIP (no LoRA)")
    model = CLIPModel.from_pretrained(BASE_MODEL)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)

    model.to(DEVICE)
    model.eval()
    return model, processor

def clip_backbone(model):
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def list_image_paths(split: str):
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    return sorted(
        [
            p
            for p in split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: (int(p.parent.name), p.name),
    )


def labels_from_paths(paths):
    return [p.parent.name for p in paths]


def encode_images(model, processor, image_paths, batch_size=32):
    features = []
    backbone = clip_backbone(model)

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            batch_features = backbone.get_image_features(**inputs)

            # 1. 반환값이 텐서가 아닐 경우 핵심 텐서 추출
            if not isinstance(batch_features, torch.Tensor):
                if hasattr(batch_features, "image_embeds") and batch_features.image_embeds is not None:
                    batch_features = batch_features.image_embeds
                elif hasattr(batch_features, "pooler_output"):
                    batch_features = batch_features.pooler_output
                elif isinstance(batch_features, (tuple, list)):
                    batch_features = batch_features[0]

            # 2. 차원(Dimension) 크기 검사 및 선택적 투영(Projection)
            if hasattr(backbone, "visual_projection"):
                proj = backbone.visual_projection
                # 현재 특징값의 마지막 차원이 투영 레이어의 '입력 차원(예: 768)'과 일치할 때만 투영
                # 이미 512차원이라면 이 과정을 안전하게 건너뜁니다.
                if batch_features.shape[-1] == proj.in_features:
                    batch_features = proj(batch_features)

            # 3. 최종 정규화
            batch_features = torch.nn.functional.normalize(batch_features, dim=-1)

        features.append(batch_features.cpu())

    return torch.cat(features, dim=0)


def valid_unseen_image_retrieval(model, processor, split="valid_unseen", batch_size=32):
    image_paths = list_image_paths(split)
    labels = labels_from_paths(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found for split: {split}")

    features = encode_images(model, processor, image_paths, batch_size=batch_size)
    similarities = features @ features.T
    similarities.fill_diagonal_(-float("inf"))

    results = {}
    for k in (1, 5, 10):
        topk = similarities.topk(k=min(k, len(image_paths) - 1), dim=1).indices
        correct = 0

        for i, neighbor_indices in enumerate(topk):
            if any(labels[i] == labels[j] for j in neighbor_indices.tolist()):
                correct += 1

        results[f"image_recall@{k}"] = correct / len(image_paths)

    return results, image_paths, labels


def load_caption_rows(split: str):
    try:
        import psycopg2
        from dotenv import load_dotenv
    except ImportError as exc:
        print(f"Caption eval skipped: missing dependency ({exc})")
        return []

    load_dotenv(BASE_DIR / ".env")

    try:
        from database.postgres_final import DATABASE_URL

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT image_path, caption
            FROM cafe_final_images
            WHERE split = %s
            ORDER BY image_path ASC
            """,
            (split,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
    except Exception as exc:
        print(f"Caption eval skipped: DB connection/query failed ({exc})")
        return []

    caption_rows = []
    for image_name, caption in rows:
        if not caption:
            continue

        cafe_id = Path(image_name).stem.split("_")[0]
        image_path = DATA_DIR / split / cafe_id / image_name

        if image_path.exists():
            caption_rows.append((image_path, caption))

    return caption_rows


def caption_contrastive_eval(model, processor, split="valid_unseen", batch_size=4):
    rows = load_caption_rows(split)
    if len(rows) == 0:
        return None

    total_loss = 0.0
    total_count = 0
    image_correct = 0
    text_correct = 0
    batch_count = 0

    for start in tqdm(range(0, len(rows), batch_size), desc="Caption contrastive eval"):
        batch = rows[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path, _ in batch]
        captions = [caption for _, caption in batch]

        inputs = processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )

        labels = torch.arange(len(batch), device=DEVICE)
        loss_img = torch.nn.functional.cross_entropy(outputs.logits_per_image, labels)
        loss_txt = torch.nn.functional.cross_entropy(outputs.logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        image_correct += (outputs.logits_per_image.argmax(dim=1) == labels).sum().item()
        text_correct += (outputs.logits_per_text.argmax(dim=1) == labels).sum().item()
        total_loss += loss.item()
        total_count += len(batch)
        batch_count += 1

    return {
        "caption_loss": total_loss / batch_count,
        "caption_image_acc": image_correct / total_count,
        "caption_text_acc": text_correct / total_count,
        "caption_pairs": total_count,
    }


def print_split_sanity(split):
    train_labels = set(labels_from_paths(list_image_paths("train")))
    split_labels = set(labels_from_paths(list_image_paths(split)))

    print("\n===== Split sanity =====")
    print(f"train classes: {len(train_labels)}")
    print(f"{split} classes: {len(split_labels)}")
    print(f"class overlap with train: {len(train_labels & split_labels)}")
    if split == "valid_unseen" and len(train_labels & split_labels) == 0:
        print("Note: train-vs-valid_unseen class recall is invalid because classes do not overlap.")

def zero_shot_classification_eval(model, processor, image_paths, labels, batch_size=32):
    print(f"Evaluating Zero-shot classification...")
    backbone = clip_backbone(model)
    
    # 1. 평가할 고유 클래스(정답 보기) 목록 추출
    unique_labels = sorted(list(set(labels)))

    # 2. 텍스트 프롬프트 생성 (예: "a photo of cafe 1001")
    prompts = [f"a photo of cafe {label}" for label in unique_labels]

    # 3. 텍스트 임베딩 추출
    text_inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        text_features = backbone.get_text_features(**text_inputs)
        
        # --- 텍스트 텐서 추출 안전장치 ---
        if not isinstance(text_features, torch.Tensor):
            # 텍스트의 경우 'text_embeds' 라는 이름으로 주로 담겨 있습니다.
            if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = text_features.pooler_output
            elif isinstance(text_features, (tuple, list)):
                text_features = text_features[0]
        # ------------------------------------------------

        # 텍스트 투영(Projection) 레이어 처리 (차원 일치를 위한 안전장치)
        if hasattr(backbone, "text_projection"):
            proj = backbone.text_projection
            if text_features.shape[-1] == proj.in_features:
                text_features = proj(text_features)
                
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    # 4. 이미지 임베딩 추출
    text_features = text_features.cpu()
    image_features = encode_images(model, processor, image_paths, batch_size=batch_size)

    # 5. 유사도 계산 및 가장 높은 점수를 받은 클래스 예측
    similarities = image_features @ text_features.T
    predictions = similarities.argmax(dim=1)

    # 6. 정확도(Accuracy) 계산
    correct = 0
    for i, pred_idx in enumerate(predictions.tolist()):
        if unique_labels[pred_idx] == labels[i]:
            correct += 1
            
    return {"zero_shot_acc": correct / len(image_paths)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid_unseen")
    parser.add_argument("--lora-path", type=Path, default=LORA_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--caption-batch-size", type=int, default=4)
    args = parser.parse_args()

    model, processor = load_model(args.lora_path)
    print_split_sanity(args.split)

    retrieval, image_paths, labels = valid_unseen_image_retrieval(
        model,
        processor,
        split=args.split,
        batch_size=args.batch_size,
    )

    caption_result = caption_contrastive_eval(
        model,
        processor,
        split=args.split,
        batch_size=args.caption_batch_size,
    )
    zero_shot_result = zero_shot_classification_eval(
        model, processor, image_paths, labels, batch_size=args.batch_size
    )

    print("\n===== Results =====")
    print(f"split: {args.split}")
    print(f"images: {len(image_paths)}")
    print(f"classes: {len(set(labels))}")
    for name, value in retrieval.items():
        print(f"{name}: {value:.4f}")

    if caption_result is not None:
        for name, value in caption_result.items():
            if isinstance(value, float):
                print(f"{name}: {value:.4f}")
            else:
                print(f"{name}: {value}")

    # =========================
    # Base 모델 비교
    # =========================
    print("\n==============================")
    print("Base CLIP 비교 평가 시작")
    print("==============================")

    base_model, base_processor = load_base_model()

    base_retrieval, _, _ = valid_unseen_image_retrieval(
        base_model,
        base_processor,
        split=args.split,
        batch_size=args.batch_size,
    )

    base_caption = caption_contrastive_eval(
        base_model,
        base_processor,
        split=args.split,
        batch_size=args.caption_batch_size,
    )
    base_zero_shot = zero_shot_classification_eval(
        base_model, base_processor, image_paths, labels, batch_size=args.batch_size
    )
    print("\n===== LoRA vs Base 비교 =====")

    for k in retrieval.keys():
        print(f"{k}: LoRA={retrieval[k]:.4f} | Base={base_retrieval[k]:.4f}")

    print(f"zero_shot_acc: LoRA={zero_shot_result['zero_shot_acc']:.4f} | Base={base_zero_shot['zero_shot_acc']:.4f}")
    if caption_result and base_caption:
        for k in caption_result.keys():
            if isinstance(caption_result[k], float):
                print(f"{k}: LoRA={caption_result[k]:.4f} | Base={base_caption[k]:.4f}")
            else:
                print(f"{k}: LoRA={caption_result[k]} | Base={base_caption[k]}")
if __name__ == "__main__":
    main()
