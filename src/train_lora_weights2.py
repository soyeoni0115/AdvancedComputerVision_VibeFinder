import csv
from pathlib import Path

import psycopg2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model_utils import get_lora_clip_model
from database.postgres_final import DATABASE_URL
#lora_weights2 때의 코드-복구함
# ---------------------------
# 0. 설정
# ---------------------------
IMAGE_DIR = "cafe_final_images"


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_AUG_DIR = PROJECT_ROOT / "data" / "train_aug"
device = "cuda" if torch.cuda.is_available() else "cpu"


class CafeDataset(Dataset):
    def __init__(self, processor, split):
        self.processor = processor
        self.split = split

        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        cur.execute("""
            SELECT image_path, caption
            FROM cafe_final_images
            WHERE split = %s
        """, (split,))
        rows = cur.fetchall()

        cur.close()
        conn.close()

        self.data = []

        for image_name, caption in rows:
            cafe_id = image_name.split("_")[0]

            if split == "train":
                img_stem = Path(image_name).stem
                aug_dir = TRAIN_AUG_DIR / cafe_id
                aug_images = sorted(aug_dir.glob(f"{img_stem}*.jpg"))

                for aug_path in aug_images:
                    self.data.append((aug_path, caption))
            else:
                img_path = IMAGE_DIR / split / cafe_id / image_name
                self.data.append((img_path, caption))

        print(f"{split} data count: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        full_path, caption = self.data[idx]

        if not full_path.exists():
            return None

        try:
            image = Image.open(full_path).convert("RGB")
            image = image.resize((224, 224))
        except:
            return None

        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}


def collate_fn(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}


def get_clip_loss(model, batch):
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
    )

    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text

    labels = torch.arange(len(logits_per_image)).to(device)

    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2

    return loss


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    image_correct = 0
    text_correct = 0
    total_count = 0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
            )

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(len(logits_per_image)).to(device)

            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            image_pred = logits_per_image.argmax(dim=1)
            text_pred = logits_per_text.argmax(dim=1)

            image_correct += (image_pred == labels).sum().item()
            text_correct += (text_pred == labels).sum().item()
            total_count += len(labels)

            total_loss += loss.item()
            batch_count += 1

    model.train()

    if batch_count == 0:
        return None

    return {
        "loss": total_loss / batch_count,
        "image_acc": image_correct / total_count,
        "text_acc": text_correct / total_count,
    }

def train_lora_clip():
    BATCH_SIZE = 4
    EPOCHS = 20
    LR = 5e-5
    MIN_EPOCHS = 10
    PATIENCE = 5

    model, processor = get_lora_clip_model()
    model.to(device)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    train_dataset = CafeDataset(processor, "train")
    valid_seen_dataset = CafeDataset(processor, "valid_seen")
    valid_unseen_dataset = CafeDataset(processor, "valid_unseen")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_seen_loader = DataLoader(
        valid_seen_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )
    valid_unseen_loader = DataLoader(
        valid_unseen_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    model.train()
    print("LoRA training start")

    best_valid_loss = float("inf")
    patience_count = 0
    save_path = PROJECT_ROOT / "models" / "lora_weights2"

    for epoch in range(EPOCHS):
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = get_clip_loss(model, batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            train_loss = total_loss / batch_count
            print(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")
        else:
            print(f"Epoch {epoch + 1}: no valid batch")

        valid_seen_result = evaluate(model, valid_seen_loader)
        valid_unseen_result = evaluate(model, valid_unseen_loader)

        if valid_seen_result is not None:
            print(
                f"Epoch {epoch + 1} valid_seen loss: {valid_seen_result['loss']:.4f}, "
                f"image acc: {valid_seen_result['image_acc']:.4f}, "
                f"text acc: {valid_seen_result['text_acc']:.4f}"
            )

        if valid_unseen_result is not None:
            print(
                f"Epoch {epoch + 1} valid_unseen loss: {valid_unseen_result['loss']:.4f}, "
                f"image acc: {valid_unseen_result['image_acc']:.4f}, "
                f"text acc: {valid_unseen_result['text_acc']:.4f}"
            )

            if valid_unseen_result["loss"] < best_valid_loss:
                best_valid_loss = valid_unseen_result["loss"]
                patience_count = 0
                model.save_pretrained(save_path)
                print("best model saved")
            elif epoch + 1 >= MIN_EPOCHS:
                patience_count += 1
                print(f"early stopping count: {patience_count}/{PATIENCE}")
            else:
                print("early stopping check skipped")

            if epoch + 1 >= MIN_EPOCHS and patience_count >= PATIENCE:
                print("early stopping")
                break

    print("LoRA training complete")

if __name__ == "__main__":
    train_lora_clip()
