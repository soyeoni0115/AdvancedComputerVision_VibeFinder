import os
import psycopg2
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from model_utils import get_lora_clip_model

# ---------------------------
# 0. 설정
# ---------------------------
IMAGE_DIR = "cafe_images"  # 🔥 이미지 폴더

DB_CONFIG = {
    "host": "ep-xxx.neon.tech",
    "database": "neondb",
    "user": "neondb_owner",
    "password": "npg_eHtYc0ABqF5k",
    "sslmode": "require"
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# 1. Dataset (DB 기반)
# ---------------------------
class CafeDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT image_path, caption FROM cafe_images")
        self.data = cur.fetchall()

        cur.close()
        conn.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name, caption = self.data[idx]

        # 🔥 파일명 + 폴더 합치기
        full_path = os.path.join(IMAGE_DIR, image_name)

        # 🔥 예외 처리
        if not os.path.exists(full_path):
            return self.__getitem__((idx + 1) % len(self.data))

        image = Image.open(full_path).convert("RGB")
        image = image.resize((224, 224))  # 안정성

        inputs = self.processor(
            text=[caption],
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {k: v.squeeze(0) for k, v in inputs.items()}

# ---------------------------
# 2. 학습 함수
# ---------------------------
def train_lora_clip():
    BATCH_SIZE = 4
    EPOCHS = 3
    LR = 5e-5

    model, processor = get_lora_clip_model()
    model.to(device)

    # 🔥 LoRA만 학습
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    dataset = CafeDataset(processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    model.train()
    print("🚀 LoRA 학습 시작")

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()

            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values']
            )

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text

            labels = torch.arange(len(logits_per_image)).to(device)

            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"📉 Epoch {epoch+1}: {total_loss / len(dataloader):.4f}")

    # ---------------------------
    # 저장
    # ---------------------------
    model.save_pretrained("models/lora_weights")
    print("✅ LoRA 저장 완료")

# ---------------------------
if __name__ == "__main__":
    train_lora_clip()