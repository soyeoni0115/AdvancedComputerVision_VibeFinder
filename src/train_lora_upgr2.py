import csv

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from train_lora_first import (
    CafeDataset,
    PROJECT_ROOT,
    collate_fn,
    device,
    evaluate,
    get_clip_loss,
)

BASE_MODEL = "openai/clip-vit-base-patch32"

START_LORA_PATH = PROJECT_ROOT / "models" / "lora_weights4"
SAVE_PATH = PROJECT_ROOT / "models" / "lora_weights5"
LOG_PATH = PROJECT_ROOT / "models" / "lora_continue2_log.csv"


def train_lora_continue():
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 1e-6   # ↓ 낮춤
    MIN_EPOCHS = 2
    PATIENCE = 2   # ↓ 빠르게 끊기

    base_model = CLIPModel.from_pretrained(BASE_MODEL)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)

    model = PeftModel.from_pretrained(
        base_model,
        START_LORA_PATH,
        is_trainable=True
    )
    model.to(device)

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

    # optimizer 개선
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,   # upgr1과 다르게 새롭게 추가
    )

    # scheduler 추가
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_valid_loss = float("inf")
    patience_count = 0

    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "valid_seen_loss",
            "valid_seen_image_acc",
            "valid_seen_text_acc",
            "valid_unseen_loss",
            "valid_unseen_image_acc",
            "valid_unseen_text_acc",
            "is_best",
        ])

    print("LoRA continue training start")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0
        batch_count = 0
        is_best = False
        stop_now = False

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            if batch is None:
                continue

            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            loss = get_clip_loss(model, batch)
            loss.backward()

            # gradient clipping (안정성)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()  

        train_loss = total_loss / batch_count if batch_count > 0 else None
        print(f"Epoch {epoch + 1} train loss: {train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            valid_seen_result = evaluate(model, valid_seen_loader)
            valid_unseen_result = evaluate(model, valid_unseen_loader)

        print(
            f"Epoch {epoch + 1} valid_seen loss: {valid_seen_result['loss']:.4f}, "
            f"image acc: {valid_seen_result['image_acc']:.4f}, "
            f"text acc: {valid_seen_result['text_acc']:.4f}"
        )

        print(
            f"Epoch {epoch + 1} valid_unseen loss: {valid_unseen_result['loss']:.4f}, "
            f"image acc: {valid_unseen_result['image_acc']:.4f}, "
            f"text acc: {valid_unseen_result['text_acc']:.4f}"
        )

        # early stopping 기준 유지
        if valid_unseen_result["loss"] < best_valid_loss:
            is_best = True
            best_valid_loss = valid_unseen_result["loss"]
            patience_count = 0
            model.save_pretrained(SAVE_PATH)
            print("best continued model saved")
        else:
            if epoch + 1 >= MIN_EPOCHS:
                patience_count += 1
                print(f"early stopping count: {patience_count}/{PATIENCE}")

        if epoch + 1 >= MIN_EPOCHS and patience_count >= PATIENCE:
            print("early stopping")
            stop_now = True

        with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                valid_seen_result["loss"],
                valid_seen_result["image_acc"],
                valid_seen_result["text_acc"],
                valid_unseen_result["loss"],
                valid_unseen_result["image_acc"],
                valid_unseen_result["text_acc"],
                is_best,
            ])

        if stop_now:
            break

    print("LoRA continue training complete")


if __name__ == "__main__":
    train_lora_continue()