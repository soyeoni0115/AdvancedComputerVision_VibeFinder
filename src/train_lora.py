import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from model_utils import get_lora_clip_model  # 아까 만든 LoRA 연결 함수

#######임시############################################################
# 1. 커스텀 카페 데이터셋 정의
class CafeDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.df = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # D드라이브에 저장된 이미지 로드
        image = Image.open(row['image_path']).convert("RGB")
        caption = row['caption']
        
        # CLIP 입력 규격(224x224 등)에 맞게 이미지와 텍스트 전처리
        inputs = self.processor(text=[caption], images=image, return_tensors="pt", padding="max_length", truncation=True)
        
        # Squeeze를 통해 배치 차원 정렬 준비
        return {k: v.squeeze(0) for k, v in inputs.items()}

# 2. 파인튜닝 메인 루프
def train_lora_clip():
    # ⚙️ 하이퍼파라미터 설정 (3학년 학부 플젝 CPU 최적화 사양)
    BATCH_SIZE = 4  # CPU 메모리를 고려해 작게 설정
    EPOCHS = 3      # 3~5 에포크면 충분히 도메인 특징을 잡습니다
    LEARNING_RATE = 5e-5

    # 모델과 프로세서 로드 (model_utils에서 가져옴)
    model, processor = get_lora_clip_model()
    
    # 데이터로더 세팅
    dataset = CafeDataset(csv_file="data/dataset.csv", processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 최적화 도구 (오직 LoRA 레이어의 가중치만 업데이트하도록 설정됨)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    print("🚀 카페 데이터셋으로 LoRA 파인튜닝 시작...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            
            # CLIP에 이미지와 텍스트 동시에 입력
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch['pixel_values']
            )
            
            # 대조 학습(Contrastive Loss) 계산
            # 이미지와 텍스트가 1:1로 매칭되는 대각선 행렬이 정답이 됩니다.
            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            
            # Ground Truth 레이블 생성 (배치 내에서 자기 짝을 찾도록 설정)
            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
            
            loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
            loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
            loss = (loss_img + loss_txt) / 2
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"📉 Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
    # 💾 학습이 끝난 가벼운 LoRA 가중치만 따로 저장! (수십 MB 수준)
    model.save_pretrained("models/lora_weights")
    print("✅ LoRA 가중치 저장 완료! -> models/lora_weights")

if __name__ == "__main__":
    train_lora_clip()