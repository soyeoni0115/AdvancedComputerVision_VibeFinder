# 오리지널 CLIP-ViT-B/32 모델을 불러와 이미지 벡터를 뽑는 역할
# 실제 이미지 파일 경로를 받아서 숫자로 변환(임베딩)

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

#이미지 임베딩
class CLIPEmbedding:
    def __init__(self):

        # GPU 있으면 cuda, 없으면 cpu
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using device: {self.device}")

        # CLIP 모델 로드
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(self.device)

        # 이미지 전처리기
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # 추론 모드
        self.model.eval()

    def get_embedding(self, image_path):

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        # grad: 데이터를 AI 모델에 집어넣어 이미지의 특징을 추출
        embedding = image_features.pooler_output.cpu().numpy()[0]
        # 배열로 바꾼 후 변수에 저장

        return embedding # 계산된 512차원 배열(값)을 넘겨줌
    

 # 텍스트 임베딩 추가
    def get_text_embedding(self, text):

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        embedding = text_features.cpu().numpy()[0]

        # 정규화
        embedding = embedding / np.linalg.norm(embedding)

        return embedding.astype(np.float32)