import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedding:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # CLIP 모델 및 전처리기 로드
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def get_embedding(self, image_path):
        """📸 실제 이미지 파일을 512차원 벡터로 변환"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            # ⭐️ pooler_output으로 텐서만 정확히 추출
            image_features = self.model.get_image_features(**inputs).pooler_output
            
        # 1차원 numpy 배열로 변환
        embedding = image_features.cpu().numpy()[0]
        
        # 복잡하고 에러 나던 정규화 코드를 안전한 numpy 식으로 통일
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)
        return embedding

    def get_text_embedding(self, text_query):
        """✍️ 유저가 입력한 자연어 문장을 512차원 벡터로 변환"""
        inputs = self.processor(text=[text_query], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            # ⭐️ 텍스트도 똑같이 pooler_output을 붙여주어야 에러가 안 납니다!
            text_features = self.model.get_text_features(**inputs).pooler_output

        # 1차원 numpy 배열로 변환
        embedding = text_features.cpu().numpy()[0]
        
        # L2 정규화 수행
        embedding = embedding / (np.linalg.norm(embedding) + 1e-5)
        return embedding