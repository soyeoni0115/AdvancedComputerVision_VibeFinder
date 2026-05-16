from src.utils import CLIPEmbedding
from src.search_engine import SimpleFaissDB
import os

# 1. 초기화
embedder = CLIPEmbedding()

db = SimpleFaissDB(dimension=512)

# 2. 샘플 이미지
sample_image = "data/raw/c1.jpg" 

if os.path.exists(sample_image):
    # 3. 이미지에서 512차원 벡터 뽑기
    embedding = embedder.get_embedding(sample_image) # 만들어진 인스턴스의 함수 이용해서 이미지 파일을 숫자로 변환
    print(f"벡터 추출 성공! 모양: {embedding.shape}")

    # 4. FAISS DB에 저장하기
    db.add_vectors(embedding, sample_image)

    # 5. DB를 파일로 굽기
    db.save()

    # 6. 똑같은 벡터로 다시 검색해보기
    print("\n--- DB 검색 테스트 ---")
    search_results = db.search(embedding, k=1)
    print(f"검색된 유사 이미지 경로: {search_results}")
else:
    print(f"{sample_image} 파일이 없습니다. 경로를 확인해주세요.")