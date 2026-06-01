# 내부 사진 폴더(data/raw)의 이미지들을 전부 읽어서 FAISS DB 파일(faiss_vibe.index)로 구워주는 빌드 파일
# 이걸 한 번 실행godi 검색이 가능
# postgres cafe images table에 새로 생긴 캡션 추가해야 함
import os
import json
from utils import CLIPEmbedding
from search_engine import SimpleFaissDB

def build_vector_db():
    print("이미지 임베딩 및 FAISS DB 구축 시작...")
    
    # 클래스 초기화
    clip = CLIPEmbedding()
    faiss_db = SimpleFaissDB(dimension=512)
    
    # 메타데이터 로드
    metadata_path = "data/metadata_updated.json"
    if not os.path.exists(metadata_path):
        print("파일이 없습니다. 크롤링을 먼저 진행해주세요.")
        return
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        db_metadata = json.load(f)

    all_embeddings = []
    all_paths = []

    # 메타데이터에 등록된 실제 정제 이미지들만 순회하며 임베딩
    for cafe_key, cafe_info in db_metadata.items():
        for img_name in cafe_info.get("images", []):
            img_path = f"data/raw/{img_name}"
            
            if os.path.exists(img_path):
                try:
                    print(f"🔤 임베딩 중: {img_name} ({cafe_info['cafe_name']})")
                    embedding = clip.get_embedding(img_path)
                    all_embeddings.append(embedding)
                    all_paths.append(img_path)
                except Exception as e:
                    print(f"임베딩 실패 ({img_name}): {e}")
            else:
                print(f"사진 유실 패스: {img_path}")

    if all_embeddings:
        # FAISS DB에 일괄 저장 및 로컬 파일 보관
        faiss_db.add_vectors(all_embeddings, all_paths)
        faiss_db.save(index_path="faiss_vibe.index", paths_path="paths.npy")
        print("FAISS 벡터 DB 구축이 완료되었습니다!")
    else:
        print("저장할 벡터 데이터가 없습니다.")

if __name__ == "__main__":
    build_vector_db()