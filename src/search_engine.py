# 이미지 벡터와 진짜 사진 파일 경로를 매핑해서 로컬에 저장하고 검색할 수 있는 뼈대
import faiss
import numpy as np
import os

class SimpleFaissDB:
    def __init__(self, dimension=512):
        # 512차원 벡터를 저장할 기본 L2(유클리드 거리) 인덱스 생성
        self.index = faiss.IndexFlatL2(dimension)
        # 인덱스 번호(0, 1, 2...)와 실제 이미지 파일 경로를 매핑할 리스트
        self.image_paths = []

    def add_vectors(self, embeddings, paths):
        """
        embeddings: np.array 형태의 (N, 512) 또는 단일 (512,) 데이터
        paths: 이미지 경로 리스트 [N개] 또는 단일 경로 문자열
        """
        # 데이터가 단건(1개)으로 들어왔을 때 처리
        if isinstance(paths, str):
            paths = [paths]
            embeddings = np.array([embeddings])
        else:
            embeddings = np.array(embeddings)

        # FAISS는 반드시 float32 타입
        embeddings = embeddings.astype('float32')
        
        # FAISS DB에 벡터 추가
        self.index.add(embeddings)
        # 경로 리스트에 경로 추가 (순서 중요)
        self.image_paths.extend(paths)
        print(f" 현재 DB에 총 {self.index.ntotal}개의 이미지가 저장되어 있습니다.")

    def save(self, index_path="faiss_vibe.index", paths_path="paths.npy"):
        """로컬 파일로 DB 저장"""
        faiss.write_index(self.index, index_path)
        np.save(paths_path, self.image_paths)
        print("FAISS DB가 로컬에 성공적으로 저장되었습니다.")

    def load(self, index_path="faiss_vibe.index", paths_path="paths.npy"):
        """로컬 파일에서 기존 DB 불러오기"""
        if os.path.exists(index_path) and os.path.exists(paths_path):
            self.index = faiss.read_index(index_path)
            self.image_paths = np.load(paths_path).tolist()
            print(" 로컬에서 기존 DB를 성공적으로 불러왔습니다.")
        else:
            print("기존 DB 파일이 없어 새로운 DB로 시작합니다.")

    def search(self, query_embedding, k=3):
        """
        query_embedding: 검색할 이미지의 [512] 벡터
        k: 가장 유사한 이미지 몇 개를 뽑을 것인지
        """
        # 검색할 벡터도 float32 형태의 2차원 배열로 변환
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # distances: 유사도 거리, indices: 매칭된 벡터의 고유 번호(ID)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.image_paths):
                results.append(self.image_paths[idx])
                
        return results