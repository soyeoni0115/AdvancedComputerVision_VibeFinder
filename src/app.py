from pathlib import Path

import faiss
import numpy as np
import psycopg2
import streamlit as st
import torch

from model_utils import get_lora_clip_model


st.set_page_config(page_title="Vibe Finder", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_DIR = BASE_DIR / "data" / "raw"
INDEX_PATH = BASE_DIR / "faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"
LORA_PATH = BASE_DIR / "models" / "lora_weights"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource(show_spinner="CLIP 모델을 불러오는 중입니다...")
def load_model():
    model, processor = get_lora_clip_model()

    if LORA_PATH.exists():
        model.load_adapter(str(LORA_PATH), adapter_name="default")
        model.set_adapter("default")

    model.to(DEVICE)
    model.eval()
    return model, processor


@st.cache_resource
def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS 인덱스를 찾을 수 없습니다: {INDEX_PATH}")
    return faiss.read_index(str(INDEX_PATH))


@st.cache_data
def load_image_paths():
    if not PATHS_PATH.exists():
        return []
    return [str(path) for path in np.load(PATHS_PATH, allow_pickle=True).tolist()]


@st.cache_resource
def get_db_connection():
    return psycopg2.connect(
        dbname="neondb",
        user="neondb_owner",
        password="npg_eHtYc0ABqF5k",
        host="ep-misty-mud-aogsqtmk-pooler.c-2.ap-southeast-1.aws.neon.tech",
        port="5432",
        sslmode="require",
        connect_timeout=5,
    )


def encode_text(model, processor, text):
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
        text_features = text_features.text_embeds
    elif hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
        text_features = text_features.pooler_output

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features.cpu().numpy().astype("float32")


def resolve_image_path(image_path):
    path = Path(image_path)
    if path.is_absolute():
        return path

    base_relative = BASE_DIR / path
    if base_relative.exists():
        return base_relative

    return RAW_IMAGE_DIR / path.name


def fetch_cafe_from_db(vector_id):
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.cafe_name, c.address, c.map_url, ci.image_path, ci.caption
                FROM cafe_images ci
                JOIN cafes c ON ci.cafe_id = c.id
                WHERE ci.id = %s
                """,
                (int(vector_id),),
            )
            row = cur.fetchone()
    except Exception:
        return None

    if not row:
        return None

    return {
        "cafe_name": row[0],
        "address": row[1],
        "map_url": row[2],
        "image_path": row[3],
        "caption": row[4],
    }


def build_local_result(vector_id, image_paths):
    if vector_id < 0 or vector_id >= len(image_paths):
        return None

    image_path = image_paths[vector_id]
    return {
        "cafe_name": Path(image_path).stem,
        "address": "로컬 이미지 결과",
        "map_url": "",
        "image_path": image_path,
        "caption": "",
    }


def search_cafes(model, processor, index, image_paths, query, top_k=5):
    query_embedding = encode_text(model, processor, query)
    limit = min(top_k, max(index.ntotal, 1))
    distances, indices = index.search(query_embedding, k=limit)

    results = []
    for vector_id, score in zip(indices[0], distances[0]):
        vector_id = int(vector_id)
        if vector_id == -1:
            continue

        result = fetch_cafe_from_db(vector_id) or build_local_result(vector_id, image_paths)
        if result:
            result["score"] = float(score)
            results.append(result)

    return results


st.title("Vibe Finder")
st.caption("원하는 분위기를 입력하면 가장 가까운 카페 이미지를 찾아줍니다.")

try:
    model, processor = load_model()
    index = load_index()
    image_paths = load_image_paths()
except Exception as exc:
    st.error(f"앱을 시작하는 중 오류가 발생했습니다: {exc}")
    st.stop()

vibe = st.text_input(
    "원하는 분위기를 입력하세요",
    placeholder="예: 조용하고 감성적인 골목 카페",
)

if st.button("검색 시작", type="primary"):
    query = vibe.strip()
    if not query:
        st.warning("검색어를 입력해 주세요.")
        st.stop()

    try:
        results = search_cafes(model, processor, index, image_paths, query)
    except Exception as exc:
        st.error(f"검색 중 오류가 발생했습니다: {exc}")
        st.stop()

    if not results:
        st.warning("검색 결과가 없습니다. FAISS 인덱스와 paths.npy가 같은 순서로 생성되었는지 확인해 주세요.")
        st.stop()

    st.write("### 추천 카페")

    for cafe in results:
        image_file = resolve_image_path(cafe["image_path"])
        col1, col2 = st.columns([1, 2])

        with col1:
            if image_file.exists():
                st.image(str(image_file), use_container_width=True)
            else:
                st.warning(f"이미지를 찾을 수 없습니다: {image_file}")

        with col2:
            st.subheader(cafe["cafe_name"])
            st.write(f"주소: {cafe['address']}")
            if cafe["map_url"]:
                st.write(f"[지도 보기]({cafe['map_url']})")
            if cafe["caption"]:
                st.caption(cafe["caption"])
