from pathlib import Path

import faiss
import numpy as np
import psycopg2
import streamlit as st
import torch
from database.postgres_new import DATABASE_URL
from model_utils import get_lora_clip_model

from PIL import Image

st.set_page_config(page_title="Vibe Finder", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_DIR = BASE_DIR / "data" / "raw"
INDEX_PATH = BASE_DIR / "faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"
LORA_PATH = BASE_DIR / "models" / "lora_weights5"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_CONFIG = DATABASE_URL


@st.cache_resource(show_spinner="CLIP 모델을 불러오는 중입니다...")
def load_model():
    model, processor = get_lora_clip_model()

    if LORA_PATH.exists():
        model.load_adapter(str(LORA_PATH), adapter_name="lora_adapter")
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
    if isinstance(DB_CONFIG, str):
        return psycopg2.connect(DB_CONFIG)
    return psycopg2.connect(**DB_CONFIG)


def run_query(query, params=()):
    conn = get_db_connection()
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


@st.cache_data(ttl=300)
def load_available_tags():
    try:
        rows = run_query(
            """
            SELECT DISTINCT tag_name
            FROM tags
            WHERE tag_name IS NOT NULL AND btrim(tag_name) <> ''
            ORDER BY tag_name
            """
        )
    except Exception:
        return []

    return [row[0] for row in rows]


def crop_to_16_9(image):
    width, height = image.size
    target_ratio = 16 / 9
    current_ratio = width / height

    if current_ratio > target_ratio:
        # 가로가 더 김 → 좌우 자르기
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        return image.crop((left, 0, right, height))
    else:
        # 세로가 더 김 → 위아래 자르기
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        return image.crop((0, top, width, bottom))

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
    path = Path(str(image_path))
    if path.is_absolute():
        return path

    base_relative = BASE_DIR / path
    if base_relative.exists():
        return base_relative

    return RAW_IMAGE_DIR / path.name


def normalize_result(row, matched_tags=None):
    return {
        "cafe_id": row[0],
        "cafe_name": row[1],
        "location": row[2] or "",
        "map_url": row[3] or "",
        "image_path": row[4] or "",
        "caption": row[5] or "",
        "matched_tags": matched_tags or [],
    }


def fetch_cafe_by_image_path(image_path):
    file_name = Path(str(image_path)).name
    rows = run_query(
        """
        SELECT c.id, c.cafe_name, c.location, c.map_url, ci.image_path, ci.caption
        FROM cafe_images ci
        JOIN cafes c ON ci.cafe_id = c.id
        WHERE ci.image_path = %s
           OR ci.image_path LIKE %s
        ORDER BY ci.id
        LIMIT 1
        """,
        (str(image_path), f"%{file_name}"),
    )
    return normalize_result(rows[0]) if rows else None


def fetch_cafes_by_image_paths(image_paths):
    if not image_paths:
        return {}

    path_pairs = [(str(path), Path(str(path)).name) for path in image_paths]
    rows = run_query(
        """
        WITH requested(full_path, file_name, ord) AS (
            SELECT *
            FROM unnest(%s::text[], %s::text[]) WITH ORDINALITY
        ),
        matched AS (
            SELECT DISTINCT ON (r.ord)
                r.full_path,
                c.id,
                c.cafe_name,
                c.location,
                c.map_url,
                ci.image_path,
                ci.caption
            FROM requested r
            JOIN cafe_images ci
              ON ci.image_path = r.full_path
              OR ci.image_path LIKE '%%' || r.file_name
            JOIN cafes c ON ci.cafe_id = c.id
            ORDER BY r.ord, ci.id
        )
        SELECT full_path, id, cafe_name, location, map_url, image_path, caption
        FROM matched
        """,
        ([pair[0] for pair in path_pairs], [pair[1] for pair in path_pairs]),
    )

    return {row[0]: normalize_result(row[1:]) for row in rows}


def build_local_result(vector_id, image_paths):
    if vector_id < 0 or vector_id >= len(image_paths):
        return None

    image_path = image_paths[vector_id]
    return {
        "cafe_id": None,
        "cafe_name": Path(image_path).stem,
        "location": "로컬 이미지 결과",
        "map_url": "",
        "image_path": image_path,
        "caption": "",
        "matched_tags": [],
    }


def search_by_text(model, processor, index, image_paths, query, top_k=50):
    if not query.strip():
        return []

    query_embedding = encode_text(model, processor, query)
    limit = min(top_k, max(index.ntotal, 1))
    distances, indices = index.search(query_embedding, k=limit)

    candidates = []
    for vector_id, score in zip(indices[0], distances[0]):
        vector_id = int(vector_id)
        if vector_id == -1:
            continue

        image_path = image_paths[vector_id] if vector_id < len(image_paths) else ""
        candidates.append((vector_id, image_path, float(score)))

    try:
        db_results = fetch_cafes_by_image_paths(
            [image_path for _, image_path, _ in candidates if image_path]
        )
    except Exception:
        db_results = {}

    results = []
    for vector_id, image_path, score in candidates:
        result = db_results.get(str(image_path))
        result = result or build_local_result(vector_id, image_paths)
        if result:
            result["score"] = score
            results.append(result)

    return results


def search_by_tags(selected_tags, top_k=60):
    if not selected_tags:
        return []

    rows = run_query(
        """
        WITH matched AS (
            SELECT
                c.id AS cafe_id,
                c.cafe_name,
                c.location,
                c.map_url,
                array_agg(DISTINCT t.tag_name ORDER BY t.tag_name) AS matched_tags,
                count(DISTINCT t.tag_name) AS match_count
            FROM cafes c
            JOIN tags t ON t.cafe_id = c.id
            WHERE t.tag_name = ANY(%s)
            GROUP BY c.id, c.cafe_name, c.location, c.map_url
        ),
        first_image AS (
            SELECT DISTINCT ON (ci.cafe_id)
                ci.cafe_id,
                ci.image_path,
                ci.caption
            FROM cafe_images ci
            ORDER BY ci.cafe_id, ci.id
        )
        SELECT
            m.cafe_id,
            m.cafe_name,
            m.location,
            m.map_url,
            fi.image_path,
            fi.caption,
            m.matched_tags,
            m.match_count
        FROM matched m
        LEFT JOIN first_image fi ON fi.cafe_id = m.cafe_id
        ORDER BY m.match_count DESC, m.cafe_name
        LIMIT %s
        """,
        (selected_tags, top_k),
    )

    return [
        normalize_result(row[:6], matched_tags=list(row[6] or []))
        for row in rows
    ]


def merge_results(*result_groups):
    merged = []
    seen = set()

    for group in result_groups:
        for result in group:
            key = result.get("cafe_id") or result.get("image_path") or result.get("cafe_name")
            if key in seen:
                continue
            seen.add(key)
            merged.append(result)

    return merged


def init_session_state():
    defaults = {
        "recommended_tags": [],
        "selected_tags": [],
        "search_results": [],
        "search_clicked": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f5f5f5;
        }

        .title-container {
            min-height: 78vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 150px;
        }

        .title {
            font-size: clamp(64px, 6vw, 112px);
            font-family: Georgia, 'Times New Roman', serif;
            line-height: 0.9;
            opacity: 0;
            transform: translateY(50px);
            animation: fadeUp 1s ease-out forwards;
        }

        @keyframes fadeUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .center-box {
            margin-top: 120px;
        }

        .desc {
            font-size: 34px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 16px;
        }

        .subdesc {
            font-size: 22px;
            text-align: center;
            margin-bottom: 36px;
            color: #555;
        }

        .stTextInput input {
            background-color: #c7d8d1;
            border-radius: 6px;
            height: 50px;
            font-size: 18px;
        }

        .button-container {
            display: flex;
            justify-content: flex-end;
            margin-top: 24px;
        }

        .selected-tags {
            line-height: 2;
            color: #245447;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_result(cafe):
    image_file = resolve_image_path(cafe["image_path"])
    col_img, col_info = st.columns([1, 2])

    with col_img:
        if cafe["image_path"] and image_file.exists():
            img = Image.open(image_file)
            img = crop_to_16_9(img)
            st.image(img, use_container_width=True)
        elif cafe["image_path"]:
            st.warning(f"이미지를 찾을 수 없습니다: {cafe['image_path']}")
        else:
            st.info("등록된 이미지가 없습니다.")

    with col_info:
        st.subheader(cafe["cafe_name"])
        st.write(f"주소: {cafe['location']}")
        if cafe["map_url"]:
            st.write(f"[지도 보기]({cafe['map_url']})")
        if cafe["matched_tags"]:
            st.write("일치한 태그: " + " ".join(f"#{tag}" for tag in cafe["matched_tags"]))
        if cafe["caption"]:
            st.caption(cafe["caption"])


init_session_state()
render_styles()
available_tags = []

left, right = st.columns([1, 2])

with left:
    st.markdown(
        """
        <div class="title-container">
            <div class="title">Vibe<br>Finder</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="center-box">', unsafe_allow_html=True)
    st.markdown('<div class="desc">원하는 분위기를 입력하세요</div>', unsafe_allow_html=True)
    st.markdown('<div class="subdesc">분위기에 맞는 카페를 추천해드립니다</div>', unsafe_allow_html=True)

    vibe = st.text_input(
        "분위기 입력",
        placeholder="예: 조용하고 감성적인, 공부하기 좋은, 디저트가 맛있는",
        label_visibility="collapsed",
    )

    if st.button("분위기 추천 키워드로 골라보기"):
        available_tags = load_available_tags()
        st.session_state.recommended_tags = available_tags
        if not available_tags:
            st.warning("PostgreSQL tags 테이블에 표시할 태그가 아직 없습니다.")

    if st.session_state.recommended_tags:
        st.write("### 추천 태그")
        cols = st.columns(5)

        for idx, tag in enumerate(st.session_state.recommended_tags):
            with cols[idx % 5]:
                selected = tag in st.session_state.selected_tags
                button_type = "primary" if selected else "secondary"

                if st.button(
                    f"#{tag}",
                    key=f"tag_{tag}",
                    type=button_type,
                    use_container_width=True,
                ):
                    if selected:
                        st.session_state.selected_tags.remove(tag)
                    else:
                        st.session_state.selected_tags.append(tag)
                    st.rerun()

    if st.session_state.selected_tags:
        selected_text = " ".join(f"#{tag}" for tag in st.session_state.selected_tags)
        st.write("### 선택한 태그")
        st.markdown(f'<div class="selected-tags">{selected_text}</div>', unsafe_allow_html=True)

    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    search_clicked = st.button("검색하기", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if search_clicked:
        st.session_state.search_clicked = True
        query = " ".join([vibe.strip(), *st.session_state.selected_tags]).strip()

        if not query:
            st.warning("검색어를 입력하거나 태그를 선택해 주세요.")
            st.session_state.search_results = []
        else:
            try:
                model, processor = load_model()
                index = load_index()
                image_paths = load_image_paths()
                tag_results = search_by_tags(st.session_state.selected_tags)
                text_results = search_by_text(model, processor, index, image_paths, query)
                st.session_state.search_results = merge_results(tag_results, text_results)
            except Exception as exc:
                st.error(f"검색 중 오류가 발생했습니다: {exc}")
                st.session_state.search_results = []

    if st.session_state.search_results:
    # 🔹 추가: 슬라이더 (추천 카페 위)
        num_results = st.slider(
            "보고 싶은 카페 개수",
            1,
            len(st.session_state.search_results),
            min(20, len(st.session_state.search_results))
        )

        st.write("### 추천 카페")

        # 🔹 수정: 출력 개수 제한
        for cafe in st.session_state.search_results[:num_results]:
            render_result(cafe)
    elif st.session_state.search_clicked:
        st.info("조건에 맞는 카페를 찾지 못했습니다.")

    st.markdown("</div>", unsafe_allow_html=True)
