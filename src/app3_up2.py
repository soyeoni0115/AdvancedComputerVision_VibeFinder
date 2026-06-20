from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import faiss
import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
import torch
from database.postgres_final import DATABASE_URL
from model_utils import get_lora_clip_model
from query_expander import expand_query
from PIL import Image

st.set_page_config(page_title="Vibe Finder", layout="wide")

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_IMAGE_DIR = BASE_DIR / "data" / "raw"
INDEX_PATH = BASE_DIR / "faiss_vibe.index"
PATHS_PATH = BASE_DIR / "paths.npy"
LORA_PATH = BASE_DIR / "models" / "lora_weights3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DB_CONFIG = DATABASE_URL


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

def run_query(query, params=()):
    if isinstance(DB_CONFIG, str):
        conn = psycopg2.connect(DB_CONFIG)
    else:
        conn = psycopg2.connect(**DB_CONFIG)
    with conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()




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


def normalize_result(row):
    return {
        "cafe_id": row[0],
        "cafe_name": row[1],
        "location": row[2] or "",
        "map_url": row[3] or "",
        "image_path": row[4] or "",
        "caption": row[5] or "",
    }


def fetch_cafe_by_image_path(image_path):
    file_name = Path(str(image_path)).name
    rows = run_query(
        """
        SELECT c.id, c.cafe_name, c.location, c.map_url, ci.image_path, ci.caption
        FROM cafe_final_images ci
        JOIN cafes_final c ON ci.cafe_id = c.id
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
            JOIN cafe_final_images ci
            ON ci.image_path = r.full_path
            OR ci.image_path LIKE '%%' || r.file_name
            JOIN cafes_final c
            ON ci.cafe_id = c.id
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
    except Exception as e:
        st.error(f"DB 조회 실패: {e}")
        db_results = {}

    results = []
    for vector_id, image_path, score in candidates:
        result = db_results.get(str(image_path))
        result = result or build_local_result(vector_id, image_paths)
        if result:
            result["score"] = score
            results.append(result)

    # 카페 단위로 중복 제거 (유사도가 가장 높은 이미지 1장만 대표로 남김)
    best_by_cafe = {}
    for result in results:
        key = result.get("cafe_id") or result.get("cafe_name")
        if key is None:
            continue
        existing = best_by_cafe.get(key)
        if existing is None or result["score"] > existing["score"]:  # FAISS IndexFlatIP는 score 클수록 유사
            best_by_cafe[key] = result

    deduped = sorted(best_by_cafe.values(), key=lambda r: r["score"], reverse=True)
    return deduped



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

def render_result(cafe):
    image_file = resolve_image_path(cafe["image_path"])
    col_img, col_info = st.columns([1, 2])

    with col_img:
        if cafe["image_path"] and image_file.exists():
            img = Image.open(image_file)
            img = crop_to_16_9(img)
            st.image(img, use_container_width=True)  # ← 이 줄 추가
        elif cafe["image_path"]:
            st.warning(f"이미지를 찾을 수 없습니다: {cafe['image_path']}")
        else:
            st.info("등록된 이미지가 없습니다.")

    with col_info:
        # st.write(cafe) 디버깅용
        st.subheader(cafe["cafe_name"])
        st.write(f"주소: {cafe['location']}")
        if cafe["map_url"]:
            st.write(f"[지도 보기]({cafe['map_url']})")
        if cafe["caption"]:
            st.caption(cafe["caption"])

# ✅ 페이지 상태
if "page" not in st.session_state:
    st.session_state.page = "main"

init_session_state()
render_styles()



# ✅ 상단 버튼 (오른쪽)
top_left, top_right = st.columns([9, 1])
with top_right:
    if st.button("📄 모델 설명"):
        st.session_state.page = "about"

# =========================
# ✅ 메인 페이지
# =========================
if st.session_state.page == "main":

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

        # ✅ 안내 토글
        if "show_guide" not in st.session_state:
            st.session_state.show_guide = False

        label = "사용방법 닫기" if st.session_state.show_guide else "사용방법 안내서"

        if st.button(label):
            st.session_state.show_guide = not st.session_state.show_guide

        if st.session_state.show_guide:
            st.write(
                "1. 초록색 입력창에 원하는 카페의 분위기나 조건을 작성한다.\n"
                "2. 작성을 완료하면 '검색하기'버튼을 누른다.\n"
                "3. 결과가 나오면 원하는 카페 칸에 '지도보기'를 누르면 네이버 길찾기로 이동한다."
            )

        # 검색 버튼
        st.markdown('<div class="button-container">', unsafe_allow_html=True)
        search_clicked = st.button("검색하기", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

        if search_clicked:
            st.session_state.search_clicked = True
            query = vibe.strip()
            query = expand_query(query)  # 이 줄 추가해야 llm이 검색어 확장 가능

            if not query:
                st.warning("검색어를 입력해주세요.")
                st.session_state.search_results = []
            else:
                try:
                    model, processor = load_model()
                    index = load_index()
                    image_paths = load_image_paths()

                    text_results = search_by_text(
                        model, processor, index, image_paths, query
                    )

                    st.session_state.search_results = text_results

                except Exception as exc:
                    st.error(f"검색 중 오류가 발생했습니다: {exc}")
                    st.session_state.search_results = []

        # 결과 출력
        if st.session_state.search_results:
            num_results = st.slider(
                "보고 싶은 카페 개수",
                1,
                len(st.session_state.search_results),
                min(20, len(st.session_state.search_results))
            )

            st.write("### 추천 카페")

            for cafe in st.session_state.search_results[:num_results]:
                render_result(cafe)

        elif st.session_state.search_clicked:
            st.info("조건에 맞는 카페를 찾지 못했습니다.")

        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# ✅ 모델 설명 페이지
# =========================
elif st.session_state.page == "about":

    st.title("📄 모델 설명")

    col_left, col_right = st.columns([1, 1])

    # =========================
    # ✅ 왼쪽: 기술 설명
    # =========================
    with col_left:
        st.write("""
        ### 🔍 사용한 기술

        #### 1. CLIP (Contrastive Language-Image Pretraining)
        - 이미지와 텍스트를 같은 벡터 공간으로 변환
        - "분위기" 같은 추상적인 표현 검색 가능

        #### 2. LoRA (Low-Rank Adaptation)
        - CLIP을 카페 데이터에 맞게 미세조정
        - 적은 데이터로 효율적인 학습

        #### 3. FAISS
        - 벡터 유사도 검색 엔진
        - 빠른 이미지 검색 가능

        #### 4. PostgreSQL
        - 카페 정보 및 이미지 메타데이터 저장
        
        #### 5. NeonDB
        - PostgreSQL을 편하게 사용하기 위해 사용

        ### 🔄 전체 흐름
        사용자 입력 → 텍스트 임베딩 → FAISS 검색 → DB 매칭 → 결과 출력
        """)

    # =========================
    # ✅ 오른쪽: 모델 선택 + 표
    # =========================
    with col_right:
        st.write("""
        ### 👊 모델 선택

        CLIP 모델에 LoRA를 적용하여 총 5가지 설정으로 파인튜닝을 수행하고, 
        각 모델의 성능을 image recall과 caption 기반 지표를 중심으로 비교하였다. 

        그 결과, 두 번째 실험(lora_weights2)이 가장 우수한 성능을 보여 
        최종 모델로 선정하였다.
        """)

        import pandas as pd

        data = {
            "Model": ["weights1", "weights2✔️", "weights3", "weights4", "weights5"],
            "Recall@1": [0.7125, 0.7688, 0.7250, 0.7312, 0.7250],
            "Recall@5": [0.9062, 0.9062, 0.9000, 0.9000, 0.9000],
            "Recall@10": [0.9375, 0.9563, 0.9563, 0.9563, 0.9563],
            "Caption Loss": [0.8717, 0.7981, 0.8136, 0.8090, 0.8179],
            "Image Acc": [0.7063, 0.6937, 0.6813, 0.6937, 0.6875],
            "Text Acc": [0.7125, 0.6937, 0.6500, 0.6562, 0.6687],
        }

        df = pd.DataFrame(data)

        st.subheader("📊 LoRA 모델 성능 비교")
        st.dataframe(
            df.style
              .highlight_max(subset=["Recall@1","Recall@5","Recall@10","Image Acc","Text Acc"],axis=0)
              .highlight_min(subset=["Caption Loss"], axis=0)
        )
        st.success("✔ 최종 모델: lora_weights2 (Recall@1 + Caption 성능 최고)")
        st.caption("※ Recall@1이 가장 중요한 기준이며, Caption 성능을 함께 고려하여 모델을 선정함")

    if st.button("← 돌아가기"):
        st.session_state.page = "main"