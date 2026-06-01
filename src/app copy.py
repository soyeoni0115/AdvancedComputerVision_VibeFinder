import json
from pathlib import Path

import streamlit as st
from utils import CLIPEmbedding
import faiss
st.set_page_config(
    page_title="Vibe Finder",
    layout="wide"
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMAGE_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata.json"


@st.cache_data
def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
## 가져오는 거 바꾸기
    cafes = []
    for item in data.values():
        cafes.append(
            {
                "image_path": item.get("image_path", ""),
                "img_name": item.get("img_name", ""),
                "tags": item.get("tags", []),
                "location": item.get("location", ""),
            }
        )
    return cafes


def get_all_tags(cafes):
    tags = set()
    for cafe in cafes:
        tags.update(cafe["tags"])
    return sorted(tags)


def find_best_cafes(cafes, selected_tags):
    selected_tag_set = set(selected_tags)
    scored_cafes = []

    for cafe in cafes:
        matched_tags = selected_tag_set.intersection(cafe["tags"])
        if matched_tags:
            scored_cafes.append(
                {
                    **cafe,
                    "match_count": len(matched_tags),
                    "matched_tags": sorted(matched_tags),
                }
            )

    if not scored_cafes:
        return []

    best_score = max(cafe["match_count"] for cafe in scored_cafes)
    return [cafe for cafe in scored_cafes if cafe["match_count"] == best_score]


cafes = load_metadata()
available_tags = get_all_tags(cafes)

# session_state 초기화
if "recommended_tags" not in st.session_state:
    st.session_state.recommended_tags = []

if "selected_tags" not in st.session_state:
    st.session_state.selected_tags = []

if "search_results" not in st.session_state:
    st.session_state.search_results = []

# CSS 스타일 적용
st.markdown("""
<style>
.main {
    background-color: #f5f5f5;
}

.title-container {
    height: 100vh;

    display: flex;
    justify-content: center;
    align-items: flex-start;
    padding-top: 150px;  
}

.title {
    font-size: 6vw;
    font-family: serif;
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
    font-size: 36px;
    text-align: center;
    margin-bottom: 20px;
}

.subdesc {
    font-size: 24px;
    text-align: center;
    margin-bottom: 40px;
}

.stTextInput input {
    background-color: #c7d8d1;
    border-radius: 5px;
    height: 50px;
    font-size: 20px;
}

.stSelectbox div[data-baseweb="select"] {
    border-radius: 20px;
}

.button-container {
    display: flex;
    justify-content: center;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)



# 좌우 레이아웃
left, right = st.columns([1, 2])

#clip내용가져오기
clip_model = CLIPEmbedding()

index = faiss.read_index(
    str(BASE_DIR / "faiss_vibe.index")
)



# 왼쪽 타이틀
with left:

    st.markdown("""
    <div class="title-container">
        <div class="title">
            Vibe<br>Finder
        </div>
    </div>
    """, unsafe_allow_html=True)

# 오른쪽 입력 영역
with right:
    st.markdown('<div class="center-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="desc">원하는 분위기를 입력하세요</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subdesc">분위기에 맞는 카페를 추천해드립니다</div>',
        unsafe_allow_html=True
    )
    #줄글
    vibe = st.text_input(
    "분위기 입력",
    placeholder="예: 조용한, 감성적인, 공부하기 좋은"
    )
    

    # 분위기 추천 버튼
    if st.button('분위기 추천 키워드로 골라보기'):

        # metadata.json에 실제로 존재하는 태그를 보여줍니다.
        st.session_state.recommended_tags = available_tags


    # 추천 태그 출력
    if st.session_state.recommended_tags:

        st.write("### 추천 태그")

        # 한 줄에 5개씩
        cols = st.columns(5)

        for idx, tag in enumerate(st.session_state.recommended_tags):

            with cols[idx % 5]:

                # 현재 선택 여부
                selected = tag in st.session_state.selected_tags

                # 선택 여부에 따라 버튼 색 변경
                button_type = "primary" if selected else "secondary"

                # 태그 버튼
                if st.button(
                    f"#{tag}",
                    key=f"tag_{tag}",
                    type=button_type,
                    use_container_width=True
                ):

                    # 토글 기능
                    if selected:
                        st.session_state.selected_tags.remove(tag)

                    else:
                        st.session_state.selected_tags.append(tag)

                    st.rerun()


    # 현재 선택된 태그 확인
    if st.session_state.selected_tags:

        st.write("### 선택된 태그")

        selected_text = " ".join(
            [f"#{tag}" for tag in st.session_state.selected_tags]
        )

        st.write(selected_text)


    # 검색 버튼
    st.markdown('<div class="button-container">', unsafe_allow_html=True)

    empty, button_col = st.columns([8, 1])
    search_clicked = False
    with button_col:

        #if st.button("검색시작"):
        #    search_clicked = True
            # FAISS 검색용 쿼리 생성
        #    query = " ".join(st.session_state.selected_tags)
        if st.button("검색시작"):
            search_clicked = True

        # 태그 + 사용자 입력 문장 합치기
            query = vibe + " " + " ".join(st.session_state.selected_tags)

            # 빈 입력 방지
            if query.strip():

                # 텍스트 임베딩 생성
                query_embedding = clip_model.get_text_embedding(query)

                # FAISS 검색용 shape 변환
                query_embedding = np.expand_dims(query_embedding, axis=0)

                # Top 5 검색
                D, I = index.search(query_embedding, k=5)

                results = []

                for idx in I[0]:
                    if idx < len(cafes):
                        results.append(cafes[idx])

                st.session_state.search_results = results
    if search_clicked:
        st.session_state.search_results = find_best_cafes(
            cafes,
            st.session_state.selected_tags
        )

        if st.session_state.selected_tags:
            st.success("선택된 분위기 태그로 검색합니다")
        else:
            st.warning("태그를 먼저 선택해주세요.")

            # 여기서 FAISS 검색 연결 예정
            # query_embedding = model.encode(query)
            # D, I = index.search(query_embedding, k=5)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.search_results:
        st.write("### 추천 카페")

        for cafe in st.session_state.search_results:
            image_file = RAW_IMAGE_DIR / cafe["image_path"]

            col_img, col_info = st.columns([1, 2])
            with col_img:
                if image_file.exists():
                    st.image(str(image_file), use_container_width=True)
                else:
                    st.warning(f"이미지를 찾을 수 없습니다: {cafe['image_path']}")

            with col_info:
                st.subheader(cafe["img_name"])
                st.write(f"주소: {cafe['location']}")
                if "matched_tags" in cafe:
                    st.write(
                        "일치한 태그: "
                        + " ".join(f"#{tag}" for tag in cafe["matched_tags"])
                    )

    elif search_clicked and st.session_state.selected_tags:
        st.info("선택한 태그와 일치하는 카페를 찾지 못했습니다.")
