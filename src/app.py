import streamlit as st

# session_state 초기화
if "recommended_tags" not in st.session_state:
    st.session_state.recommended_tags = []

if "selected_tags" not in st.session_state:
    st.session_state.selected_tags = []

# 페이지 기본 설정
st.set_page_config(
    page_title="Vibe Finder",
    layout="wide"
)

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
    vibe = st.text_input("", placeholder="예: 조용한, 감성적인, 공부하기 좋은")
    

    # 분위기 추천 버튼
    if st.button('분위기 추천 키워드로 골라보기'):

        # 나중에는 AI 추천 결과로 변경 가능
        st.session_state.recommended_tags = [
            "조용한",
            "감성",
            "우드톤",
            "디저트맛집",
            "공부하기좋은",
            "인스타감성",
            "일하기 좋은",
            "커피가 맛있는",
            "안락한",
            "베이커리"
        ]


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

        if st.button("검색시작"):
            search_clicked = True
            # FAISS 검색용 쿼리 생성
            query = " ".join(st.session_state.selected_tags)
    if search_clicked:
        st.success(
            f"선택된 분위기 태그로 검색합니다"
        )

            # 여기서 FAISS 검색 연결 예정
            # query_embedding = model.encode(query)
            # D, I = index.search(query_embedding, k=5)

    st.markdown('</div>', unsafe_allow_html=True)