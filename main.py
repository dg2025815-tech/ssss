import streamlit as st
import pandas as pd
import plotly.express as px

# 1. 페이지 설정 (탭 제목, 아이콘 설정)
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 메인 타이틀
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("본 화면은 뇌졸중 데이터셋의 주요 특성과 구성을 살펴보는 **데이터 소개 페이지**입니다.")
st.divider()

# 2. 주요 지표 카드 (4개)
total_count = len(df)
col_count = len(df.columns)
stroke_count = int(df['stroke'].sum()) if 'stroke' in df.columns else 0
stroke_ratio = (stroke_count / total_count * 100) if total_count > 0 else 0

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(label="전체 사람 수", value=f"{total_count:,} 명")
with m2:
    st.metric(label="열 개수", value=f"{col_count} 개")
with m3:
    st.metric(label="뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,} 명")
with m4:
    st.metric(label="뇌졸중 환자 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# 3. 열 정보 표 (우리말 뜻 직접 수정 가능하도록 data_editor 활용)
st.subheader("📋 데이터 열(Column) 정보")
st.caption("아래 '우리말 뜻' 칸에 교재를 참고하여 내용을 직접 입력해 보세요.")

# 열 정보 데이터프레임 생성
column_info = []
for col in df.columns:
    dtype_str = str(df[col].dtype)
    null_count = int(df[col].isnull().sum())
    column_info.append({
        "열 이름 (Column)": col,
        "우리말 뜻": "",  # 사용자가 직접 채워 넣을 수 있는 빈 칸
        "값의 종류 (Dtype)": dtype_str,
        "빈 값 개수 (Missing)": null_count
    })

info_df = pd.DataFrame(column_info)

# data_editor로 편집 가능한 데이터프레임 표시
edited_info_df = st.data_editor(
    info_df,
    use_container_width=True,
    num_rows="fixed",
    disabled=["열 이름 (Column)", "값의 종류 (Dtype)", "빈 값 개수 (Missing)"],
    hide_index=True
)

st.divider()

# 4. 상위 5개 행 데이터 보기
st.subheader("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 5. 데이터 출처 작성 영역
st.subheader("📌 데이터 출처")
data_source_text = st.text_area(
    "교재에 기재된 데이터 출처 정보를 아래에 입력하세요:",
    placeholder="예: Kaggle Stroke Prediction Dataset / 출처 설명 작성 위치...",
    height=100
)

if data_source_text:
    st.info(f"**작성된 출처 정보:** {data_source_text}")
