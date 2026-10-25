
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

DATA_URL = "https://raw.githubusercontent.com/dg2025815-tech/ssss/refs/heads/main/human_cognitive_performance.csv"

st.set_page_config(page_title="인지 능력 데이터 분석", layout="wide")

st.title("🧠 Human Cognitive Performance 데이터 분석")
st.write("인지 능력 데이터를 전처리하고 다양한 그래프로 분석하는 웹앱입니다.")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

try:
    original = load_data()
except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.stop()

st.sidebar.header("데이터 전처리")

df = original.copy()

missing_method = st.sidebar.selectbox(
    "결측치 처리 방법",
    ["유지", "결측치 행 삭제", "중앙값/최빈값 대체"]
)

if missing_method == "결측치 행 삭제":
    df = df.dropna()

elif missing_method == "중앙값/최빈값 대체":
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode = df[col].mode()
            if len(mode) > 0:
                df[col] = df[col].fillna(mode[0])

if st.sidebar.checkbox("중복 데이터 제거", value=True):
    df = df.drop_duplicates()

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

if st.sidebar.checkbox("IQR 기준 이상치 제거"):
    selected = st.sidebar.multiselect(
        "이상치 검사 변수",
        numeric_cols,
        default=numeric_cols
    )

    mask = pd.Series(True, index=df.index)

    for col in selected:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        mask &= df[col].between(
            q1 - 1.5 * iqr,
            q3 + 1.5 * iqr
        ) | df[col].isna()

    df = df[mask]

st.subheader("1. 데이터 기본 정보")

c1, c2, c3 = st.columns(3)
c1.metric("전체 데이터 수", len(original))
c2.metric("전처리 후 데이터 수", len(df))
c3.metric("변수 개수", len(df.columns))

st.dataframe(df.head(100), use_container_width=True)

st.subheader("2. 기초 통계 분석")
if numeric_cols:
    st.dataframe(df[numeric_cols].describe(), use_container_width=True)

st.subheader("3. 결측치 분석")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) > 0:
    fig = px.bar(
        x=missing.index,
        y=missing.values,
        title="변수별 결측치 개수",
        labels={"x": "변수", "y": "결측치 수"}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("결측치가 없습니다.")

st.subheader("4. 데이터 시각화")

chart = st.selectbox(
    "그래프 선택",
    [
        "히스토그램",
        "박스 플롯",
        "산점도",
        "상관관계 히트맵",
        "범주형 막대그래프",
        "바이올린 플롯"
    ]
)

if chart == "히스토그램":
    if numeric_cols:
        col = st.selectbox("변수 선택", numeric_cols)
        fig = px.histogram(
            df, x=col, marginal="box",
            title=f"{col} 데이터 분포"
        )
        st.plotly_chart(fig, use_container_width=True)

elif chart == "박스 플롯":
    if numeric_cols:
        col = st.selectbox("변수 선택", numeric_cols)
        fig = px.box(df, y=col, points="outliers")
        st.plotly_chart(fig, use_container_width=True)

elif chart == "산점도":
    if len(numeric_cols) >= 2:
        x = st.selectbox("X축 변수", numeric_cols)
        y = st.selectbox(
            "Y축 변수",
            [c for c in numeric_cols if c != x]
        )
        fig = px.scatter(df, x=x, y=y, trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

elif chart == "상관관계 히트맵":
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="변수 간 상관관계"
        )
        st.plotly_chart(fig, use_container_width=True)

elif chart == "범주형 막대그래프":
    if categorical_cols:
        col = st.selectbox("변수 선택", categorical_cols)
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "빈도"]
        fig = px.bar(counts, x=col, y="빈도")
        st.plotly_chart(fig, use_container_width=True)

elif chart == "바이올린 플롯":
    if numeric_cols:
        col = st.selectbox("변수 선택", numeric_cols)
        fig = px.violin(df, y=col, box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("5. 전처리 데이터 다운로드")

csv = df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    "전처리 데이터 CSV 다운로드",
    data=csv,
    file_name="cognitive_cleaned.csv",
    mime="text/csv"
)
