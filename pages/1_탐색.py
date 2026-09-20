import streamlit as st
import pandas as pd
import plotly.express as px

# 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실",
    page_icon="📊",
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

st.title("📊 데이터 탐색 (EDA)")
st.markdown("뇌졸중 데이터셋의 주요 변수 분포 및 뇌졸중 발생 간의 연관성을 탐색합니다.")
st.divider()

# 1. 나이와 평균 혈당 분포 (히스토그램)
st.subheader("1. 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age_hist = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(Age) 분포",
        labels={"age": "나이"},
        color_discrete_sequence=["#1f77b4"]
    )
    st.plotly_chart(fig_age_hist, use_container_width=True)

with col2:
    fig_glucose_hist = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 수치(Avg Glucose Level) 분포",
        labels={"avg_glucose_level": "평균 혈당"},
        color_discrete_sequence=["#ff7f0e"]
    )
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# 2. 뇌졸중 여부에 따른 나이 및 평균 혈당 비교 (상자그림 & 평균 표)
st.subheader("2. 뇌졸중 여부별 나이 및 평균 혈당 비교")

df_stroke_label = df.copy()
df_stroke_label['stroke_label'] = df_stroke_label['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

col3, col4 = st.columns(2)

with col3:
    fig_age_box = px.box(
        df_stroke_label,
        x="stroke_label",
        y="age",
        color="stroke_label",
        title="뇌졸중 여부별 나이 분포",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"}
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with col4:
    fig_glucose_box = px.box(
        df_stroke_label,
        x="stroke_label",
        y="avg_glucose_level",
        color="stroke_label",
        title="뇌졸중 여부별 평균 혈당 분포",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"}
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

# 그룹별 평균값 표
st.markdown("##### 📌 뇌졸중 여부에 따른 평균값 비교표")
avg_summary = df.groupby('stroke')[['age', 'avg_glucose_level']].mean().reset_index()
avg_summary['stroke'] = avg_summary['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})
avg_summary.columns = ['뇌졸중 여부', '평균 나이', '평균 혈당 수치']
st.dataframe(avg_summary, use_container_width=True, hide_index=True)

st.divider()

# 3. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 (막대그래프)
st.subheader("3. 기저질환(고혈압·심장병) 유무에 따른 뇌졸중 비율")

col5, col6 = st.columns(2)

# 고혈압 뇌졸중 비율 계산
ht_stroke = df.groupby('hypertension')['stroke'].mean().reset_index()
ht_stroke['stroke_ratio'] = ht_stroke['stroke'] * 100
ht_stroke['hypertension_label'] = ht_stroke['hypertension'].map({0: '고혈압 없음 (0)', 1: '고혈압 있음 (1)'})

with col5:
    fig_ht = px.bar(
        ht_stroke,
        x="hypertension_label",
        y="stroke_ratio",
        text_auto=".2f",
        title="고혈압 유무별 뇌졸중 비율 (%)",
        labels={"hypertension_label": "고혈압 여부", "stroke_ratio": "뇌졸중 비율 (%)"},
        color="hypertension_label"
    )
    fig_ht.update_yaxes(range=[0, 20])
    st.plotly_chart(fig_ht, use_container_width=True)

# 심장병 뇌졸중 비율 계산
hd_stroke = df.groupby('heart_disease')['stroke'].mean().reset_index()
hd_stroke['stroke_ratio'] = hd_stroke['stroke'] * 100
hd_stroke['hd_label'] = hd_stroke['heart_disease'].map({0: '심장병 없음 (0)', 1: '심장병 있음 (1)'})

with col6:
    fig_hd = px.bar(
        hd_stroke,
        x="hd_label",
        y="stroke_ratio",
        text_auto=".2f",
        title="심장병 유무별 뇌졸중 비율 (%)",
        labels={"hd_label": "심장병 여부", "stroke_ratio": "뇌졸중 비율 (%)"},
        color="hd_label"
    )
    fig_hd.update_yaxes(range=[0, 20])
    st.plotly_chart(fig_hd, use_container_width=True)

st.divider()

# 4. BMI 결측치 집단과 전체 뇌졸중 비율 비교
st.subheader("4. 체질량지수(BMI) 결측 집단의 뇌졸중 비율 비교")

bmi_null_df = df[df['bmi'].isnull()]
null_count = len(bmi_null_df)
null_stroke_ratio = (bmi_null_df['stroke'].sum() / null_count * 100) if null_count > 0 else 0
total_stroke_ratio = (df['stroke'].sum() / len(df) * 100)

bmi_comparison = pd.DataFrame([
    {
        "구분": "BMI 결측치 집단",
        "인원 수": f"{null_count:,} 명",
        "뇌졸중 환자 수": f"{bmi_null_df['stroke'].sum():,} 명",
        "뇌졸중 비율 (%)": f"{null_stroke_ratio:.2f}%"
    },
    {
        "구분": "전체 집단",
        "인원 수": f"{len(df):,} 명",
        "뇌졸중 환자 수": f"{df['stroke'].sum():,} 명",
        "뇌졸중 비율 (%)": f"{total_stroke_ratio:.2f}%"
    }
])

st.dataframe(bmi_comparison, use_container_width=True, hide_index=True)

st.divider()

# 5. 흡연 상태별 인원수
st.subheader("5. 흡연 상태(smoking_status)별 인원수")

smoking_counts = df['smoking_status'].value_counts().reset_index()
smoking_counts.columns = ['흡연 상태 (smoking_status)', '인원수 (명)']

st.dataframe(smoking_counts, use_container_width=True, hide_index=True)
