import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as px_go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import _tree

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="분류 모델 - 뇌졸중 예측 실습실",
    page_icon="🤖",
    layout="wide"
)

# 데이터 로드 (캐싱)
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

st.title("🤖 분류 모델 (Classification Models)")
st.markdown("특성(속성)을 선택하여 로지스틱 회귀 및 의사결정트리 모델을 학습시키고 평가 및 시각화를 진행합니다.")
st.divider()

# 우리말 이름 매핑 사전
FEATURE_MAP = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}
REV_FEATURE_MAP = {v: k for k, v in FEATURE_MAP.items()}

# 2. 속성 선택 영역
st.subheader("1. 입력 속성(Feature) 선택")
default_selected_korean = ['나이', '평균 혈당', '고혈압', '심장병']
selected_korean = st.multiselect(
    "모델 학습 및 예측에 사용할 속성을 2개 이상 선택하세요:",
    options=list(FEATURE_MAP.values()),
    default=default_selected_korean
)

if len(selected_korean) < 2:
    st.warning("⚠️ 속성을 2개 이상 고르셔야 모델 학습 및 결정 경계 시각화가 가능합니다.")
    st.stop()

selected_features = [REV_FEATURE_MAP[k] for k in selected_korean]

# 3. 데이터 정렬 및 Train/Test 분할
# "번호 순으로 정렬한 뒤 열 명씩 묶어, 각 묶음의 앞 세 명을 테스트용으로 고정"
df_sorted = df.copy()
if 'id' in df_sorted.columns:
    df_sorted = df_sorted.sort_values(by='id').reset_index(drop=True)
else:
    df_sorted = df_sorted.reset_index(drop=True)

test_mask = (df_sorted.index % 10) < 3
train_df = df_sorted[~test_mask].copy().reset_index(drop=True)
test_df = df_sorted[test_mask].copy().reset_index(drop=True)

# 체질량지수(bmi) 선택 여부에 따른 결측치 훈련용 중앙값 대치
if 'bmi' in selected_features:
    bmi_median_train = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(bmi_median_train)
    test_df['bmi'] = test_df['bmi'].fillna(bmi_median_train)

X_train = train_df[selected_features].copy()
y_train = train_df['stroke'].values
X_test = test_df[selected_features].copy()
y_test = test_df['stroke'].values

# 4. 모델 구축 및 학습
# (1) 가장 많은 클래스로만 답하는 베이스라인 모델
majority_class = train_df['stroke'].mode()[0]
train_acc_base = (y_train == majority_class).mean()
test_acc_base = (y_test == majority_class).mean()

# (2) 로지스틱 회귀 (확률로 답하는 모델) - 표준화 스케일러 적용
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

train_acc_lr = log_reg.score(X_train_scaled, y_train)
test_acc_lr = log_reg.score(X_test_scaled, y_test)

# (3) 의사결정트리 (질문으로 답하는 모델)
# max_depth=3 (질문 3번까지), min_samples_leaf=5 (마지막 마디 5명 미만 더 나누지 않음), random_state 고정
dt_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_tree.fit(X_train, y_train)

train_acc_dt = dt_tree.score(X_train, y_train)
test_acc_dt = dt_tree.score(X_test, y_test)

# 5. 모델 정확도 카드 출력
st.subheader("2. 모델 성능 평가 (정확도)")
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.metric(
        label="입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
        value=f"{test_acc_base * 100:.2f}%"
    )
    st.caption(f"훈련 정확도: **{train_acc_base*100:.2f}%** | 테스트 정확도: **{test_acc_base*100:.2f}%**")

with col_m2:
    st.metric(
        label="로지스틱 회귀(확률로 답하는 모델)",
        value=f"{test_acc_lr * 100:.2f}%"
    )
    st.caption(f"훈련 정확도: **{train_acc_lr*100:.2f}%** | 테스트 정확도: **{test_acc_lr*100:.2f}%**")

with col_m3:
    st.metric(
        label="의사결정트리(질문으로 답하는 모델)",
        value=f"{test_acc_dt * 100:.2f}%"
    )
    st.caption(f"훈련 정확도: **{train_acc_dt*100:.2f}%** | 테스트 정확도: **{test_acc_dt*100:.2f}%**")

st.divider()

# 6. 결정 경계 산점도 시각화
st.subheader("3. 2차원 결정 경계 산점도")

c_x, c_y = st.columns(2)
with c_x:
    x_axis_kor = st.selectbox("가로축(X축) 속성 선택:", selected_korean, index=0)
with c_y:
    y_axis_kor = st.selectbox("세로축(Y축) 속성 선택:", selected_korean, index=1 if len(selected_korean) > 1 else 0)

x_feat = REV_FEATURE_MAP[x_axis_kor]
y_feat = REV_FEATURE_MAP[y_axis_kor]

if x_feat == y_feat:
    st.warning("⚠️ 가로축과 세로축 속성을 서로 다르게 선택해 주세요.")
else:
    # 비축 속성 고정치 계산 (테스트 데이터의 중앙값)
    other_features = [f for f in selected_features if f not in [x_feat, y_feat]]
    fixed_values = {}
    fixed_info_strings = []
    
    for f in other_features:
        med_val = test_df[f].median()
        fixed_values[f] = med_val
        fixed_info_strings.append(f"**{FEATURE_MAP[f]}**: {med_val:.2f}")

    if fixed_info_strings:
        st.info("📌 두 축이 아닌 나머지 선택 속성은 **테스트 데이터의 중앙값**으로 고정하여 계산하였습니다: " + ", ".join(fixed_info_strings))
    else:
        st.info("📌 선택된 속성이 2개이므로 두 축으로만 전체 모델을 계산합니다.")

    # 그리드 생성
    x_min, x_max = test_df[x_feat].min() - 1, test_df[x_feat].max() + 1
    y_min, y_max = test_df[y_feat].min() - 1, test_df[y_feat].max() + 1

    # 이진 변수인 경우 범위 조정
    if x_feat in ['hypertension', 'heart_disease']:
        x_min, x_max = -0.2, 1.2
    if y_feat in ['hypertension', 'heart_disease']:
        y_min, y_max = -0.2, 1.2

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid_df = pd.DataFrame({
        x_feat: xx.ravel(),
        y_feat: yy.ravel()
    })
    for f, val in fixed_values.items():
        grid_df[f] = val

    # 열 순서 일치
    grid_df = grid_df[selected_features]

    # 의사결정트리 예측 (그리드 옅은 색 배경)
    Z_dt = dt_tree.predict(grid_df).reshape(xx.shape)

    # Plotly 객체 구축
    fig = px_go.Figure()

    # (1) 의사결정트리 영억 면 표현 (옅은 색 등고선)
    fig.add_trace(px_go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z_dt,
        showscale=False,
        colorscale=[[0, 'rgba(173, 216, 230, 0.25)'], [1, 'rgba(255, 182, 193, 0.35)']],
        opacity=0.6,
        hoverinfo='skip',
        name='의사결정트리 영역'
    ))

    # (2) 로지스틱 회귀 0.5 결정 경계 (P(Stroke)=0.5)
    grid_scaled = scaler.transform(grid_df)
    Z_lr_prob = log_reg.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    # 경계선이 영역 내 존재하는지 확인
    is_boundary_in_range = np.any((Z_lr_prob >= 0.48) & (Z_lr_prob <= 0.52))

    fig.add_trace(px_go.Contour(
        x=np.linspace(x_min, x_max, 200),
        y=np.linspace(y_min, y_max, 200),
        z=Z_lr_prob,
        contours_start=0.5,
        contours_end=0.5,
        contours_coloring='lines',
        line_width=3,
        line_color='black',
        showscale=False,
        name='로지스틱 회귀 경계선 (0.5)'
    ))

    # (3) 테스트 데이터 산점도 (실제 뇌졸중 여부 색상)
    test_df_display = test_df.copy()
    test_df_display['stroke_label'] = test_df_display['stroke'].map({0: '정상 (0)', 1: '뇌졸중 (1)'})

    for stroke_val, color, name in [(0, 'blue', '정상 (0)'), (1, 'red', '뇌졸중 (1)')]:
        sub_df = test_df_display[test_df_display['stroke'] == stroke_val]
        fig.add_trace(px_go.Scatter(
            x=sub_df[x_feat],
            y=sub_df[y_feat],
            mode='markers',
            marker=dict(color=color, size=6, opacity=0.8),
            name=f"테스트: {name}"
        ))

    fig.update_layout(
        title=f"테스트 데이터 산점도 및 결정 경계 ({x_axis_kor} vs {y_axis_kor})",
        xaxis_title=x_axis_kor,
        yaxis_title=y_axis_kor,
        legend_title="구분",
        margin=dict(l=40, r=40, t=50, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    if not is_boundary_in_range:
        st.warning("⚠️ 로지스틱 회귀 모델의 0.5 결정 경계선이 현재 시각화된 그래프 범위 밖에 위치하고 있습니다.")

st.divider()

# 7. 의사결정트리 가지 시각화 (Graphviz DOT)
st.subheader("4. 의사결정트리 가지 구조 시각화")

tree_ = dt_tree.tree_
feature_names = [FEATURE_MAP[f] for f in selected_features]

dot_lines = [
    'digraph Tree {',
    'node [shape=box, style="filled, rounded", color="black", fontname="NanumGothic, Malgun Gothic, sans-serif"] ;',
    'edge [fontname="NanumGothic, Malgun Gothic, sans-serif"] ;'
]

leaf_nodes = []
asked_features = set()

def recurse(node, depth):
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_names[tree_.feature[node]]
        asked_features.add(name)
        threshold = tree_.threshold[node]
        
        n_node_samples = tree_.n_node_samples[node]
        val = tree_.value[node][0]
        stroke_cnt = int(val[1]) if len(val) > 1 else 0
        ratio = (stroke_cnt / n_node_samples * 100) if n_node_samples > 0 else 0
        
        label_text = f"{name} <= {threshold:.2f}\\n사람 수: {n_node_samples:,}명\\n뇌졸중: {stroke_cnt:,}명 ({ratio:.1f}%)"
        dot_lines.append(f'{node} [label="{label_text}", fillcolor="#ffffff"] ;')
        
        left_child = tree_.children_left[node]
        right_child = tree_.children_right[node]
        
        dot_lines.append(f'{node} -> {left_child} [labeldistance=2.5, labelangle=45, headlabel="예"] ;')
        dot_lines.append(f'{node} -> {right_child} [labeldistance=2.5, labelangle=-45, headlabel="아니요"] ;')
        
        recurse(left_child, depth + 1)
        recurse(right_child, depth + 1)
    else:
        n_node_samples = tree_.n_node_samples[node]
        val = tree_.value[node][0]
        stroke_cnt = int(val[1]) if len(val) > 1 else 0
        ratio = (stroke_cnt / n_node_samples * 100) if n_node_samples > 0 else 0
        
        pred_class = np.argmax(val)
        pred_label = "뇌졸중 (1)" if pred_class == 1 else "아님 (0)"
        fillcolor = "#ffcccc" if pred_class == 1 else "#cce5ff"
        
        label_text = f"예측: {pred_label}\\n사람 수: {n_node_samples:,}명\\n뇌졸중: {stroke_cnt:,}명 ({ratio:.1f}%)"
        dot_lines.append(f'{node} [label="{label_text}", fillcolor="{fillcolor}"] ;')
        
        leaf_nodes.append({
            'node': node,
            'pred_class': pred_class
        })

recurse(0, 1)
dot_lines.append('}')
dot_graph = "\n".join(dot_lines)

# Graphviz 트리 시각화
st.graphviz_chart(dot_graph)

# 나무 구조 통계 요약
total_leafs = len(leaf_nodes)
not_stroke_leafs = sum(1 for leaf in leaf_nodes if leaf['pred_class'] == 0)
asked_feat_list_str = ", ".join(list(asked_features)) if asked_features else "없음"

st.markdown(f"- **최종 답을 내는 마디(리프 노드) 수:** 총 **{total_leafs}**개 마디 중 **{not_stroke_leafs}**개 마디가 **'아님(0)'**이라고 답합니다.")
st.markdown(f"- **이 나무가 실제로 물어본 속성:** {asked_feat_list_str}")
