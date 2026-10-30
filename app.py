import re
import unicodedata
from pathlib import Path
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="유해 표현 탐지 인사이트 대시보드",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 유해 표현 탐지 & 분석 대시보드")

MODEL_DIR = Path("models")
CHAR_VECTORIZER_PATH = MODEL_DIR / "char_vectorizer.pkl"
CHAR_MODEL_PATH = MODEL_DIR / "char_model.pkl"
WORD_VECTORIZER_PATH = MODEL_DIR / "word_vectorizer.pkl"
WORD_MODEL_PATH = MODEL_DIR / "word_model.pkl"

# 전처리 함수
def normalize_char(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    original = re.sub(r"\s+", "", text)
    normalized = re.sub(r"(?<=[가-힣])[0-9\W_]+(?=[가-힣])", "", original)
    return original + " " + normalized

def normalize_word(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource
def load_models():
    char_vectorizer = joblib.load(CHAR_VECTORIZER_PATH)
    char_model = joblib.load(CHAR_MODEL_PATH)
    word_vectorizer = joblib.load(WORD_VECTORIZER_PATH)
    word_model = joblib.load(WORD_MODEL_PATH)
    return char_vectorizer, char_model, word_vectorizer, word_model

try:
    char_vectorizer, char_model, word_vectorizer, word_model = load_models()
except Exception as e:
    st.error(f"모델 파일을 찾을 수 없습니다. 경로를 확인해주세요: {e}")
    st.stop()

def get_harmful_probability(model, vectorized_text):
    classes = list(model.classes_)
    harmful_index = classes.index(1)
    return float(model.predict_proba(vectorized_text)[0][harmful_index])

def get_linear_svm_coef(model):
    if hasattr(model, "coef_"):
        return model.coef_[0]
    if hasattr(model, "calibrated_classifiers_"):
        coefficients = []
        for calibrated_model in model.calibrated_classifiers_:
            estimator = getattr(calibrated_model, "estimator", None) or getattr(calibrated_model, "base_estimator", None)
            if estimator is not None and hasattr(estimator, "coef_"):
                coefficients.append(estimator.coef_[0])
        if coefficients:
            return sum(coefficients) / len(coefficients)
    raise AttributeError("coef_를 추출할 수 없습니다.")

def build_feature_dataframe(vectorizer, model, input_vector):
    try:
        feature_names = vectorizer.get_feature_names_out()
        coefficients = get_linear_svm_coef(model)
        input_vector = input_vector.tocoo()
        features = []
        for idx, value in zip(input_vector.col, input_vector.data):
            weight = float(coefficients[idx])
            features.append({
                "특징": feature_names[idx],
                "TF-IDF 점수": round(float(value), 4),
                "유해 가중치": round(weight, 4),
                "기여 점수": round(float(value * weight), 4),
            })
        return pd.DataFrame(features)
    except Exception:
        return pd.DataFrame()

# =========================================================
# 탭 구성 (대시보드 형태)
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔍 실시간 탐지", "📊 모델 성능 리포트", "💡 주요 키워드 인사이트"])

# TAB 1: 실시간 분석
with tab1:
    st.subheader("문장 실시간 유해성 분석")
    user_text = st.text_area("분석할 문장을 입력하세요.", placeholder="예: 안녕하세요, 오늘 하루도 잘 부탁드립니다!", height=120)
    
    if st.button("분석 실행", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("문장을 입력해주세요.")
        else:
            char_text = normalize_char(user_text)
            word_text = normalize_word(user_text)
            
            char_vec = char_vectorizer.transform([char_text])
            word_vec = word_vectorizer.transform([word_text])
            
            char_prob = get_harmful_probability(char_model, char_vec)
            word_prob = get_harmful_probability(word_model, word_vec)
            
            is_harmful = (char_prob >= 0.5) or (word_prob >= 0.4)
            
            st.divider()
            if is_harmful:
                st.error("🚨 **주의:** 유해 표현 또는 스팸/피싱 가능성이 높은 문장입니다.")
            else:
                st.success("✅ **정상:** 정상적인 문장으로 판단됩니다.")
                
            col1, col2 = st.columns(2)
            with col1:
                st.metric("비속어, 유해 언어 모델 유해 확률", f"{char_prob * 100:.2f}%")
                st.progress(char_prob)
            with col2:
                st.metric("스팸, 피싱 모델 유해 확률", f"{word_prob * 100:.2f}%")
                st.progress(word_prob)

            # 기여도 차트 시각화
            word_df = build_feature_dataframe(word_vectorizer, word_model, word_vec)
            if not word_df.empty:
                st.markdown("#### 📌 스펨, 광고 모델 판정 기여 요소 Top 5")
                top_features = word_df.sort_values(by="기여 점수", ascending=False).head(5)
                fig = px.bar(top_features, x="기여 점수", y="특징", orientation='h', color="기여 점수",
                             color_continuous_scale="Reds", title="유해 영향력 단어 Top 5")
                st.plotly_chart(fig, use_container_width=True)

# TAB 2: 모델 평가 리포트
with tab2:
    st.subheader("Linear SVM 모델 평가 결과")
    st.info("학습 완료된 비속어, 유해 단어(Char) 및 스팸, 피싱(Word) SVM 모델의 성과 지표입니다.")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("비속어, 유해 단어 모델 정확도", "98.71%")
    m2.metric("스팸, 피싱 모델 정확도", "98.30%")
    m3.metric("비속어, 유해 단어 모델 F1-Score", "0.99")
    m4.metric("스팸, 피싱 모델 F1-Score", "0.98")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**임계값별 문자 모델 성능 변화**")
        char_th_data = pd.DataFrame({
            'Threshold': [0.3, 0.4, 0.5, 0.6, 0.7],
            'Accuracy': [0.9849, 0.9864, 0.9871, 0.9860, 0.9832],
            'Precision': [0.9783, 0.9856, 0.9907, 0.9942, 0.9965],
            'Recall': [0.9918, 0.9871, 0.9834, 0.9776, 0.9699]
        })
        fig_th = px.line(char_th_data, x='Threshold', y=['Precision', 'Recall', 'Accuracy'], markers=True)
        st.plotly_chart(fig_th, use_container_width=True)
        
    with col_b:
        st.markdown("**학습 데이터셋 구성 비율**")
        dataset_df = pd.DataFrame({
            'Dataset': ['Dataset1 (욕설/비하)', 'Dataset2 (우회표현)', 'Dataset3 (피싱/스팸)'],
            'Count': [200000, 11010, 13447]
        })
        fig_pie = px.pie(dataset_df, values='Count', names='Dataset', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# TAB 3: 피처 가중치 분석 (인사이트)
with tab3:
    st.subheader("모델 피처 가중치 (Coefficients) 분석")
    st.write("Linear SVM 학습 결과, 유해 문장 판정에 가장 큰 영향을 준 단어 패턴입니다.")
    
    feature_names = word_vectorizer.get_feature_names_out()
    coefs = get_linear_svm_coef(word_model)
    
    coef_df = pd.DataFrame({'단어/표현': feature_names, '가중치': coefs})
    top_harmful = coef_df.sort_values(by='가중치', ascending=False).head(15)
    
    fig_bar = px.bar(top_harmful, x='가중치', y='단어/표현', orientation='h', 
                     title="유해 판정 기여도 상위 단어 Top 15", color='가중치', color_continuous_scale="Tealgrn", template="plotly_white")
    st.plotly_chart(fig_bar, use_container_width=True)
