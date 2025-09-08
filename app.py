import streamlit as st
import pandas as pd
from kobert_model import predict_aspect_sentiment
from report_agent import generate_report

st.title("IT기기 리뷰 속성 기반 감성 분석")

# 1. CSV 업로드
uploaded_file = st.file_uploader("리뷰 CSV 파일 업로드", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터", df.head())

    # 2. 모델 추론
    results = predict_aspect_sentiment(df)  # 속성별 감성 예측

    # 3. 보고서 생성 (템플릿 기반)
    report_text = generate_report(results)
    st.subheader("자동 분석 보고서")
    st.text(report_text)

    # 4. 그래프 시각화
    st.subheader("속성별 감성 분포")
    st.bar_chart(results.groupby("aspect")["sentiment"].value_counts().unstack())
