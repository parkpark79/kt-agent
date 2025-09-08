# PDF 기능 제거
import io


import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
import sys
sys.path.append('.')
from kobert_model import predict_aspect_sentiment

# LM Studio OpenAI API 서버 주소
VLLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

def call_mideum_vllm(prompt, model="K-intelligence/Midm-2.0-Base-Instruct"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "당신은 한국어 데이터 기반 보고서 생성 AI입니다. 입력된 분석 결과와 옵션을 반영해, 수치 기반 문장과 다양한 스타일의 보고서를 생성하세요."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.7
    }
    response = requests.post(VLLM_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return "모델 응답 오류"

def analyze_sentiment(df, aspect_col="aspect", sentiment_col="sentiment"):
    # 속성별 긍/부정/중립 비율 계산
    result = {}
    for aspect in df[aspect_col].unique():
        sub = df[df[aspect_col] == aspect]
        total = len(sub)
        pos = (sub[sentiment_col] == 2).sum()
        neu = (sub[sentiment_col] == 1).sum()
        neg = (sub[sentiment_col] == 0).sum()
        result[aspect] = {
            "긍정": round(pos/total*100, 1) if total else 0,
            "중립": round(neu/total*100, 1) if total else 0,
            "부정": round(neg/total*100, 1) if total else 0,
            "총": total
        }
    return result

def plot_sentiment_bar(result, focus=None):
    aspects = list(result.keys())
    pos = [result[a]["긍정"] for a in aspects]
    neu = [result[a]["중립"] for a in aspects]
    neg = [result[a]["부정"] for a in aspects]
    font_path = os.path.join("NanumGothicCoding-2.5", "NanumGothicCoding.ttf")
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'NanumGothicCoding'
    fig, ax = plt.subplots()
    bars1 = ax.bar(aspects, pos, label="긍정", color=["gold" if a==focus else "skyblue" for a in aspects])
    bars2 = ax.bar(aspects, neu, bottom=pos, label="중립", color="gray")
    bars3 = ax.bar(aspects, neg, bottom=[p+n for p,n in zip(pos,neu)], label="부정", color="salmon")
    for i, a in enumerate(aspects):
        ax.text(i, pos[i]/2, f"{pos[i]:.1f}%", ha='center', va='center', fontsize=10)
        ax.text(i, pos[i]+neu[i]/2, f"{neu[i]:.1f}%", ha='center', va='center', fontsize=10)
        ax.text(i, pos[i]+neu[i]+neg[i]/2, f"{neg[i]:.1f}%", ha='center', va='center', fontsize=10)
    ax.set_ylabel("비율(%)")
    ax.set_title("속성별 감성 비율")
    ax.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

def plot_sentiment_pie(result, focus=None):
    st.markdown("#### 선택한 속성의 감성 비율 파이차트")
    if focus and focus in result:
        pie_labels = ["긍정", "중립", "부정"]
        pie_values = [result[focus]["긍정"], result[focus]["중립"], result[focus]["부정"]]
        pie_colors = ["gold", "gray", "salmon"]
        fig2, ax2 = plt.subplots()
        wedges, texts, autotexts = ax2.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', colors=pie_colors, startangle=90, textprops={'fontsize':12})
        ax2.set_title(f"{focus} 감성 비율")
        st.pyplot(fig2)
    else:
        st.info("포커스 속성을 선택하면 파이차트가 표시됩니다.")

# 개선 제안 자동 분류용 예시
EXAMPLE_PROMPT = (
    "아래는 IT기기 리뷰 데이터의 속성별 감성 분석 결과입니다.\n"
    "디자인: 긍정 80.5%, 중립 10.2%, 부정 9.3% (총 100건)\n"
    "성능: 긍정 70.1%, 중립 20.3%, 부정 9.6% (총 100건)\n"
    "배터리: 긍정 60.4%, 중립 25.5%, 부정 14.1% (총 100건)\n"
    "화질: 긍정 90.2%, 중립 5.1%, 부정 4.7% (총 100건)\n\n"
    "요약 레벨: 상세\n"
    "포커스 속성: 전체\n"
    "보고서 버전: 경영진용 요약\n"
    "분석 결과를 바탕으로, 수치 기반 문장과 카테고리별 개선 제안, 마케팅 포인트, 핵심 메시지(요약/마케팅/개발용)를 포함한 맞춤형 종합 보고서를 작성해 주세요.\n"
    "특히 '디자인' 속성에 집중해 주세요.\n"
    "카테고리별 개선 제안은 '설치', '배송', '품질', '가격' 등으로 분류해 주세요.\n"
    "리뷰 샘플:\n"
    "- 이 제품은 디자인이 정말 뛰어납니다. 사용자가 원하는 모든 기능이 잘 갖춰져 있어요.\n"
    "- 성능이 기대 이하입니다. 웹서핑할 때 가끔씩 멈추는 현상이 있습니다.\n"
    "- 배터리 수명이 짧아서 자주 충전해야 하는 불편함이 있습니다.\n"
    "- 화질은 가격대비 만족스럽지만, 어두운 곳에서는 다소 아쉬움이 남습니다."
)

def make_prompt(result, options, df):
    # 수치 기반 문장, 옵션(요약레벨, 포커스, 보고서버전) 반영 프롬프트 생성
    table = "\n".join([f"{a}: 긍정 {v['긍정']}%, 중립 {v['중립']}%, 부정 {v['부정']}% (총 {v['총']}건)" for a,v in result.items()])
    focus = options.get("focus", "전체")
    level = options.get("level", "상세")
    version = options.get("version", "경영진용 요약")
    prompt = f"아래는 IT기기 리뷰 데이터의 속성별 감성 분석 결과입니다.\n{table}\n\n"
    prompt += f"요약 레벨: {level}\n포커스 속성: {focus}\n보고서 버전: {version}\n"
    prompt += "분석 결과를 바탕으로, 수치 기반 문장과 카테고리별 개선 제안, 마케팅 포인트, 핵심 메시지(요약/마케팅/개발용)를 포함한 맞춤형 종합 보고서를 작성해 주세요.\n"
    if focus != "전체" and focus in result:
        prompt += f"특히 '{focus}' 속성에 집중해 주세요.\n"
    # 개선 제안 자동 분류용 예시
    prompt += "카테고리별 개선 제안은 '설치', '배송', '품질', '가격' 등으로 분류해 주세요.\n"
    # 데이터 일부 샘플 포함(신뢰 강화)
    sample_reviews = df.head(5)["상품평"].tolist()
    prompt += "\n리뷰 샘플:\n" + "\n".join(sample_reviews)
    return prompt

# LM Studio OpenAI API 서버 주소
VLLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"

def call_mideum_vllm(text, model="K-intelligence/Midm-2.0-Base-Instruct"):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "당신은 한국어 보고서 요약 AI입니다. 입력된 리뷰를 분석해 요약, 감성, 핵심포인트를 생성하세요."},
            {"role": "user", "content": text}
        ],
        "max_tokens": 512,
        "temperature": 0.7
    }
    response = requests.post(VLLM_API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    else:
        return "모델 응답 오류"

def generate_report(df, model):
    # 모든 리뷰를 하나의 텍스트로 합침
    all_reviews = "\n".join(df['상품평'].astype(str).tolist())
    st.info("전체 리뷰를 종합하여 보고서를 생성합니다. 시간이 다소 소요될 수 있습니다.")
    summary = call_mideum_vllm(
        f"아래는 여러 IT기기 리뷰입니다. 전체 리뷰를 종합적으로 분석하여, 주요 특징, 긍정/부정 경향, 개선점, 마케팅 포인트 등을 포함한 종합 보고서를 작성해 주세요.\n\n{all_reviews}",
        model
    )
    return summary

def generate_pdf(report_text):
    pass  # PDF 기능 제거

st.title("믿음:음 2.0(vLLM) 기반 자동 보고서 생성 Agent")

uploaded_file = st.file_uploader("CSV 파일 업로드", type="csv")
model_type = st.selectbox("모델 타입 선택", [
    "K-intelligence/Midm-2.0-Base-Instruct",
    "K-intelligence/Midm-2.0-Mini-Instruct"
])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 데이터", df.head())

    # KoBERT로 aspect, sentiment 컬럼 자동 생성 (없을 때만)
    if "aspect" not in df.columns or "sentiment" not in df.columns:
        st.info("aspect, sentiment 컬럼이 없어 KoBERT 모델로 감성 분석을 진행합니다. 리뷰 수에 따라 시간이 오래 걸릴 수 있습니다.")
        # '상품평' 컬럼을 'review'로 변환
        if '상품평' in df.columns:
            df['review'] = df['상품평']
        # 'aspect' 컬럼 없으면 기본값 '품질'로 생성
        if 'aspect' not in df.columns:
            df['aspect'] = ['품질'] * len(df)
        # KoBERT 감성 분석
        df = predict_aspect_sentiment(df)
        st.success("KoBERT 기반 aspect, sentiment 컬럼 생성 완료!")

    # 사용자 옵션: 요약레벨, 포커스 속성, 보고서 버전 선택
    st.sidebar.header("보고서 옵션")
    level = st.sidebar.selectbox("요약 레벨", ["짧게", "중간", "상세"], index=2)
    if "aspect" in df.columns:
        focus_options = ["전체"] + list(df["aspect"].unique())
    else:
        focus_options = ["전체"]
    focus = st.sidebar.selectbox("포커스 속성", focus_options)
    version = st.sidebar.selectbox("보고서 버전", ["경영진용 요약", "마케팅용", "제품개발용"])
    graph_type = st.sidebar.selectbox("그래프 타입", ["막대그래프", "파이차트"])

    # 속성별 감성 분석 및 시각화
    if "aspect" in df.columns and "sentiment" in df.columns:
        result = analyze_sentiment(df)
        st.subheader("속성별 감성 비율 표")
        st.dataframe(pd.DataFrame(result).T)
        st.subheader("속성별 감성 시각화")
        if graph_type == "막대그래프":
            plot_sentiment_bar(result, focus=focus if focus!="전체" else None)
        elif graph_type == "파이차트":
            plot_sentiment_pie(result, focus=focus if focus!="전체" else None)
    else:
        st.warning("데이터에 'aspect', 'sentiment' 컬럼이 필요합니다.")
        result = {}

    # 보고서 생성
    if st.button("자동 보고서 생성"):
        options = {"level": level, "focus": focus, "version": version}
        prompt = make_prompt(result, options, df)
        with st.spinner("맞춤형 종합 보고서를 생성 중입니다. 잠시만 기다려주세요..."):
            report = call_mideum_vllm(prompt, model_type)
        st.subheader("자동 종합 보고서")
        st.write(report)
        if report:
            st.download_button("텍스트로 다운로드", report, "report.txt")

