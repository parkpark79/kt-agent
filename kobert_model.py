# KoBERT 모델 추론 함수 (kobert-transformers 사용)
import pandas as pd
import torch
from safetensors.torch import load_file
from kobert_transformers import get_tokenizer
from transformers import BertForSequenceClassification

MODEL_PATH = 'result/model.safetensors'
TOKENIZER_PATH = 'result/'

tokenizer = get_tokenizer()
state_dict = load_file(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(TOKENIZER_PATH)
model.load_state_dict(state_dict)

def predict_aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    aspects = df['aspect'] if 'aspect' in df.columns else ['품질'] * len(df)
    reviews = df['review'] if 'review' in df.columns else [''] * len(df)
    sentiments = []
    for aspect, review in zip(aspects, reviews):
        inputs = tokenizer(f"{aspect} {review}", return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
        sentiments.append(pred)
    df['aspect'] = aspects
    df['sentiment'] = sentiments
    return df
