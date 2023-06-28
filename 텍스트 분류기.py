import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import re

# 텍스트 데이터 수집 및 전처리
def preprocess(text):
    # 소문자 변환
    text = text.lower()
    # 구두점 제거
    text = re.sub(r'[^\w\s]','',text)
    # 숫자 제거
    text = re.sub(r'\d+', '', text)
    # 불용어 제거
    stopwords = ['the', 'and', 'a', 'an', 'in', 'to', 'that', 'of', 'for', 'with', 'on', 'at', 'from', 'by']
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# 분류 모델 학습
def train_model():
    # 데이터 수집
    data = [
        {'text': 'I am so happy.', 'category': 'positive'},
        {'text': 'I am so sad.', 'category': 'negative'},
        {'text': 'I am feeling alright.', 'category': 'neutral'}
    ]
    df = pd.DataFrame(data)

    # 데이터 전처리
    df['text'] = df['text'].apply(preprocess)

    # 벡터화
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])

    # 학습
    model = LogisticRegression()
    model.fit(X, df['category'])

    return model, vectorizer

# gradio 인터페이스 정의
def classify_text(text):
    model, vectorizer = train_model()
    preprocessed_text = preprocess(text)
    X = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X)[0]
    if prediction == "neutral":
        return None
    else:
        return prediction

iface = gr.Interface(
    fn=classify_text, 
    inputs=gr.inputs.Textbox(label="노트를 작성해보세요!"), 
    outputs="text",
    title="텍스트 감정 분류 프로그램",
    description="노트를 작성하면 이 프로그램이 긍정적 또는 부정적인 텍스트인지 분류해줍니다."
)

iface.launch()

