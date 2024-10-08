# prepare_data.py - 1. 데이터 준비 단계
# 이 파일은 LLMOps 파이프라인에서 데이터를 수집하고 정제하여 학습에 적합한 형태로 준비하는 역할을 수행합니다.

import pandas as pd
import urllib.request
from config import DATA_CSV, END_TOKEN, START_TOKEN, TRAIN_TXT

# 데이터 다운로드 및 로드
# 외부 소스에서 데이터 파일을 다운로드하고 로컬에 저장합니다.
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename=DATA_CSV,
)
data = pd.read_csv(DATA_CSV)  # 저장된 CSV 파일을 데이터프레임으로 로드합니다.

# 텍스트 포맷 변환 (레이블 포함)
# 각 데이터 행을 포맷팅하여 학습에 사용할 수 있는 텍스트 형식으로 변환합니다.
texts = []
for idx, row in data.iterrows():
    q = row['Q']  # 질문 텍스트
    a = row['A']  # 답변 텍스트
    label = row['label']  # 레이블 정보
    text = f"{START_TOKEN}레이블: {label}\n질문: {q}\n답변: {a}{END_TOKEN}"
    texts.append(text)

# 텍스트 파일로 저장
# 변환된 텍스트 데이터를 파일로 저장하여 이후 학습 단계에서 사용합니다.
with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
    for t in texts:
        f.write(t + '\n')