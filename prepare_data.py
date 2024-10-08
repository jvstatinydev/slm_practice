import pandas as pd
import urllib.request
from config import DATA_CSV, END_TOKEN, START_TOKEN, TRAIN_TXT

# 데이터 다운로드 및 로드
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
    filename=DATA_CSV,
)
data = pd.read_csv(DATA_CSV)

# 텍스트 포맷 변환 (레이블 포함)
texts = []
for idx, row in data.iterrows():
    q = row['Q']
    a = row['A']
    label = row['label']
    text = f"{START_TOKEN}레이블: {label}\n질문: {q}\n답변: {a}{END_TOKEN}"
    texts.append(text)

# 텍스트 파일로 저장
with open(TRAIN_TXT, 'w', encoding='utf-8') as f:
    for t in texts:
        f.write(t + '\n')
