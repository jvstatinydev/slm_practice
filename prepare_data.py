# prepare_data.py - 1. 데이터 준비 단계
# 이 파일은 LLMOps 파이프라인에서 데이터를 수집하고 정제하여 학습 및 검증에 적합한 형태로 준비하는 역할을 수행합니다.

import pandas as pd
import urllib.request
import json
import os
from config import DATA_CSV, END_TOKEN, START_TOKEN, TRAIN_TXT, DATA_DIR, TRAIN_JSON_DIR, VALID_JSON_DIR, VALID_TXT  # VALID 관련 변수 추가

def download_data(url, filename):
    """외부 소스에서 데이터 파일을 다운로드하고 로컬에 저장합니다."""
    urllib.request.urlretrieve(url, filename)

def load_data(file_path):
    """저장된 CSV 파일을 데이터프레임으로 로드합니다."""
    return pd.read_csv(file_path)

def format_texts(data):
    """각 데이터 행을 포맷팅하여 학습에 사용할 수 있는 텍스트 형식으로 변환합니다."""
    texts = []
    for idx, row in data.iterrows():
        q = row['Q']  # 질문 텍스트
        a = row['A']  # 답변 텍스트
        label = row['label']  # 레이블 정보
        text = f"{START_TOKEN}레이블: {label}\n질문: {q}\n답변: {a}{END_TOKEN}"
        texts.append(text)
    return texts

def load_and_format_json_data(json_data_dir):
    """JSON 형식의 대화 데이터를 로드하고 학습 텍스트 형식으로 변환합니다."""
    all_texts = []
    json_file_paths = []
    for root, _, files in os.walk(json_data_dir):
        for file in files:
            if file.endswith(".json"):
                json_file_paths.append(os.path.join(root, file))

    for file_path in json_file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                for item in data['info']:
                    if 'annotations' in item and 'lines' in item['annotations']:
                        conversation_text = ""
                        for line in item['annotations']['lines']:
                            speaker_id = line['speaker']['id']
                            text = line['text']
                            conversation_text += f"{speaker_id} : {text}\n"
                        formatted_text = f"{START_TOKEN}대화:\n{conversation_text.strip()}{END_TOKEN}"
                        all_texts.append(formatted_text)
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError in file: {file_path}")
                print(f"Error details: {e}")

    return all_texts

def save_texts_to_file(texts, file_path, append=False):
    """변환된 텍스트 데이터를 파일로 저장합니다."""
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as f:
        for t in texts:
            f.write(t + '\n')

def main():
    # 학습 데이터 처리
    # url = "https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv"
    # download_data(url, DATA_CSV)
    # data = load_data(DATA_CSV)
    # csv_texts = format_texts(data)
    # save_texts_to_file(csv_texts, TRAIN_TXT)
    # print(f"학습 CSV 데이터가 {TRAIN_TXT}에 저장되었습니다.")

    json_train_texts = load_and_format_json_data(TRAIN_JSON_DIR)
    save_texts_to_file(json_train_texts, TRAIN_TXT)
    print(f"학습 JSON 데이터가 {TRAIN_TXT}에 추가되었습니다.")

    # 검증 데이터 처리 (JSON 형식이라고 가정)
    if VALID_JSON_DIR:
        json_valid_texts = load_and_format_json_data(VALID_JSON_DIR)
        save_texts_to_file(json_valid_texts, VALID_TXT)
        print(f"검증 JSON 데이터가 {VALID_TXT}에 저장되었습니다.")
    else:
        print("검증 JSON 데이터 디렉토리가 설정되지 않았습니다.")

if __name__ == "__main__":
    main()