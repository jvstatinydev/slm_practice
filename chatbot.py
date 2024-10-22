# chatbot.py - 3. 챗봇 응용 단계
# 이 파일은 파인튜닝된 모델을 사용하여 사용자가 입력한 질문에 응답하는 챗봇을 구현합니다.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn import CrossEntropyLoss
from config import END_TOKEN, MODEL_DIR, PAD_TOKEN, START_TOKEN, RESULTS_DIR
import os
import matplotlib.pyplot as plt
import chainlit as cl
import chainlit.data as cl_data
from chainlit.user import PersistedUser, User
from chainlit.element import Element, ElementDict
from chainlit.step import StepDict
from chainlit.types import (
    Feedback,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
import json
from typing import Optional, Dict, List

# 전역 변수 선언
chat_history = []
sentiment_analyzer = None
device = None
tokenizer = None
model = None
optimizer = None
criterion = None


# **Custom Data Layer 정의**
class CustomDataLayer(cl_data.BaseDataLayer):

    async def upsert_feedback(self, feedback: Feedback) -> str:
        filename = f"{RESULTS_DIR}/results.json"
        with open(filename, "r", encoding="utf-8") as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
        for i, data in enumerate(existing_data):
            try:
                if data['parent_id'] == feedback.forId:
                    existing_data[i]['feedback'] = {
                        "value": feedback.value,
                        "comment": feedback.comment,
                    }
                    if feedback.value == 0:
                        existing_data[i]['feedback']['score'] = -1
                    else:
                        existing_data[i]['feedback']['score'] = 1
            except KeyError:
                pass
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(existing_data, file, indent=4)
        return feedback.forId

    async def get_user(self, identifier: str) -> Optional["PersistedUser"]:
        pass

    async def create_user(self, user: "User") -> Optional["PersistedUser"]:
        pass

    async def delete_feedback(
        self,
        feedback_id: str,
    ) -> bool:
        pass

    async def create_element(self, element: "Element"):
        pass

    async def get_element(
        self, thread_id: str, element_id: str
    ) -> Optional["ElementDict"]:
        pass

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        pass

    async def create_step(self, step_dict: "StepDict"):
        pass

    async def update_step(self, step_dict: "StepDict"):
        pass

    async def delete_step(self, step_id: str):
        pass

    async def get_thread_author(self, thread_id: str) -> str:
        return ""

    async def delete_thread(self, thread_id: str):
        pass

    async def list_threads(
        self, pagination: "Pagination", filters: "ThreadFilter"
    ) -> "PaginatedResponse[ThreadDict]":
        pass

    async def get_thread(self, thread_id: str) -> "Optional[ThreadDict]":
        pass

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        pass

    async def build_debug_url(self) -> str:
        pass

# **Custom Data Layer 설정**
cl_data._data_layer = CustomDataLayer()


# 자동 레이블 결정 함수
def determine_label(user_input, sentiment_model):
    # 입력된 질문을 감정 분석하여 레이블을 결정합니다
    sentiment = sentiment_model(user_input)[0]

    # 감정 분석 결과 출력
    print(f"감정: {sentiment['label']}, 점수: {sentiment['score']:.2f}")

    # 감정 레이블에서 idx 추출
    id2label = sentiment_model.model.config.id2label
    label_key = None
    for key, value in id2label.items():
        if value == sentiment['label']:
            label_key = key
            break

    # 감정을 부정, 긍정, 중립으로 매핑
    if label_key in [7, 8, 9, 10]:
        return "1"  # 부정
    elif label_key in [0, 1, 2, 3, 4]:
        return "2"  # 긍정
    else:
        return "0"  # 중립 혹은 일상


# 응답 다듬기 함수
def refine_response(input_ids, output_ids, tokenizer_instance):
    response = tokenizer_instance.decode(output_ids[0], skip_special_tokens=False)

    # 입력 텍스트 길이만큼 잘라서 생성된 응답만 추출
    generated_answer = response[len(tokenizer_instance.decode(input_ids[0], skip_special_tokens=False)):]
    generated_answer = generated_answer.strip()

    # END_TOKEN 토큰이 있을 경우 해당 위치에서 자름
    if END_TOKEN in generated_answer:
        generated_answer = generated_answer.split(END_TOKEN)[0].strip()

    # START_TOKEN 토큰이 있을 경우 해당 위치 이전까지만 사용
    if START_TOKEN in generated_answer:
        generated_answer = generated_answer.split(START_TOKEN)[0].strip()

    # 불필요한 특수 토큰 제거
    for token in [PAD_TOKEN, END_TOKEN, START_TOKEN]:
        generated_answer = generated_answer.replace(token, '').strip()

    # 빈 응답 처리
    if len(generated_answer) == 0 or generated_answer.isspace():
        generated_answer = "죄송해요, 잘 이해하지 못했어요."

    return generated_answer

@cl.on_chat_start
async def start():
    global sentiment_analyzer, device, tokenizer, model, optimizer, criterion

    # 감정 분석 파이프라인 생성
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlp04/korean_sentiment_analysis_kcelectra")

    # 모델을 올릴 device 설정. GPU 또는 CPU 사용. 연산 속도를 향상시킵니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 파인튜닝된 모델 로드
    # 저장된 모델과 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # 모델 학습 시 필요한 토큰 설정 추가
    if tokenizer.pad_token is None or tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN, 'eos_token': END_TOKEN})
        model.resize_token_embeddings(len(tokenizer))

    # 모델을 GPU/CPU로 이동
    model.to(device)

    # 옵티마이저 설정 (RLHF 학습에 사용)
    learning_rate = 1e-5
    weight_decay = 1e-5  # 가중치 감쇠를 위한 하이퍼파라미터 설정 (정규화 기법 적용)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 손실 함수 정의
    criterion = CrossEntropyLoss(reduction='none')  # 개별 샘플의 손실을 계산하기 위해 'none'으로 설정

    # 업데이트된 모델과 옵티마이저 상태 로드 또는 저장
    if os.path.exists("updated_model.pth"):
        checkpoint = torch.load("updated_model.pth", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        await cl.Message(content="SYSTEM: 이전 학습 내용을 로드했습니다.").send()
    else:
        # 초기 모델과 옵티마이저 상태 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "updated_model.pth")

    await cl.Message(content="안녕?").send()

@cl.on_message
async def main(user_input):
    global chat_history, model, tokenizer, device, sentiment_analyzer
    user_text = user_input.content

    # 감정 레이블 결정
    label = determine_label(user_text, sentiment_model=sentiment_analyzer)

    # 입력 텍스트 생성
    input_text = f"{START_TOKEN}레이블: {label}\n질문: {user_input.content}\n답변:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    output_ids = model.generate(
        input_ids,

        # 모델이 생성할 새로운 토큰의 최대 개수
        max_new_tokens=50,

        # 최소 응답 길이 설정
        min_length=20,

        # 확률 분포를 기반으로 샘플링. False일 경우 가장 높은 확률의 토큰을 고정적으로 선택. True일 경우 다양하고 창의적인 응답을 생성.
        # True인 경우 temperature, top_p 값 설정 가능.
        do_sample=True,

        # 생성된 텍스트의 다양성을 조절하는 파라미터. 낮은 값은 더 결정적인 응답을 생성하며, 높은 값은 더 창의적이거나 예측 불가능한 응답
        temperature=0.8,

        # 확률 축적 방법을 사용해 생성할 토큰을 결정하는 top-p 샘플링. 다양성을 유지하면서 비정상적인 토큰이 선택되는 것을 방지.
        # 낮은 top_p 값은 모델이 더 결정적이고 예측 가능한 응답을 생성하게 합니다.
        # 높은 top_p 값은 모델이 더 창의적이고 다양성 있는 응답을 생성하게 합니다.
        top_p=0.85,

        # 반복되는 단어에 대해 페널티를 부여해 응답의 품질을 높임. 1을 기준으로 높은 값은 반복을 피하고 낮은 값은 반복을 허용.
        repetition_penalty=1.5,

        # 입력의 길이를 맞추기 위해 사용
        pad_token_id=tokenizer.pad_token_id,

        # 모델은 이 종료 토큰을 만나면 생성 프로세스를 중단. 응답의 끝을 자연스럽게 인식하고 멈출 수 있음.
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_answer = refine_response(input_ids, output_ids, tokenizer_instance=tokenizer)

    # 사용자에게 메시지 전송
    message = await cl.Message(content=generated_answer).send()

    # 사용자로부터 피드백 수집
    chat_info = {
        "message_id": message.id,
        "parent_id": message.parent_id,
        "thread_id": user_input.thread_id,
        "user_text": user_text, "generated_answer":generated_answer,
        "input_ids": input_ids.cpu().tolist(), "output_ids": output_ids.cpu().tolist()
    }

    # 기존 채팅 기록 로드 또는 초기화
    if os.path.exists(f"{RESULTS_DIR}/results.json",):
        with open(f"{RESULTS_DIR}/results.json", "r", encoding="utf-8") as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
    else:
        chat_history = []

    # 새로운 대화 정보를 채팅 기록에 추가
    chat_history.append(chat_info)

    # 채팅 기록을 JSON 파일로 저장
    with open(f"{RESULTS_DIR}/results.json", "w", encoding="utf-8") as file:
        json.dump(chat_history, file, ensure_ascii=False, indent=4)




if __name__ == "__main__":

    # 감정 분석 파이프라인 생성
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlp04/korean_sentiment_analysis_kcelectra")

    # 모델을 올릴 device 설정. GPU 또는 CPU 사용. 연산 속도를 향상시킵니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 파인튜닝된 모델 로드
    # 저장된 모델과 토크나이저를 로드합니다.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

    # 모델 학습 시 필요한 토큰 설정 추가
    if tokenizer.pad_token is None or tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN, 'eos_token': END_TOKEN})
        model.resize_token_embeddings(len(tokenizer))

    # 모델을 GPU/CPU로 이동
    model.to(device)

    # 옵티마이저 설정 (RLHF 학습에 사용)
    learning_rate = 1e-5
    weight_decay = 1e-5  # 가중치 감쇠를 위한 하이퍼파라미터 설정 (정규화 기법 적용)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 손실 함수 정의
    criterion = CrossEntropyLoss(reduction='none')  # 개별 샘플의 손실을 계산하기 위해 'none'으로 설정

    # 업데이트된 모델과 옵티마이저 상태 로드 또는 저장
    if os.path.exists("updated_model.pth"):
        checkpoint = torch.load("updated_model.pth", weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # 초기 모델과 옵티마이저 상태 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, "updated_model.pth")