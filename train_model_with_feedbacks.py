
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

# RLHF 보상 함수
def set_reward(user_feedback):
    if user_feedback == 1:
        return 1  # 긍정적인 피드백
    elif user_feedback == -1:
        return -1  # 부정적인 피드백
    else:
        return 0  # 중립 피드백

# 피드백 수집 및 보상 반영 함수 - RLHF 학습
def apply_rewards_by_user_feedback(chat_history_data, model_instance, optimizer_instance, loss_function, device_type, max_epochs=10, patience=2):
    if len(chat_history_data) == 0:  # 피드백이 없는 경우
        print("피드백이 없어 RLHF 학습을 진행할 수 없습니다.")
        return []

    # 조기 종료를 위한 변수 초기화
    best_loss = float('inf')
    counter = 0  # 성능 향상이 없는 에포크 수를 세는 카운터
    losses_list = []

    for epoch in range(max_epochs):
        batch_input_ids = []
        batch_output_ids = []
        batch_rewards = []

        for entry in chat_history_data:
            input_ids = entry["input_ids"]
            output_ids = entry["output_ids"]
            user_feedback = entry["user_feedback"]
            reward = set_reward(user_feedback)

            # 보상 스케일링 방식 개선: 보상을 -1에서 1 사이로 클리핑
            reward = max(min(reward, 1), -1)

            if reward != 0:
                batch_input_ids.append(input_ids)
                batch_output_ids.append(output_ids)
                batch_rewards.append(reward)

        if batch_rewards:
            batch_input_ids = torch.stack(batch_input_ids).to(device_type)
            batch_output_ids = torch.stack(batch_output_ids).to(device_type)
            batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device_type)

            # 모델 출력 계산
            logits = model_instance(batch_input_ids).logits
            shift_logits = logits[:, :-1, :].contiguous()
            labels = batch_output_ids[:, 1:].contiguous()
            labels = labels[:, :shift_logits.size(1)]

            # 개별 샘플의 손실 계산
            loss = loss_function(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
            loss = loss.view(batch_input_ids.size(0), -1).mean(dim=1)

            # 보상에 따라 손실 조정
            adjusted_loss = loss * (-batch_rewards)  # 보상이 높을수록 손실이 감소하도록 설정

            # 총 손실 계산
            total_loss = adjusted_loss.mean()

            # 조기 종료 조건 확인
            if total_loss < best_loss:
                best_loss = total_loss
                counter = 0
                # 모델과 옵티마이저의 현재 상태 저장
                torch.save({
                    'model_state_dict': model_instance.state_dict(),
                    'optimizer_state_dict': optimizer_instance.state_dict(),
                }, "best_model.pth")
            else:
                counter += 1
                if counter >= patience:
                    print("조기 종료 조건에 따라 학습을 종료합니다.")
                    # 최적의 모델과 옵티마이저 상태 로드
                    checkpoint = torch.load("best_model.pth", weights_only=False)
                    model_instance.load_state_dict(checkpoint['model_state_dict'])
                    optimizer_instance.load_state_dict(checkpoint['optimizer_state_dict'])
                    break

            # 역전파 및 파라미터 업데이트
            optimizer_instance.zero_grad()
            total_loss.backward()
            optimizer_instance.step()

            # 손실 로그에 추가
            losses_list.append(total_loss.item())
            print(f"에포크 {epoch + 1}: 모델이 {len(batch_rewards)}개의 보상을 반영하여 학습되었습니다. 총 손실: {total_loss.item()}")

        else:
            print("유효한 피드백이 없어 학습을 건너뜁니다.")
            break  # 유효한 피드백이 없으면 학습 종료

    # 업데이트된 모델과 옵티마이저 상태 저장
    torch.save({
        'model_state_dict': model_instance.state_dict(),
        'optimizer_state_dict': optimizer_instance.state_dict(),
    }, "updated_model.pth")

    return losses_list  # 손실 로그 반환


# 챗봇 대화 종료 후 손실 시각화
def plot_losses(losses_data):
    if len(losses_data) == 0:
        print("학습할 손실 데이터가 없어 시각화를 생략합니다.")
        return
    plt.plot(losses_data)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('RLHF Training Loss Log')
    plt.show()



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

    # 기존 채팅 기록 로드 또는 초기화
    if os.path.exists(f"{RESULTS_DIR}/results.json", ):
        with open(f"{RESULTS_DIR}/results.json", "r", encoding="utf-8") as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
    else:
        chat_history = []

    # 사용자 피드백 데이터 로드
    losses = apply_rewards_by_user_feedback(
        chat_history_data=chat_history,
        model_instance=model,
        optimizer_instance=optimizer,
        loss_function=criterion,
        device_type=device
    )
    plot_losses(losses)
    # 학습 후 채팅 기록 초기화
    chat_history.clear()
