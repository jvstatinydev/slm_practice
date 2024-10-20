# chatbot.py - 3. 챗봇 응용 단계
# 이 파일은 파인튜닝된 모델을 사용하여 사용자가 입력한 질문에 응답하는 챗봇을 구현합니다.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch.nn import CrossEntropyLoss
from config import END_TOKEN, MODEL_DIR, PAD_TOKEN, START_TOKEN
import os
import matplotlib.pyplot as plt

# 자동 레이블 결정 함수
def determine_label(_user_input):
    # 입력된 질문을 감정 분석하여 레이블을 결정합니다
    sentiment = sentiment_analyzer(_user_input)[0]

    # 감정 분석 결과 출력
    print(f"감정: {sentiment['label']}, 점수: {sentiment['score']:.2f}")

    # 감정 레이블에서 idx 추출
    id2label = sentiment_analyzer.model.config.id2label
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
def refine_response(_input_ids, _output_ids, _tokenizer):
    response = _tokenizer.decode(_output_ids[0], skip_special_tokens=False)

    # 입력 텍스트 길이만큼 잘라서 생성된 응답만 추출
    _generated_answer = response[len(_tokenizer.decode(_input_ids[0], skip_special_tokens=False)):]
    _generated_answer = _generated_answer.strip()

    # END_TOKEN 토큰이 있을 경우 해당 위치에서 자름
    if END_TOKEN in _generated_answer:
        _generated_answer = _generated_answer.split(END_TOKEN)[0].strip()

    # START_TOKEN 토큰이 있을 경우 해당 위치 이전까지만 사용
    if START_TOKEN in _generated_answer:
        _generated_answer = _generated_answer.split(START_TOKEN)[0].strip()

    # 불필요한 특수 토큰 제거
    for token in [PAD_TOKEN, END_TOKEN, START_TOKEN]:
        _generated_answer = _generated_answer.replace(token, '').strip()

    # 빈 응답 처리
    if len(_generated_answer) == 0 or _generated_answer.isspace():
        _generated_answer = "죄송해요, 잘 이해하지 못했어요."

    return _generated_answer

# RLHF 보상 함수
def reward_function(user_feedback):
    if user_feedback == "1":
        return 1  # 긍정적인 피드백
    elif user_feedback == "2":
        return -1  # 부정적인 피드백
    else:
        return 0  # 중립 피드백

# 피드백 수집 및 보상 반영 함수
def collect_feedback_and_apply_reward(_input_ids, _output_ids, _model, _optimizer, _criterion):
    user_feedback = input("사용자의 피드백 (1-좋음, 2-별로, 미입력-의견없음): ")
    reward = reward_function(user_feedback)

    # RLHF 학습을 위한 파라미터 업데이트
    if reward != 0:
        # 보상에 따라 손실(loss) 계산 및 학습
        logits = _model(_input_ids).logits  # 입력에 대한 모델의 출력을 가져옴
        shift_logits = logits[:, :-1, :].contiguous()  # 출력의 마지막 토큰을 제외하고 입력과 맞춤
        labels = _output_ids[:, 1:].contiguous()  # 모델 출력에서 첫 번째 토큰을 제외한 나머지를 레이블로 사용
        labels = labels[:, :shift_logits.size(1)]  # 출력의 길이와 레이블 길이를 맞춤

        loss = _criterion(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
        if reward > 0:
            loss = reward * loss  # 긍정적인 보상일 때 손실을 증가시켜 학습을 강화
        elif reward < 0:
            loss = loss / abs(reward)  # 부정적인 보상일 때 손실을 약화시켜 학습을 조절

        # 손실을 역전파하고 옵티마이저로 파라미터 업데이트
        loss.backward()
        _optimizer.step()
        _optimizer.zero_grad()

        # 손실 로그에 추가
        losses.append(loss.item())

        print(f"모델이 {reward}의 보상을 반영하여 학습되었습니다.")
        # 업데이트된 모델 저장
        torch.save(_model.state_dict(), "updated_model.pth")

# 챗봇 대화 종료 후 손실 시각화
def plot_losses():
    plt.plot(losses)
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('RLHF Training Loss Log')
    plt.show()

# 손실 로그 기록용 배열
losses = []

# 감정 분석 파이프라인 생성
sentiment_analyzer = pipeline("sentiment-analysis", model="nlp04/korean_sentiment_analysis_kcelectra")

# 모델을 GPU 또는 CPU에 올려서 연산 속도를 향상시킵니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파인튜닝된 모델 로드
# 저장된 모델과 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

# 모델 학습 시 필요한 토큰 설정 추가
if tokenizer.pad_token is None or tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN, 'eos_token': END_TOKEN})
    model.resize_token_embeddings(len(tokenizer))

# 업데이트된 모델 파라미터를 저장하여 일관성 유지
if not os.path.exists("updated_model.pth"):
    torch.save(model.state_dict(), "updated_model.pth")
# 이전에 저장된 모델 파라미터가 있는 경우 로드
else:
    model.load_state_dict(torch.load("updated_model.pth", weights_only=True))
    print("이전 학습 내용을 로드했습니다.")

# 모델을 GPU로 이동
model.to(device)

# 옵티마이저 설정 (RLHF 학습에 사용)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = CrossEntropyLoss()

# 챗봇 대화 루프
# 사용자의 입력을 받아 모델이 응답을 생성합니다.
model.eval()
print("챗봇이 준비되었습니다. 종료하려면 '종료' 또는 'exit'를 입력하세요.")
while True:
    user_input = input("사용자: ")
    if user_input.lower() in ["종료", "exit"]:
        print("챗봇을 종료합니다.")
        break

    label = determine_label(user_input)
    input_text = f"{START_TOKEN}레이블: {label}\n질문: {user_input}\n답변:"
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

    generated_answer = refine_response(input_ids, output_ids, tokenizer)
    print(f"챗봇: {generated_answer}")

    # 사용자 피드백 수집 및 보상 반영
    collect_feedback_and_apply_reward(input_ids, output_ids, model, optimizer, criterion)

# 대화 종료 후 손실 시각화
plot_losses()
