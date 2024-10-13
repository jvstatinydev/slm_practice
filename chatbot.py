# chatbot.py - 3. 챗봇 응용 단계
# 이 파일은 파인튜닝된 모델을 사용하여 사용자가 입력한 질문에 응답하는 챗봇을 구현합니다.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import END_TOKEN, MODEL_DIR, PAD_TOKEN, START_TOKEN


# 레이블 결정 함수
def determine_label(user_input):
    # 입력된 질문을 바탕으로 레이블을 결정합니다
    if any(keyword in user_input for keyword in ["슬퍼", "힘들어", "눈물", "이별", "망쳤어"]):
        return "1"  # 이별(부정)
    elif any(keyword in user_input for keyword in ["사랑", "고백", "행복", "좋아"]):
        return "2"  # 사랑(긍정)
    else:
        return "0"  # 일상다반사
    
    
# GPU 설정
# 모델을 GPU에 올려서 연산 속도를 향상시킵니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파인튜닝된 모델 로드
# 저장된 모델과 토크나이저를 로드합니다.
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(device)

# 토크나이저에 pad_token 추가
# 모델 학습 시 필요한 pad_token이 설정되지 않았을 경우 추가합니다.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))

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
        min_length=25,

        # 확률 분포를 기반으로 샘플링. False일 경우 가장 높은 확률의 토큰을 고정적으로 선택. True일 경우 다양하고 창의적인 응답을 생성.
        # True인 경우 temperature, top_p 값 설정 가능.
        do_sample=True,

        # 생성된 텍스트의 다양성을 조절하는 파라미터. 낮은 값은 더 결정적인 응답을 생성하며, 높은 값은 더 창의적이거나 예측 불가능한 응답
        temperature=0.7,
        
        # 확률 축적 방법을 사용해 생성할 토큰을 결정하는 top-p 샘플링. 다양성을 유지하면서 비정상적인 토큰이 선택되는 것을 방지.
        # 낮은 top_p 값은 모델이 더 결정적이고 예측 가능한 응답을 생성하게 합니다.
        # 높은 top_p 값은 모델이 더 창의적이고 다양성 있는 응답을 생성하게 합니다.
        top_p=0.9,

        # 반복되는 단어에 대해 페널티를 부여해 응답의 품질을 높임. 1을 기준으로 높은 값은 반복을 피하고 낮은 값은 반복을 허용.
        repetition_penalty=2.0,

        # 입력의 길이를 맞추기 위해 사용
        pad_token_id=tokenizer.pad_token_id,

        # 모델은 이 종료 토큰을 만나면 생성 프로세스를 중단. 응답의 끝을 자연스럽게 인식하고 멈출 수 있음.
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # 입력 텍스트 길이만큼 잘라서 생성된 응답만 추출
    generated_answer = response[len(tokenizer.decode(input_ids[0], skip_special_tokens=False)):]
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

    print(f"챗봇: {generated_answer}")
