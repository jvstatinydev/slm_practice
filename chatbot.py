import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import END_TOKEN, MODEL_DIR, PAD_TOKEN, START_TOKEN

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 파인튜닝된 모델 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.to(device)

# 토크나이저에 pad_token 추가
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    model.resize_token_embeddings(len(tokenizer))

def determine_label(user_input):
    # 입력된 질문을 바탕으로 레이블을 결정합니다
    if any(keyword in user_input for keyword in ["슬퍼", "힘들어", "눈물", "이별", "망쳤어"]):
        return "1"  # 이별(부정)
    elif any(keyword in user_input for keyword in ["사랑", "고백", "행복", "좋아"]):
        return "2"  # 사랑(긍정)
    else:
        return "0"  # 일상다반사

# 챗봇 대화 루프
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
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        # early_stopping=True
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

    if len(generated_answer) == 0 or generated_answer.isspace():
        generated_answer = "죄송해요, 잘 이해하지 못했어요."

    print(f"챗봇: {generated_answer}")
