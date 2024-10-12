# train_model.py - 2. 모델 학습 단계
# 이 파일은 LLMOps 파이프라인에서 모델을 파인튜닝하는 과정을 구현합니다.

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from config import MODEL_DIR, RESULTS_DIR, TRAIN_TXT, END_TOKEN, START_TOKEN

# 1. 토크나이저 및 모델 불러오기
# 사전 학습된 KoGPT2 모델과 토크나이저를 불러옵니다.
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")

# 커스텀 토큰 목록 추가
new_tokens = [START_TOKEN, END_TOKEN]
num_added_tokens = tokenizer.add_tokens(new_tokens)

# 모델의 임베딩 레이어를 새로 추가된 토큰에 맞게 확장
model.resize_token_embeddings(len(tokenizer))

# 2. 데이터셋 로드
# 학습에 사용할 데이터를 로드하고 토큰화합니다.
def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

dataset = load_dataset(TRAIN_TXT, tokenizer)
data_collator = get_data_collator(tokenizer)

# 3. 모델 파인튜닝
# 학습을 위한 설정을 정의하고 Trainer를 이용해 모델을 파인튜닝합니다.
training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

# 4. 파인튜닝된 모델 저장
# 파인튜닝된 모델과 토크나이저를 저장합니다.
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
