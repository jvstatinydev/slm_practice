from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

from config import MODEL_DIR, RESULTS_DIR, TRAIN_TXT

# 1. 토크나이저 및 모델 불러오기
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")

# 2. 데이터셋 로드
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
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)
