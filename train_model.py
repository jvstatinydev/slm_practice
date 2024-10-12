# train_model.py - 2. 모델 학습 단계
# 이 파일은 LLMOps 파이프라인에서 모델을 파인튜닝하는 과정을 구현합니다.

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from config import MODEL_DIR, RESULTS_DIR, TRAIN_TXT, END_TOKEN, START_TOKEN

def load_tokenizer_and_model():
    """토크나이저와 모델을 불러오고, 커스텀 토큰을 추가합니다."""
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    new_tokens = [START_TOKEN, END_TOKEN]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def load_dataset(file_path, tokenizer):
    """데이터셋을 불러오고 토큰화합니다."""
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

def get_data_collator(tokenizer):
    """언어 모델링을 위한 데이터 콜레이터를 가져옵니다."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

def train_model(model, tokenizer, dataset, data_collator):
    """주어진 데이터셋과 데이터 콜레이터로 모델을 학습시킵니다."""
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
    return trainer

def save_model(trainer, tokenizer):
    """파인튜닝된 모델과 토크나이저를 저장합니다."""
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

def main():
    tokenizer, model = load_tokenizer_and_model()
    dataset = load_dataset(TRAIN_TXT, tokenizer)
    data_collator = get_data_collator(tokenizer)
    trainer = train_model(model, tokenizer, dataset, data_collator)
    save_model(trainer, tokenizer)

if __name__ == "__main__":
    main()