# train_model.py - 2. 모델 학습 단계
# 이 파일은 LLMOps 파이프라인에서 모델을 파인튜닝하는 과정을 구현합니다.
import torch
from transformers import get_linear_schedule_with_warmup
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

        # 작은 값일 수록 gpu 메모리 사용량이 줄어듭니다.
        block_size=64
    )


def get_data_collator(tokenizer):
    """언어 모델링을 위한 데이터 콜레이터를 가져옵니다."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )


def train_model(model, tokenizer, dataset, data_collator):
    """주어진 데이터셋과 데이터 콜레이터로 모델을 학습시킵니다."""
    training_args = TrainingArguments(
        # 학습 결과를 저장할 디렉토리
        output_dir=RESULTS_DIR,

        # 기존 출력 디렉토리를 덮어쓸지 여부
        overwrite_output_dir=True,

        # 학습 반복 횟수, 모델이 데이터셋을 몇 번 반복 학습할지 설정
        num_train_epochs=5,

        # 각 디바이스(GPU 등)당 배치 크기, 한 번에 처리할 데이터 샘플 수
        per_device_train_batch_size=2,

        # 그래디언트를 누적할 스텝 수, 메모리 절약을 위해 사용
        gradient_accumulation_steps=2,

        # 학습률, 모델이 학습할 때 가중치를 업데이트하는 속도
        learning_rate=2e-5,

        # 모델을 저장할 스텝 간격, n 스텝마다 모델을 저장
        save_steps=1000,

        # 저장할 모델의 최대 개수, 최신 n개의 모델만 저장
        save_total_limit=2,

        # 학습 중 평가 메트릭을 확인할지 여부, False로 설정 시 평가 메트릭을 확인
        prediction_loss_only=False,

        # 혼합 정밀도(16-bit floating point) 학습 사용 여부, GPU가 지원하면 True로 설정
        fp16=False,
    )

    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)

    # 총 학습 스텝 수 계산
    total_steps = (len(dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs

    # 러닝 레이트 스케줄러 설정
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Trainer 설정 시 옵티마이저와 스케줄러를 전달
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        optimizers=(optimizer, scheduler)  # 여기에서 옵티마이저와 스케줄러를 전달합니다.
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