# train_model.py - 2. 모델 학습 단계
# 이 파일은 LLMOps 파이프라인에서 모델을 파인튜닝하는 과정을 구현합니다.

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
import optuna
from config import MODEL_DIR, RESULTS_DIR, TRAIN_TXT, END_TOKEN, START_TOKEN, TRAIN_PARAMETER_JSON
import torch
from torch.utils.data import random_split

class OptunaCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial

    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs.get("metrics", {}).get("eval_loss")
        if eval_loss is not None:
            self.trial.report(eval_loss, state.global_step)

        # Prune trial if needed
        if self.trial.should_prune():
            raise optuna.TrialPruned()

def load_tokenizer_and_model():
    """토크나이저와 모델을 불러오고, 커스텀 토큰을 추가합니다."""
    tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
    model = AutoModelForCausalLM.from_pretrained("skt/kogpt2-base-v2")
    new_tokens = [START_TOKEN, END_TOKEN]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def load_dataset(file_path, tokenizer, block_size):
    """데이터셋을 불러오고 토큰화합니다."""
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

def get_data_collator(tokenizer):
    """언어 모델링을 위한 데이터 콜레이터를 가져옵니다."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )

def load_hyperparameters(file_path):
    """최적의 하이퍼파라미터를 JSON 파일에서 불러옵니다."""
    with open(file_path, 'r') as f:
        return json.load(f)

def train_model_with_hyperparameters(batch_size, num_train_epochs, learning_rate, block_size, trial=None):
    # GPU 사용 여부 확인
    if torch.cuda.is_available():
        print("GPU 사용 중: GPU 가속을 사용하여 학습합니다.")
    else:
        print("GPU 사용 불가: CPU로 학습합니다.")

    """하이퍼파라미터를 인자로 받아 모델을 학습하고 검증 손실을 반환합니다."""
    tokenizer, model = load_tokenizer_and_model()
    dataset = load_dataset(TRAIN_TXT, tokenizer, block_size)
    data_collator = get_data_collator(tokenizer)

    # 데이터셋을 학습 및 검증 데이터로 분할
    # 튜닝 시에만 데이터셋의 일부를 샘플링 (trial이 있을 때)
    if trial is not None:
        print("Optuna 튜닝 중: 데이터셋 10% 샘플링")
        subset_size = int(len(dataset) * 0.1)  # 데이터의 10%만 사용
        valid_length = int(subset_size * 0.1)  # 그 중 10%를 검증 데이터로 사용
        train_length = subset_size - valid_length
        # 전체 데이터셋에서 10%만 샘플링하여 train/valid로 나누기
        subset_dataset, _ = random_split(dataset, [subset_size, len(dataset) - subset_size])
        train_dataset, valid_dataset = random_split(subset_dataset, [train_length, valid_length])
    else:
        print("전체 데이터 사용 중: 실제 학습")
        valid_length = int(len(dataset) * 0.1)  # 전체 데이터셋 기준 10% 검증 데이터
        train_length = len(dataset) - valid_length
        train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length])

    # 학습 인자 설정
    training_args = TrainingArguments(
        # 학습 결과를 저장할 디렉토리
        output_dir=RESULTS_DIR,

        # 기존 출력 디렉토리를 덮어쓸지 여부
        overwrite_output_dir=True,

        # 학습 반복 횟수, 모델이 데이터셋을 몇 번 반복 학습할지 설정
        num_train_epochs=num_train_epochs,

        # 각 디바이스(GPU 등)당 배치 크기, 한 번에 처리할 데이터 샘플 수
        per_device_train_batch_size=batch_size,

        # 그래디언트를 누적할 스텝 수, 메모리 절약을 위해 사용
        gradient_accumulation_steps=1,

        # 학습률, 모델이 학습할 때 가중치를 업데이트하는 속도
        learning_rate=learning_rate,

        # 모델을 저장할 스텝 간격, n 스텝마다 모델을 저장
        save_steps=500,

        # 저장할 모델의 최대 개수, 최신 n개의 모델만 저장
        save_total_limit=2,

        # 평가 전략, 'steps'로 설정하여 일정 스텝마다 평가
        eval_strategy='steps',

        # 평가 스텝 간격, n 스텝마다 평가
        eval_steps=500,

        # 로그 출력 스텝 간격, n 스텝마다 로그 출력
        logging_steps=500,

        # 학습 종료 시 가장 좋은 모델을 로드할지 여부
        load_best_model_at_end=True,

        # 가장 좋은 모델을 결정할 평가 메트릭, 'eval_loss'로 설정
        metric_for_best_model='eval_loss',

        # 평가 메트릭이 낮을수록 좋은지 여부, False로 설정
        greater_is_better=False,

        # 혼합 정밀도(16-bit floating point) 학습 사용 여부, GPU가 지원하면 True로 설정
        fp16=torch.cuda.is_available(),

        # 로그를 출력으로만 제한
        report_to=None,
    )

    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    # Optuna의 Pruner 사용
    if trial is not None:
        trainer.add_callback(OptunaCallback(trial))

    # 모델 학습
    trainer.train()

    # 검증 손실 계산
    eval_result = trainer.evaluate()
    validation_loss = eval_result['eval_loss']

    return validation_loss

def main():
    # JSON 파일에서 하이퍼파라미터 로드
    best_hyperparameters = load_hyperparameters(TRAIN_PARAMETER_JSON)

    # 하이퍼파라미터 적용하여 학습
    train_model_with_hyperparameters(
        batch_size=best_hyperparameters['batch_size'],
        num_train_epochs=best_hyperparameters['num_train_epochs'],
        learning_rate=best_hyperparameters['learning_rate'],
        block_size=best_hyperparameters['block_size']
    )

if __name__ == "__main__":
    main()
