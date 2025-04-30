# train_model.py - 2. 모델 학습 단계
# 이 파일은 LLMOps 파이프라인에서 모델을 파인튜닝하는 과정을 구현합니다.

import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, TrainerCallback
import optuna
from datasets import load_dataset
from config import VALID_TXT, MODEL_DIR, RESULTS_DIR, TRAIN_TXT
from config import PAD_TOKEN, END_TOKEN, START_TOKEN
from config import TRAIN_PARAMETER_JSON, DEFAULT_TRAIN_PARAMETER_JSON
import torch

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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def load_dataset_hf(file_path, tokenizer):
    dataset = load_dataset('text', data_files=file_path)
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=True),
        batched=True
    )
    return tokenized_dataset['train']

def get_data_collator(tokenizer, block_size):
    """언어 모델링을 위한 데이터 콜레이터를 가져오고 패딩을 적용합니다."""
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=block_size
    )

def load_hyperparameters(file_path):
    """최적의 하이퍼파라미터를 JSON 파일에서 불러옵니다."""
    if not os.path.exists(file_path):
        file_path = DEFAULT_TRAIN_PARAMETER_JSON
    
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
    train_dataset = load_dataset_hf(TRAIN_TXT, tokenizer)
    eval_dataset = None
    data_collator = get_data_collator(tokenizer, block_size)

    if os.path.exists(VALID_TXT):
        eval_dataset = load_dataset_hf(VALID_TXT, tokenizer)
        print(f"검증 데이터셋 로드 완료: {VALID_TXT}")
    else:
        print("검증 데이터셋 파일이 없습니다.")

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

        dataloader_num_workers=os.cpu_count() // 2, # 사용 가능한 코어 수의 절반

        # 그래디언트를 누적할 스텝 수, 메모리 절약을 위해 사용
        gradient_accumulation_steps=1,

        # 학습률, 모델이 학습할 때 가중치를 업데이트하는 속도
        learning_rate=learning_rate,

        lr_scheduler_type='cosine',

        # 모델을 저장할 스텝 간격, n 스텝마다 모델을 저장
        save_steps=500,

        # 저장할 모델의 최대 개수, 최신 n개의 모델만 저장
        save_total_limit=2,

        # 평가 전략, 'steps'로 설정하여 일정 스텝마다 평가
        eval_strategy='steps' if eval_dataset else 'no',

        # 평가 스텝 간격, n 스텝마다 평가
        eval_steps=500 if eval_dataset else None,

        # 로그 출력 스텝 간격, n 스텝마다 로그 출력
        logging_steps=500,

        # 학습 종료 시 가장 좋은 모델을 로드할지 여부
        load_best_model_at_end=True if eval_dataset else False,

        # 가장 좋은 모델을 결정할 평가 메트릭, 'eval_loss'로 설정
        metric_for_best_model='eval_loss' if eval_dataset else None,

        # 평가 메트릭이 낮을수록 좋은지 여부, False로 설정
        greater_is_better=False if eval_dataset else None,

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
        eval_dataset=eval_dataset,
    )

    # Optuna의 Pruner 사용
    if trial is not None:
        trainer.add_callback(OptunaCallback(trial))

    # 모델 학습
    trainer.train(resume_from_checkpoint=True)

    """파인튜닝된 모델과 토크나이저를 저장합니다."""
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

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
