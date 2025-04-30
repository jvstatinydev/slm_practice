# hyperparameter_tuning.py
# Optuna를 활용하여 하이퍼파라미터를 자동으로 탐색하는 스크립트

import optuna
import json
import psutil
from train_model import train_model_with_hyperparameters
from config import TRAIN_PARAMETER_JSON


def get_optimal_n_trials():
    # 시스템의 CPU 코어 수와 메모리 크기를 기준으로 n_trials 값을 설정
    cpu_count = psutil.cpu_count(logical=True)
    memory = psutil.virtual_memory().total / (1024 ** 3)  # GB 단위로 변환

    # 예시: CPU 코어 수와 메모리 크기에 따라 n_trials 값을 조정
    if cpu_count >= 8 and memory >= 16:
        return 30
    elif cpu_count >= 4 and memory >= 8:
        return 10
    else:
        return 5

def objective(trial):
    # 최적화할 하이퍼파라미터 범위 설정
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    block_size = trial.suggest_categorical('block_size', [64, 128, 256])

    # 학습 함수 호출 및 검증 손실 반환
    validation_loss = train_model_with_hyperparameters(
        batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        block_size=block_size,
        trial=trial  # Pruner를 위해 trial 객체 전달
    )
    return validation_loss

def save_best_hyperparameters(params, file_path):
    """최적의 하이퍼파라미터를 JSON 파일로 저장합니다."""
    with open(file_path, 'w') as f:
        json.dump(params, f, indent=4)

if __name__ == "__main__":
    n_trials = get_optimal_n_trials()
    # 스터디 생성
    study = optuna.create_study(direction='minimize')
    # 하이퍼파라미터 최적화 수행
    study.optimize(objective, n_trials=n_trials)

    # 최적의 하이퍼파라미터 출력 및 저장
    best_params = study.best_params
    print("최적의 파라미터:", best_params)

    # 최적의 하이퍼파라미터를 JSON 파일로 저장
    save_best_hyperparameters(best_params, TRAIN_PARAMETER_JSON)