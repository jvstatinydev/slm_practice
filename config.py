BASE_DIR = "."
DATA_DIR = f"{BASE_DIR}/data"
MODEL_DIR = f"{BASE_DIR}/fine_tuned_model"
RESULTS_DIR = f"{BASE_DIR}/results"
TRAIN_JSON_DIR = f"{DATA_DIR}/020.주제별 텍스트 일상 대화 데이터/01.데이터/1.Training/라벨링데이터"
VALID_JSON_DIR = f"{DATA_DIR}/020.주제별 텍스트 일상 대화 데이터/01.데이터/2.Validation/라벨링데이터"

DATA_CSV = f"{DATA_DIR}/data.csv"
TRAIN_TXT = f"{DATA_DIR}/train.txt"
VALID_TXT = f"{DATA_DIR}/valid.txt"
TRAIN_PARAMETER_JSON = f"{BASE_DIR}/train_parameter.json"
DEFAULT_TRAIN_PARAMETER_JSON = f"{BASE_DIR}/default_train_parameter.json"

START_TOKEN = f"<|startoftext|>"
END_TOKEN = f"<|endoftext|>"
PAD_TOKEN = f"<|pad|>"
MASK_TOKEN = f"<|mask|>"
SENT_TOKEN = f"<|sent|>"

# 기타 설정
MAX_EPOCHS = 5
PATIENCE = 2
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY=1e-4