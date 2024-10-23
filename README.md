# 기본 챗봇 연습 예제

- 챗봇 훈련 데이터: [https://github.com/songys/Chatbot_data](https://github.com/songys/Chatbot_data)
- 한국어 언어 모델: [https://github.com/SKT-AI/KoGPT2](https://github.com/SKT-AI/KoGPT2)

## 준비
python 버전 3.12

필요 패키지 설치
- Mac
   ```
   pip install -r requirements_mac.txt
   ```
- Windows
   ```
   pip install -r requirements_windows.txt
   ```

## 사용방법

1. **데이터 준비 (prepare)**

   ```sh
   python prepare_data.py
   ```
2. **모델 훈련 파라미터 튜닝 (hyperparameter)**

   (default_train_parameter.json 의 기본값이 적합하지 않은 경우에 사용)

   ```sh
   python hyperparameter_tuning.py
   ```
3. **모델 훈련 (train)**

   ```sh
   python train_model.py
   ```
4. **챗봇 실행 (chat)**

   ```sh
   chainlit run chatbot.py -w --port 원하는_포트번호
   ```
4. **수집된 피드백 학습 (RLHF)**

   ```sh
   python train_model_with_feedback.py
   ```
## 실행예시

 ![img.png](example/exampleRun.png)
