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
2. **모델 훈련 (train)**

   ```sh
   python train_model.py
   ```
3. **챗봇 실행 (chat)**

   ```sh
   python chatbot.py
   ```
## 실행예시

   ```
   챗봇이 준비되었습니다. 종료하려면 '종료' 또는 'exit'를 입력하세요.
   사용자: 오늘 몸이 아파
   챗봇: 감기 조심하세요. 어서 주무세요! 기운 내길 바랍니다. 응원합니다~~~
   사용자:  초코송이 좋아해?
   챗봇: 맛있게 먹으면 영칼로리!!!!!!!
   사용자: 종료
   챗봇을 종료합니다.
   ```
