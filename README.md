## Mediapipeデモ（要WebCamera）
### ハンド
src/hand_mp.py

### ポーズ
src/holistic_mp.py

### ポーズ（作成したモデルを使用したテスト）
src/holistic_test_realtime.py

## モデル作成
### 訓練データの取得
src/holistic_collect_data.py
取得した訓練データはMP_Dataフォルダの中に作成されます

### モデルの学習
src/holistic_build_lstm_nn.py
モデルはmodelsフォルダに作成されます

## Setup
pip install tensorflow opencv-python mediapipe scikit-learn matplotlib
