from datetime import datetime
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow import keras

if __name__ == '__main__':
    # 4. Setup Folders for Collection
    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join(os.path.dirname(__file__), '../MP_Data') 
    # Actions that we try to detect
    actions = np.array(['Idle', 'RSw', 'LSw', 'ByeBye', 'Boxing' ])
    # Thirty videos worth of data
    no_sequences = 30
    # Videos are going to be 30 frames in length
    sequence_length = 30

    # 6. Preprocess Data and Create Labels and Features    
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    y = keras.utils.to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # 7. Build LSTM Neural Network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(keras.layers.LSTM(128, return_sequences=True, activation='relu'))
    model.add(keras.layers.LSTM(64, return_sequences=False, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # 7. Train LSTM Neural Network
    log_dir = os.path.join(os.path.dirname(__file__), 'Logs')
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    model.fit(X_train, y_train, epochs=100, callbacks=[tb_callback])

    # Disp summary
    model.summary()

    # 9. Save Weights
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model.save(os.path.join(os.path.dirname(__file__), f'../models/lstm_action_model_{timestamp}.h5'))
