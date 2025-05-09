
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPool3D, TimeDistributed, Flatten, Bidirectional, LSTM, Dropout, Dense
import os

def load_model():
    model = Sequential([
        Conv3D(128, 3, padding='same', activation='relu', input_shape=(75, 46, 140, 1)),
        MaxPool3D((1, 2, 2)),
        Conv3D(256, 3, padding='same', activation='relu'),
        MaxPool3D((1, 2, 2)),
        Conv3D(75, 3, padding='same', activation='relu'),
        MaxPool3D((1, 2, 2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.5),
        Dense(41, activation='softmax')
    ])

    weights_path = os.path.join('models', 'checkpoint.weights.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    return model
