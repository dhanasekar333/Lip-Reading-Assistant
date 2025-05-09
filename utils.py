
import tensorflow as tf
import cv2
import numpy as np

# Vocabulary
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

def load_data(path):
    path = path.numpy().decode("utf-8")
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame[190:236, 80:220]
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    if frames.shape[0] < 75:
        pad_width = 75 - frames.shape[0]
        frames = np.pad(frames, ((0, pad_width), (0, 0), (0, 0)), mode='constant')
    else:
        frames = frames[:75]

    return tf.convert_to_tensor(frames[..., np.newaxis], dtype=tf.float32), None
