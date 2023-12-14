import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import pyaudio
import wave
import librosa
import librosa.display
from IPython.display import Audio
import speech_recognition as sr_lib
import json
from tensorflow import keras
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import (
    Dense,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.python.keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# data argumentation
def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


# feature 생성
def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))
    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))
    return result


# feature vector
def get_features(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.0)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.concatenate((result, res2), axis=0)

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.concatenate((result, res3), axis=0)

    return result


emotion_model = keras.models.load_model("./models/emotion_model3000.h5")
file_name = "5e2b035e5807b852d9e0220c.wav"
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"]
audio_features = get_features(file_name)

# 초기화
recognizer = sr_lib.Recognizer()

# 오디오 파일 읽기
with sr_lib.AudioFile(file_name) as source:
    audio_data = recognizer.record(source)

# Google Web Speech API를 사용하여 오디오 인식
try:
    text = recognizer.recognize_google(audio_data, language="ko-KR")  # 한국어로 설정
    print("음성 파일에서 추출한 텍스트:", text)
except sr_lib.UnknownValueError:
    print("Google 음성 인식이 오디오를 이해하지 못했습니다.")
except sr_lib.RequestError as e:
    print("Google 음성 인식 서비스에 대한 결과를 요청할 수 없습니다; {0}".format(e))

model_name = "jhgan/ko-sbert-multitask"
embedding_model = SentenceTransformer(model_name)
embedding_vec = embedding_model.encode(text)

final_features = np.concatenate((audio_features, embedding_vec))
final_features_reshaped = final_features.reshape(1, 1254, 1)
predictions = emotion_model.predict(final_features_reshaped)

predicted_labels = np.argmax(predictions, axis=1)
predicted_probabilities = predictions

print(predicted_labels)
print(predicted_probabilities)
