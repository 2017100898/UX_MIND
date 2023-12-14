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

from sentence_transformers import SentenceTransformer
import keras
from flask import Flask, request, render_template, Response
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


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


# Function to perform emotion analysis
def analyze_emotion():
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise"]

    def generate_prediction():
        while True:
            r = sr_lib.Recognizer()

            with sr_lib.Microphone() as source:
                print("Say something!")
                audio = r.listen(
                    source, timeout=10, phrase_time_limit=10
                )  # Wait for input for 10 seconds

            # WAV file overwrite
            file_name = "recorded_audio.wav"
            with open(file_name, "wb") as f:
                f.write(audio.get_wav_data())

            audio_features = get_features(file_name)

            # reset
            recognizer = sr_lib.Recognizer()
            # Read audio file
            with sr_lib.AudioFile(file_name) as source:
                audio_data = recognizer.record(source)
            try:
                audio_text = recognizer.recognize_google(audio_data, language="ko-KR")
                print(audio_text)
            except sr_lib.UnknownValueError:
                print("Google Speech Recognition did not understand the audio.")
                yield 'data:{"error": "Recognition failed"}\n\n'
                continue
            except sr_lib.RequestError as e:
                print(
                    "Unable to request results from Google speech recognition service; {0}".format(
                        e
                    )
                )
                yield 'data:{"error": "Request error"}\n\n'
                continue

            model_name = "jhgan/ko-sbert-multitask"
            embedding_model = SentenceTransformer(model_name)
            embedding_vec = embedding_model.encode(audio_text)

            final_features = np.concatenate((audio_features, embedding_vec))
            final_features_reshaped = final_features.reshape(1, 1254, 1)
            predictions = emotion_model.predict(final_features_reshaped)

            predicted_labels = np.argmax(predictions, axis=1)
            predicted_probabilities = predictions  # Convert numpy array to list
            print(predicted_probabilities)
            json_data = json.dumps(
                {"emotions": emotions, "probabilities": predicted_probabilities}
            )

            yield f"data:{json_data}\n\n"

    return generate_prediction()


# Load the emotion analysis model
emotion_model = keras.models.load_model("./models/emotion_model3000.h5")


@app.route("/audio_feed")
def audio_feed():
    return render_template("audio_feed.html")


@app.route("/audio_feed_model")
def start_analysis():
    def analyze():
        response = analyze_emotion()
        try:
            for line in response:
                yield line
        except Exception as e:
            print(f"Exception occurred: {e}")
            yield "An error occurred"

    return Response(analyze(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app
