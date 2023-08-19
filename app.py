import cv2
from flask import Flask, Response, render_template, request
import io
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import OpenposeDetector
import atexit
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import mne
import matplotlib.pyplot as plt
import scipy
from flask_cors import CORS
import torch
from flask import Flask, Response, render_template, stream_with_context
import time
from mne import create_info
from scipy import signal
import pickle
from joblib import load
import warnings
warnings.filterwarnings(action='ignore')

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

# 웹캠 비디오 캡처 객체 생성
cap = cv2.VideoCapture(0)

def load_eeg_data():
    file_names = './datas/eeg_record3.mat'
    mat = scipy.io.loadmat(file_names)
    data = mat['o']['data'][0,0]
    sfreq = mat['o']['sampFreq'][0][0][0][0]
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    info = create_info(channel_names, sfreq, ch_types=['eeg']*len(channel_names))
    info.set_montage('standard_1020')
    info['description'] = 'AIMS'

    concatenated_data = data[5000:15000, 3:17].T
    return info, concatenated_data

########################################🌟 MNE TOPOLOGY###################################

# Precompute the processed EEG data
def preprocess_EEG_data(peoplelist, sfreq, band_pass_low, band_pass_high, sample_count):
    processed_EEG_data = {}
    for people in peoplelist:
        st = "./datas/" + people + '_epoched.set'
        st = st.replace(" ", "")
        epochs = mne.io.read_epochs_eeglab(st).apply_baseline(baseline=(-1, 0))
        epochs = epochs.set_eeg_reference('average').apply_baseline(baseline=(-0.5, 0))
        cropped_data = epochs.crop(tmin=0, tmax=3.999).get_data()
        downsampled_data = scipy.signal.resample(cropped_data, sfreq * 4, axis=2)
        filtered_data = mne.filter.filter_data(downsampled_data, sfreq, band_pass_low, band_pass_high)
        processed_EEG_data[people] = filtered_data[:sample_count]

    return processed_EEG_data

# Generate MNE topomaps
def generate_mne(info, concatenated_data):
    time_step = 0
    while True:
        if time_step > concatenated_data.shape[1]:
            time_step = 0

        # Plot the topomap
        fig = plt.figure(figsize=(3, 3), facecolor='none')
        ax = fig.add_subplot(111)

        mne.viz.plot_topomap(
            concatenated_data[:, time_step],
            info,
            vlim=(4000, 5000),
            axes=ax,
            show=False,
            outlines='head',
            cmap='jet',
            contours=0
        )

        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

        # Convert the plot to an image
        canvas = FigureCanvas(fig)
        buf = io.BytesIO()
        canvas.print_png(buf)
        buf.seek(0)

        time_step += 1 # x axis -> time stamp 정수로 뜨도록 수정 필요

        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + buf.read() + b'\r\n')

# Route to display MNE topomaps
@app.route('/mne_feed_model')
def mne_feed_model():
    info, concatenated_data = load_eeg_data()
    response = Response(generate_mne(info, concatenated_data), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    
    return response

@app.route('/mne_feed')
def mne_feed():
    return render_template('mne_feed.html')


########################################🌟 EEG PLOT###################################

def pull_vreed_data(concatenated_data, time_step):
    sample = concatenated_data[:,time_step]
    sample =sample*1e4

    if time_step > concatenated_data.shape[1]:
        time_step = 0
    time_step += 1

    return sample

def generate_random_data(concatenated_data):

    time_step = 0

    while True:
        sample = pull_vreed_data(concatenated_data, time_step).tolist()
        time_step += 1

        json_data = json.dumps(
            {'time': time_step, 'value': sample})

        yield f"data:{json_data}\n\n"
        time.sleep(0.04)

@app.route('/eeg_feed_model')
def eeg_feed_model():
    _, concatenated_data = load_eeg_data()
    response = Response(stream_with_context(generate_random_data(concatenated_data)), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response

@app.route('/eeg_feed')
def eeg_feed():
    return render_template('eeg_feed.html')





########################################🌟 ATTENTION PLOT###################################

# Define constants
CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
SFREQ = 128
BAND_PASS_LOW = 0.16
SELECT_CH = ['F7', 'F3', 'AF4', 'P7', 'P8', 'O1', 'O2']
TIME_WINDOW = 5

select_ch = ['F7', 'F3', 'AF4', 'P7', 'P8', 'O1', 'O2']
use_channel_inds = []

# Apply a Butterworth high-pass filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Apply a high-pass filter to EEG data
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    x = signal.filtfilt(b, a, data)
    y = signal.filtfilt(b, a, x)
    return y

# Extract features from EEG data
def extract_features(concatenated_eeg, time_window, time_points, window_blackman):

    col = len(select_ch) #7
    power_eeg = {}
    bin_eeg = {}
    bin_eeg_avg = {}
    knn_focus ={}

    power_eeg['data']=np.zeros([7, 513, time_window+1])
    bin_eeg['data']=np.zeros([7, 36, time_window+1])
    bin_eeg_avg['data']=np.zeros([7, 36, 1])
    knn_focus['data']=np.zeros([252,1])

    if concatenated_eeg.shape[0] < time_points:
        original_array = np.random.rand(concatenated_eeg.shape[0],7)
        additional_values = np.random.rand(time_points - concatenated_eeg.shape[0], 7)
        concatenated_eeg = np.concatenate((original_array, additional_values))

    else:
        pass
    
    for i in range(col):

        # print(concatenated_eeg[:,i].shape) # 1920
        concatenated_eeg[:,i] = butter_highpass_filter(concatenated_eeg[:,i], 0.16, 128, 5)
        f, t, y1=scipy.signal.stft(concatenated_eeg[:,i],fs=128, window=window_blackman, nperseg=128, 
                      noverlap=0, nfft=1024, detrend=False,return_onesided=True, boundary='zeros',
                      padded=True)
        # print(t)
        # print(np.abs(y1).shape) # (512, 16)
        power_eeg['data'][i,:, :]=(np.abs(y1))**2
        # print('power_eeg is ', power_eeg) # 7, 513 16
        
    for chn in range(col):
        j=0
        for i in range(1,144,4):
            bin_eeg['data'][chn,j,:]=np.average(power_eeg['data'][chn,i:i+4,:],axis=0)
            # print(power_eeg['data'][chn,i:i+4,:])
            j+=1

    for chn in range(col):
        j=0
        for k in range(0,1):
            bin_eeg_avg['data'][chn,:,j]=np.average(bin_eeg['data'][chn,:,k:k+(601-time_window+1)],axis=1)
            j += 1

    for j in range(1):      
        knn_focus['data'][:,j]=bin_eeg_avg['data'][:,:,j].reshape(1,-1)

    knn_focus['data'] = 10*np.log(knn_focus['data'])
    if time_window == 15:
        loaded_scaler = load('./models/scaler_knn.joblib')
        with open('./models/saved_model', 'rb') as f:
            mod = pickle.load(f)


    elif time_window == 5:
        loaded_scaler = load('./models/scaler_knn_5second.joblib')
        with open('./models/saved_model_5second', 'rb') as f:
            mod = pickle.load(f)

    elif time_window == 10:
        loaded_scaler = load('./models/scaler_knn_10second.joblib')
        with open('./models/saved_model_10second', 'rb') as f:
            mod = pickle.load(f)

    elif time_window == 1:
        loaded_scaler = load('./models/scaler_knn_1second.joblib')
        with open('./models/saved_model_1second', 'rb') as f:
            mod = pickle.load(f)

    return knn_focus['data'].T,loaded_scaler, mod

def get_attention(concatenated_data):
    time_step = 0
    time_points = TIME_WINDOW * SFREQ
    t_win = np.arange(0, 128)
    M = 12
    window_blackman = 0.42 - 0.5 * np.cos((2 * np.pi * t_win) / (M - 1)) + 0.08 * np.cos((4 * np.pi * t_win) / (M - 1))

    while True:
        if time_step > concatenated_data.shape[1]:
            time_step = 0

        samples = concatenated_data[:, time_step:time_step + time_points]
        time_step += time_points

        samples = np.array([row[3:17] for row in samples]).T
        realtime_data = samples[use_channel_inds, :]

        realtime_data, loaded_scaler, mod = extract_features(realtime_data.T, TIME_WINDOW, time_points, window_blackman)
        realtime_data_scaled = loaded_scaler.transform(realtime_data)
        
        value = mod.predict(realtime_data_scaled)[0]
        #result = "focus" if value == 0 else ("unfocus" if value == 1 else ("drowsy" if value == 2 else "unknown"))

        yield f"data: {value}\n\n"
        
        time.sleep(0.1)  # Pause for 0.5 seconds before the next update

@app.route('/attention_feed_model')
def attention_feed_model():
    _, concatenated_data = load_eeg_data()
    response = Response(stream_with_context(get_attention(concatenated_data)), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response

@app.route('/attention_feed')
def attention_feed():
    return render_template('attention_feed.html')


########################################🌟 DIFFUSION MODEL###################################

cmd = "supermario"

# 이미지를 생성하여 스트리밍하는 함수
def generate_images(openpose, pipe):
    global cmd

    while True:
        ret, frame = cap.read()
        pose_img = openpose(frame)
        image_output = pipe(cmd + ", masterpiece,  distinct_image, high_contrast, 8K, best quality, high_resolution", pose_img, negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality", num_inference_steps=15).images[0]
        
        # 이미지를 바이트 형태로 변환
        img_byte_array = io.BytesIO()
        image_output.save(img_byte_array, format='JPEG')
        img_bytes = img_byte_array.getvalue()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

@app.route('/diffusion_post_cmd', methods = ['POST'])
def diffusion_post_cmd():
    global cmd
    data = request.get_json()
    cmd = data.get('text_input')

    return Response(status=200)

@app.route('/diffusion_feed_model', methods = ['GET'])
def diffusion_feed_model():
    # OpenPose 모델 및 Diffusion 초기화
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.enable_xformers_memory_efficient_attention()

    response = Response(generate_images(openpose, pipe), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response

@app.route('/diffusion_feed')
def diffusion_feed():
    return render_template('diffusion_feed.html')


########################################🌟 EMOTION RECOGNITION###################################

@app.route('/emotion_feed_model')
def emotion_feed_model():
    emotions = ["sad", "disgust", "angry", "surprise", "fear", "neutral", "happy"]

    def generate_emotion_data():
        while True:
            success, frame = cap.read()                               
            if success:
                # Perform emotion analysis
                predictions = DeepFace.analyze(frame, actions=['emotion'], detector_backend="opencv", enforce_detection=False, silent=True)
                emotion_data = predictions[0]['emotion']
                probabilities = [emotion_data[emotion] for emotion in emotions]

                # Create JSON data to send to the front-end
                json_data = json.dumps({'emotions': emotions, 'probabilities': probabilities})
                
                yield f"data:{json_data}\n\n"
            
    response = Response(generate_emotion_data(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response

@app.route('/emotion_feed')
def emotion_feed():
    return render_template('emotion_feed.html')

########################################🌟 POSE ESTIMATION###################################

def generate_frames(mp_holistic):
    while True:
        success, frame = cap.read()  # 프레임 읽기
        if success:
            # Mediapipe Holistic 모델로 landmark 감지
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_holistic.process(frame_rgb)

            # Holistic landmark 그리기
            if results.pose_landmarks:
                # 연결선 그리기
                for connection in mp.solutions.holistic.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = tuple(np.multiply([results.pose_landmarks.landmark[start_idx].x, results.pose_landmarks.landmark[start_idx].y], [frame.shape[1], frame.shape[0]]).astype(int))
                    end_point = tuple(np.multiply([results.pose_landmarks.landmark[end_idx].x, results.pose_landmarks.landmark[end_idx].y], [frame.shape[1], frame.shape[0]]).astype(int))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
                
                # 랜드마크 점 그리기
                for landmark in results.pose_landmarks.landmark:
                    landmark_point = tuple(np.multiply([landmark.x, landmark.y], [frame.shape[1], frame.shape[0]]).astype(int))
                    cv2.circle(frame, landmark_point, 2, (255, 255, 255), -1)

            # 프레임을 바이트로 변환하여 스트리밍
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# 비디오 스트리밍 라우트
@app.route('/pose_feed_model')
def pose_feed_model():
    # Mediapipe Holistic 모델 초기화
    mp_holistic = mp.solutions.holistic.Holistic(model_complexity=1)

    response = Response(generate_frames(mp_holistic), mimetype='multipart/x-mixed-replace; boundary=frame')
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"

    return response

@app.route('/pose_feed')
def pose_feed():
    return render_template('pose_feed.html')


###########################################################################################

# Flask 애플리케이션 종료시에 비디오 캡처 릴리즈
@atexit.register
def release_capture():
    cap.release()

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
