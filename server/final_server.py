import os
import datetime
import torch
import cv2
import numpy as np
import pandas as pd
import torchaudio
import re
from collections import Counter
from flask import Flask, request, jsonify, send_file, render_template
from flask_socketio import SocketIO, emit
import mediapipe as mp
import pickle
import torch.nn as nn
import base64
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from collections import deque
import warnings
import tempfile
import soundfile as sf

app = Flask(__name__)
socketio = SocketIO(app)

API_KEY = "1234"
print("Your API Key:", API_KEY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sign-to-Text load ---
sign_model_path = r"D:\server\best_transformer_final.pt"
sign_metadata_path = r"D:\server\metadata.pkl"

class SimpleTransformerClassifier(nn.Module):
    def __init__(self, input_size=1662, d_model=512, num_heads=12, num_layers=6, num_classes=2000):
        super(SimpleTransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.pooling(x.transpose(1, 2)).squeeze(-1)
        x = self.batch_norm(x)
        return self.fc(x)

with open(sign_metadata_path, 'rb') as f:
    metadata = pickle.load(f)
num_classes = metadata['num_classes']
label_encoder = metadata['label_encoder']
index_to_label = {i: label for i, label in enumerate(label_encoder.classes_)} if hasattr(label_encoder, 'classes_') else label_encoder

sign_to_text_model = SimpleTransformerClassifier(num_classes=num_classes).to(device)
sign_to_text_model.load_state_dict(torch.load(sign_model_path, map_location=device))
sign_to_text_model.eval()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

keypoints_buffer = deque(maxlen=60)

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return image

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


speech_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
speech_asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
speech_to_sign_model = torch.jit.load(r'D:\server\text_to_sign_model.pt', map_location=device)
speech_to_sign_model.eval()

vocab_size = speech_to_sign_model.embedding.weight.shape[0]
print(f"Model vocab size: {vocab_size}")

corpus_csv_path = r"D:\server\corpus.csv"
corpus_df = pd.read_csv(corpus_csv_path)
all_text = " ".join(corpus_df['SENTENCE'].dropna().astype(str).tolist())
tokens = re.findall(r'\w+', all_text.lower())
vocab_counter = Counter(tokens)
vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + sorted(vocab_counter.keys())
word2idx = {w: i for i, w in enumerate(vocab) if i < vocab_size}  # vocab_size 제한
idx2word = {i: w for w, i in word2idx.items()}
unk_idx = word2idx.get('<unk>', 0)
pad_idx = word2idx.get('<pad>', 0)

if len(word2idx) > vocab_size:
    print(f"Warning: word2idx size ({len(word2idx)}) exceeds model vocab size ({vocab_size}). Truncating...")
    word2idx = {w: i for w, i in word2idx.items() if i < vocab_size}
    idx2word = {i: w for w, i in word2idx.items()}

max_text_len = 50
output_dim = 66

def speech_to_text(audio_file_path):
    speech_array, sampling_rate = torchaudio.load(audio_file_path)
    speech_array = speech_array.squeeze().numpy()
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        speech_array = resampler(torch.tensor(speech_array)).numpy()
    inputs = speech_processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = speech_asr_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    return speech_processor.batch_decode(predicted_ids)[0]

def encode_sentence(sentence):
    tokens = re.findall(r'\w+', sentence.lower())
    tokens = ['<sos>'] + tokens + ['<eos>']
    return [word2idx.get(t, 0) for t in tokens]

def generate_keypoints_for_sentence(sentence):
    encoded_text = encode_sentence(sentence)
    if len(encoded_text) < max_text_len:
        encoded_text += [word2idx['<pad>']] * (max_text_len - len(encoded_text))
    else:
        encoded_text = encoded_text[:max_text_len]
    src_tensor = torch.tensor(encoded_text, dtype=torch.long).unsqueeze(1).to(device)
    initial_decoder_input = torch.zeros((1, 1, output_dim)).to(device)
    generated = speech_to_sign_model(src_tensor, initial_decoder_input)
    return generated.detach().cpu().numpy()

joint_connections = [
    # --------------------
    # Face connections
    # --------------------
    # Left eye
    (1, 2), (2, 3),
    # Right eye
    (4, 5), (5, 6),
    # Bridge nose to eyes
    (1, 0), (0, 4),
    # Left ear -> left eye inner, right ear -> right eye inner
    (7, 1), (8, 4),
    # Mouth
    (9, 10),
    # Optionally connect nose to mouth corners
    (0, 9), (0, 10),

    # --------------------
    # Torso connections
    # --------------------
    # Shoulders & hips
    (11, 12),       # Left Shoulder <-> Right Shoulder
    (11, 23),       # Left Shoulder <-> Left Hip
    (12, 24),       # Right Shoulder <-> Right Hip
    (23, 24),       # Left Hip <-> Right Hip

    # --------------------
    # Arms & Hands
    # --------------------
    # Left arm
    (11, 13),       # Left Shoulder -> Left Elbow
    (13, 15),       # Left Elbow -> Left Wrist
    # Left hand fingers (pinky, index, thumb) branching from wrist
    (15, 17),       # Left Wrist -> Left Pinky
    (15, 19),       # Left Wrist -> Left Index
    (15, 21),       # Left Wrist -> Left Thumb

    # Right arm
    (12, 14),       # Right Shoulder -> Right Elbow
    (14, 16),       # Right Elbow -> Right Wrist
    # Right hand fingers (pinky, index, thumb) branching from wrist
    (16, 18),       # Right Wrist -> Right Pinky
    (16, 20),       # Right Wrist -> Right Index
    (16, 22),       # Right Wrist -> Right Thumb

    # --------------------
    # Legs & Feet
    # --------------------
    # Left leg
    (23, 25),       # Left Hip -> Left Knee
    (25, 27),       # Left Knee -> Left Ankle
    (27, 29),       # Left Ankle -> Left Heel
    (29, 31),       # Left Heel -> Left Foot Index

    # Right leg
    (24, 26),       # Right Hip -> Right Knee
    (26, 28),       # Right Knee -> Right Ankle
    (28, 30),       # Right Ankle -> Right Heel
    (30, 32)        # Right Heel -> Right Foot Index
]

def draw_skeleton(keypoints, image_shape=(480, 640, 3), scale_factor=1.0):
    if keypoints.ndim == 1:
        keypoints = keypoints.reshape(-1, 2)
    
    img = np.zeros(image_shape, dtype=np.uint8)
    h, w = image_shape[:2]
    kp_scaled = keypoints.copy()
    
    kp_min = kp_scaled.min(axis=0)
    kp_max = kp_scaled.max(axis=0)
    kp_range = kp_max - kp_min
    if kp_range.max() > 0:  
        kp_scaled = (kp_scaled - kp_min) / kp_range
    else:
        kp_scaled = np.zeros_like(kp_scaled) + 0.5 


    scale = min(w, h) * 0.8 * scale_factor
    kp_scaled[:, 0] = kp_scaled[:, 0] * scale
    kp_scaled[:, 1] = kp_scaled[:, 1] * scale
    

    offset_x = (w - scale) / 2
    offset_y = (h - scale) / 2
    kp_scaled[:, 0] += offset_x
    kp_scaled[:, 1] += offset_y
    
    kp_scaled = kp_scaled.astype(np.int32)

    for (x, y) in kp_scaled:
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    for (i, j) in joint_connections:
        if i < kp_scaled.shape[0] and j < kp_scaled.shape[0]:
            cv2.line(img, tuple(kp_scaled[i]), tuple(kp_scaled[j]), 
                    color=(255, 0, 0), thickness=2)
    

    
    return img

def generate_video_from_keypoints(keypoints, output_folder="generated", 
                                fps=25, frame_width=640, frame_height=480, 
                                scale_factor=1.0):
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = os.path.join(output_folder, f"generated_sign_video_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, 
                                 (frame_width, frame_height))
    if keypoints.ndim == 3 and keypoints.shape[1] == 1:
        keypoints = keypoints.squeeze(1)
    
    for t in range(keypoints.shape[0]):
        frame_img = draw_skeleton(keypoints[t], 
                                image_shape=(frame_height, frame_width, 3), 
                                scale_factor=scale_factor)
        video_writer.write(frame_img)
    
    video_writer.release()
    return output_video_path

    
# --- routing / WebSocket ---
@app.route('/')
def index():
    return render_template('index.html')

# favicon.ico 
@app.route('/favicon.ico')
def favicon():
    return send_file(os.path.join(app.root_path, 'static', 'favicon.ico'), mimetype='image/vnd.microsoft.icon')

@socketio.on('connect')
def handle_connect():
    environ = request.environ  # 이게 WebSocket 환경
    query_string = environ.get('QUERY_STRING', '')
    from urllib.parse import parse_qs
    query = parse_qs(query_string)

    api_key = query.get('api_key', [None])[0]
    mode = query.get('mode', [None])[0]

    if api_key != API_KEY:
        print("Unauthorized connection attempt")
        return False
    if mode not in ['sign-to-text', 'speech-to-sign']:
        emit('error', {'error': 'Invalid mode. Use \"sign-to-text\" or \"speech-to-sign\"'})
        return False
    print(f"Client connected in {mode} mode")
    emit('message', {'data': f'Connected to server in {mode} mode'})
    environ[request.sid] = {'mode': mode}

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")
    keypoints_buffer.clear()

@socketio.on('frame')
def handle_frame(data):
    environ=request.environ
    mode = environ.get(request.sid, {}).get('mode')
    if mode != 'sign-to-text':
        emit('error', {'error': 'Frame data is only for sign-to-text mode'})
        return
    try:
        frame_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image, results = mediapipe_detection(frame, holistic)
        image = draw_landmarks(image, results)
        keypoints = extract_keypoints(results)
        keypoints_buffer.append(keypoints)

        _, buffer = cv2.imencode('.jpg', image)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        emit('video_frame', {'frame': f"data:image/jpeg;base64,{frame_base64}"}, broadcast=True)

        if len(keypoints_buffer) == 60:
            keypoints_tensor = torch.tensor(np.array(keypoints_buffer), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = sign_to_text_model(keypoints_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                max_prob, pred_idx = torch.max(probs, dim=0)
                predicted_word = index_to_label[pred_idx.item()] if max_prob.item() >= 0.7 else "Unknown"
                top5_probs, top5_indices = torch.topk(probs, 5)
                top5_results = {index_to_label[idx.item()]: prob.item() for idx, prob in zip(top5_indices, top5_probs)}

            emit('prediction', {"predicted_word": predicted_word, "probability": max_prob.item(), "top_5_predictions": top5_results})
            print(f"Prediction: {predicted_word}, Probability: {max_prob.item():.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        emit('error', {'error': str(e)})

@app.route('/handle_audio', methods=['POST'])
def handle_audio():
    mode = request.args.get('mode', 'speech-to-sign')  
    if mode != 'speech-to-sign':
        return jsonify({"error": "This endpoint is for speech-to-sign mode only"}), 400


    if 'file' not in request.files:
        return jsonify({"error": "No file found"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No filename found"}), 400
    audio_path= "recieved audio.wav"
    file.save(audio_path)

    text = speech_to_text(audio_path)
    print("Speech-to-Text conversion results:",text)
    keypoints = generate_keypoints_for_sentence(text)
    video_path = generate_video_from_keypoints(
            keypoints, 
            frame_width=640, 
            frame_height=480, 
            scale_factor=1.5 
        )
    return send_file(video_path, mimetype="video/mp4")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)