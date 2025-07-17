# Sign-to-Text Inference Server (EyEar FYP)

This is a **real-time sign-to-text translation server** built for the final year project: *EyEar – Bi-directional Sign Language Translational Wearable Device with Edge AI*.

It receives raspberry Pi camera frames through WebSocket, extracts keypoints using **MediaPipe**, and performs classification using a **Transformer-based model** trained on the **WLASL dataset**.

---

## Features

-  Real-time landmark detection using **MediaPipe Holistic**
-  Transformer classifier with 60-frame window (keypoint dim=1662)
-  WebSocket-based frame communication with Flask-SocketIO
-  API key access control
-  Top-5 prediction confidence returned live
-  Excluded speech-to-sign details

---

##  Files & Structure

```
server/
├── server.py                  # Flask-SocketIO server logic (sign-to-text)
├── best_transformer_final.pt  # Trained Transformer weights
├── metadata.pkl               # Label encoder + config info
├── corpus.csv                 # Sentence reference corpus
├── templates/index.html       # Frontend (stream + display)
├── static/favicon.ico         # Favicon (optional)
```

---

##  Running the Server (Sign-to-Text)

1. Download model assets:

   - `best_transformer_final.pt`: Transformer weights
   - `metadata.pkl`: Includes `label_encoder` and number of classes
   - `corpus.csv`: Used for reverse look-up and token matching

2. Install requirements:

```bash
pip install flask flask-socketio torchaudio opencv-python mediapipe transformers soundfile
```

3. Run:

```bash
python server.py
```

4. Visit the frontend:

```
http://localhost:5000/?api_key=1234&mode=sign-to-text
```

> Note: Change `API_KEY` in `server.py` before deployment.

---

##  Model Summary

- Transformer-based classifier for gesture sequences
- **Input:** 60 frames × 1662-dim keypoint vectors
- **Architecture:** Embedding → 6-layer Transformer → Pooling → Linear
- **Test Accuracy:** 85.59%
- **Latency:** Wearable upload delay 2.77s, retrieve delay 0.3127s

