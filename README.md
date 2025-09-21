# 🤟 SignScribe - Real-Time ASL Recognition

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SignScribe is a **machine learning-powered application** that recognizes **American Sign Language (ASL)** signs in real-time using a webcam. It can detect the alphabet (A-Z) and special gestures like **"Thank you"** and **"I Love You"**, providing a bridge between sign language users and technology.

---

## 🌟 Features

- ✅ Real-time ASL alphabet detection (A-Z)  
- ✅ Special gesture detection: "Thank you" & "I Love You"  
- ✅ Visual feedback with live webcam feed  
- ✅ Gesture history display  
- ✅ Stable predictions using motion tracking and history  
- ✅ Built with **Streamlit** for a simple, interactive interface  

---

## 🛠 How It Works

### 1. Data Collection
- `train_model.py` collects hand landmarks using **MediaPipe**.  
- Users press keys (A-Z) to record gesture samples for the model.  
- Data is saved in `sign_data.csv` for training.

### 2. Model Training
- Machine learning model reads CSV data and learns to predict signs.  
- The trained model is saved as `trained_asl_model.pkl` for real-time detection.

### 3. Real-Time Detection
- `detect_signs.py` processes live webcam feed using **MediaPipe Hands & FaceMesh**.  
- Hand landmarks are passed through the ML model to predict signs.  
- Special motion logic detects:
  - **"J"**: Pinky curves in motion  
  - **"Thank you"**: Hand touches chin and moves downward  
  - **"I Love You"**: Predefined static hand symbol  
- Detected gestures are displayed on-screen along with recent history.

---
## 🎬 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Zaku-Seven/ASL_RECOGNITION.git
cd ASL_RECOGNITION
```

2. **Create a virtual environment and activate it:**
```bash
# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python3 -m venv .venv
.venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Ensure `trained_asl_model.pkl` exists** in the project directory, or generate it by running:
```bash
python train_model.py
```

---

## 🚀 Usage

### 1. Collect Training Data (Optional)
```bash
python train_model.py
```
- Position your hand inside the camera frame.  
- Press **A-Z** to record gesture samples.  
- Press **Q** to quit.

### 2. Run Real-Time Detection
```bash
python detect_signs.py
```
- Position your hand in the green detection box.  
- Make clear ASL letters for recognition.  

**Special Gestures:**
- **J:** Make "I" then curve your pinky down  
- **Thank You:** Touch chin and move downward  
- **I Love You:** Predefined static hand symbol  

The camera feed will show detected signs along with a gesture history panel.

---

## 🗂 Project Structure
```
ASL_RECOGNITION/
│
├─ detect_signs.py        # Real-time detection & Streamlit interface
├─ train_model.py         # Train the ASL detection model
├─ sign_data.csv          # Collected hand landmark samples
├─ trained_asl_model.pkl  # Trained ML model
├─ requirements.txt       # Python dependencies
└─ README.md              # Project documentation
```

---

## ⚙️ Dependencies
- Python 3.11+
- OpenCV (cv2)
- MediaPipe
- NumPy
- Streamlit
- scikit-learn
- Pickle



ChatGPT and Claude.ai were briefly used to aid in machine learning

