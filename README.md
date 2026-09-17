# Communication Support System for the Deaf and Hard of Hearing Community

A web-based communication support system designed to facilitate communication between Deaf or hard-of-hearing users and hearing people.

The system combines computer vision, machine learning, natural language processing, and speech technologies to provide three main communication functions:

- Sign-to-Text
- Speech-to-Text
- Text-to-Speech

The system also provides an Experience Feedback function for collecting user ratings and storing feedback data in Amazon DynamoDB.

---

## 1. Project Overview

Communication barriers can make interaction difficult between Deaf or hard-of-hearing people and hearing people, especially when the participants do not share the same communication method.

This project aims to provide a web-based communication support system that can:

1. Recognize a predefined set of Vietnamese sign language signs.
2. Convert recognized signs into Vietnamese text.
3. Convert Vietnamese speech into text.
4. Convert Vietnamese text into speech.
5. Collect user experience feedback.
6. Store feedback data in Amazon DynamoDB.

The sign recognition component uses MediaPipe to extract hand, pose, and facial features. A Random Forest model is then used to classify the extracted feature sequences.

---

## 2. Main Features

### 2.1 Sign-to-Text

The system uses a webcam to capture the user's signing movements.

The processing pipeline is:

Camera
→ MediaPipe
→ Feature Extraction
→ Feature Sequence
→ Random Forest
→ Sign Label
→ Rule-based NLP
→ Vietnamese Text

The current recognition model supports a predefined set of 29 Vietnamese sign labels.

Each input sample contains:

- 30 frames
- 163 features per frame
- 4,890 feature values per sample

Calculation:

30 × 163 = 4,890

The system does not attempt to recognize the entire Vietnamese Sign Language. It performs classification within the predefined sign label set used in the project.

---

### 2.2 Speech-to-Text

The Speech-to-Text function captures Vietnamese speech through the microphone and converts the recognized speech into text.

The function is intended to support communication between hearing users and Deaf or hard-of-hearing users.

The system uses the Web Speech API provided by the browser.

---

### 2.3 Text-to-Speech

The Text-to-Speech function converts Vietnamese text into spoken audio.

Users can enter text into the interface and use the browser's speech synthesis capability to play the corresponding audio.

---

### 2.4 Experience Feedback

The system provides a feedback form containing five rating questions.

Users can select a rating from 1 to 5 for each question and submit their evaluation.

Feedback data is sent to the FastAPI backend and stored in Amazon DynamoDB.

The system can also retrieve feedback statistics, including:

- Total number of responses
- Overall average rating
- Average rating for each question

---

## 3. System Architecture

The system consists of a React/Vite frontend and a FastAPI backend.

### Frontend

The frontend provides:

- User interface
- Camera access
- MediaPipe processing
- Sign recognition interaction
- Speech-to-Text
- Text-to-Speech
- Experience Feedback

### Backend

The FastAPI backend provides:

- Sign recognition API
- Experience Feedback API
- Feedback summary API
- Random Forest model inference
- DynamoDB integration

### Database

Amazon DynamoDB is used to store experience feedback data.

The main DynamoDB table is:

`deaf-usability-feedback`

---

## 4. Technologies Used

| Component | Technology |
|---|---|
| Frontend | React |
| Build Tool | Vite |
| Backend | FastAPI |
| Programming Language | Python |
| Computer Vision | MediaPipe |
| Machine Learning | Random Forest |
| Natural Language Processing | Rule-based NLP |
| Speech Recognition | Web Speech API |
| Text-to-Speech | Web Speech API |
| Database | Amazon DynamoDB |
| API Documentation | FastAPI Swagger UI |
| Source Code | GitHub |
| Frontend Deployment | Vercel |
| Backend Deployment | Render |

---

## 5. Machine Learning Model

The sign recognition model uses a Random Forest classifier.

### Dataset

Current dataset statistics:

- Total samples: 146
- Number of sign labels: 29
- Training samples: 117
- Testing samples: 29

The dataset is divided into training and testing sets using a group-aware split based on source videos.

### Feature Representation

Each frame contains 163 extracted features.

A sequence contains 30 frames.

Therefore:

`30 × 163 = 4,890`

feature values are used for each sample.

### Model

Model type:

`Random Forest`

Model file:

`sign_holistic_model.joblib`

Model location:

`backend/ai/sign_recognition/models/sign_holistic_model.joblib`

The trained model is loaded by the backend for sign prediction.

---

## 6. Sign Label Set

The current system supports 29 predefined sign labels:

```text
an
ban
ban_1
biet
buon
cam_on
can
cho_1
co
cong_nghe
day
den
di
giup_do
hoc
khong
khong_biet
khong_cho
muon
nha
sieu_thi
thich
toi
truong
uong
ve
vui
xin_chao
yeu
```

## 7. Project Structure

```text
deaf-communication-system/
│
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   │   ├── sign.py
│   │   │   └── feedback.py
│   │   │
│   │   ├── schemas/
│   │   ├── services/
│   │   └── __init__.py
│   │
│   ├── ai/
│   │   └── sign_recognition/
│   │       └── models/
│   │           └── sign_holistic_model.joblib
│   │
│   ├── data/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── services/
│   │   ├── utils/
│   │   └── ...
│   │
│   ├── package.json
│   ├── package-lock.json
│   └── index.html
│
├── .gitignore
└── README.md
```

## 8. Local Development
### 8.1 Requirements

Before running the project locally, install:

Python 3.9 or compatible Python version
Node.js
npm
A modern web browser
Webcam
Microphone

### 8.2 Run Backend

Open a terminal and navigate to the backend directory:

```bash
cd backend
```

Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI server:

uvicorn app.main:app --host 127.0.0.1 --port 8000

The backend will be available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### 8.3 Run Frontend

Open another terminal:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the Vite development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```
