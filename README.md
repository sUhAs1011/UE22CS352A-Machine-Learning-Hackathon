# 🎵 Audio Sentiment Analysis – ML Hackathon

### Leadboard Position
<img width="1571" height="229" alt="image" src="https://github.com/user-attachments/assets/a4ff7f1e-0e61-45a3-b1e3-299347e70222" />

### DataSet
https://drive.google.com/drive/folders/14Oo-73toqu1mImA3YQEGMSQN8bXyNT-w



### 📌 Project Overview

This project was developed as part of the ML Hackathon – Set 1: Sentiment Analysis challenge.
The task was to build a sentiment classification model that detects emotions from audio clips of a TV show dataset. The model classifies each clip into one of three categories:

😊 Positive

😐 Neutral

😞 Negative

Although the original problem statement involved multimodal fusion (video + text), this solution focuses on audio-based sentiment detection.


### ⚙️ How It Works

- Dataset Loading – Reads training labels (train.csv) and test file metadata.

- Feature Extraction – Extracts MFCC (Mel Frequency Cepstral Coefficients) features from each audio file using librosa.

- Preprocessing:
   Standardizes features using StandardScaler
   Applies PCA (Principal Component Analysis) for dimensionality reduction

- Model Training – Uses a Random Forest Classifier to learn sentiment patterns.

- Prediction – Classifies test audio clips and outputs results to audio_sentiment_predictions.csv.
