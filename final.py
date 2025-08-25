import pandas as pd
import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Load labels from CSV
def load_labels(csv_path):
    return pd.read_csv(csv_path, encoding='ISO-8859-1')

# Feature extraction from audio files
def extract_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# Train the RandomForest model for audio sentiment classification
def train_audio_model(audio_folder, labels_df, scaler, pca, sentiment_map):
    audio_features = []
    audio_labels = []

    # Extract features from each audio file
    for index, row in labels_df.iterrows():
        file_path = os.path.join(audio_folder, row['file_name'])
        if os.path.exists(file_path):
            features = extract_audio_features(file_path)
            audio_features.append(features)
            audio_labels.append(sentiment_map[row['sentiment']])

    audio_features = np.array(audio_features)
    audio_labels = np.array(audio_labels)

    # Scale and apply PCA
    audio_features = scaler.fit_transform(audio_features)
    audio_features = pca.fit_transform(audio_features)

    # Train RandomForest model
    model = RandomForestClassifier()
    model.fit(audio_features, audio_labels)
    
    return model

# Predict sentiment for test audio files
def predict_audio_sentiment(audio_folder, labels_df, model, scaler, pca, sentiment_map_reverse):
    audio_features = []
    ids = []

    # Extract features from each audio file in the test set
    for index, row in labels_df.iterrows():
        file_path = os.path.join(audio_folder, row['file_name'])
        if os.path.exists(file_path):
            features = extract_audio_features(file_path)
            audio_features.append(features)
            ids.append(index + 1)  # ID starts from 1 instead of 0

    audio_features = np.array(audio_features)
    
    # Scale and apply PCA to the test features
    audio_features = scaler.transform(audio_features)
    audio_features = pca.transform(audio_features)

    # Predict sentiment
    predictions = model.predict(audio_features)

    # Map predictions back to sentiment labels
    predicted_sentiments = [sentiment_map_reverse[pred] for pred in predictions]
    
    return ids, predicted_sentiments

# Main function to train model and output results to CSV
def main():
    train_csv_path = "M:/ML Hackathon/ml-hackathon-ec-campus-set-1/train.csv"
    audio_folder_train = "M:/ML Hackathon/audio"
    test_csv_path = "M:/ML Hackathon/ml-hackathon-ec-campus-set-1/test"
    
    # Load labels
    labels_df_train = load_labels(train_csv_path)
    
    # Create sentiment map for training and reverse map for prediction
    sentiment_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    sentiment_map_reverse = {0: 'positive', 1: 'neutral', 2: 'negative'}
    
    # StandardScaler and PCA for feature scaling and dimensionality reduction
    scaler = StandardScaler()
    pca = PCA(n_components=20)
    
    # Train the audio model
    audio_model = train_audio_model(audio_folder_train, labels_df_train, scaler, pca, sentiment_map)
    
    # Predict sentiment on test data
    ids, predicted_sentiments = predict_audio_sentiment(audio_folder_train, labels_df_train, audio_model, scaler, pca, sentiment_map_reverse)
    
    # Create a DataFrame for output
    output_df = pd.DataFrame({
        'ID': ids,
        'Sentiment': predicted_sentiments
    })
    
    # Save output to a CSV file with ISO-8859-1 encoding
    output_df.to_csv('audio_sentiment_predictions.csv', index=False, encoding='ISO-8859-1')
    print("Predictions saved to 'audio_sentiment_predictions.csv'.")

if __name__ == "__main__":
    main()
