import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
from pipeline.classification.SVM import AudioClassificationSVM
from pipeline.classification.LSTM_mfcc import AudioClassificationLSTM
from pipeline.classification.LSTM_wav2vec import AudioClassificationWav2Vec, DataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load feature datasets
vowel_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
phrase_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))

# Drop NaN values to align both models' training and test sets
vowel_feature_dataset = vowel_feature_dataset.dropna()

# Load models
vowel_svm_model = joblib.load(os.path.join('models', 'vowel_svm_model.joblib'))
vowel_mfcc_model = load_model(os.path.join('models', 'vowel_mfcc_lstm_model.keras'))
phrase_mfcc_model = load_model(os.path.join('models', 'phrase_mfcc_lstm_model.keras'))
phrase_wav2vec_model = load_model(os.path.join('models', 'phrase_wav2vec_lstm_model.keras'))

# Extract test set from featuresets
_, vowel_svm_X_test, _, vowel_svm_y_test = AudioClassificationSVM._extract_svm_features(vowel_feature_dataset)
_, vowel_mfcc_X_test, _, vowel_mfcc_y_test = AudioClassificationLSTM._extract_lstm_features(vowel_feature_dataset)
_, phrase_mfcc_X_test, _, phrase_mfcc_y_test = AudioClassificationLSTM._extract_lstm_features(phrase_feature_dataset)
_, phrase_wav2vec_X_test, _, phrase_wav2vec_y_test = AudioClassificationWav2Vec._extract_wav2vec_features(phrase_feature_dataset)
wav2vec_test_generator = DataGenerator(phrase_wav2vec_X_test, phrase_wav2vec_y_test, shuffle=False)

# Predictions
vowel_svm_predictions = vowel_svm_model.predict(vowel_svm_X_test)
vowel_mfcc_lstm_predictions = vowel_mfcc_model.predict(vowel_mfcc_X_test)
phrase_mfcc_lstm_predictions = phrase_mfcc_model.predict(phrase_mfcc_X_test)
phrase_wav2vec_lstm_predictions = AudioClassificationWav2Vec.predict(phrase_wav2vec_model, wav2vec_test_generator, len(wav2vec_test_generator))

## ----------------- VOWEL MODEL --------------------------------
# Assume vowel_mfcc_y_test has the full labels including the dropped rows for SVM
# Convert LSTM probabilities to class labels
vowel_mfcc_lstm_predictions = np.argmax(vowel_mfcc_lstm_predictions, axis=1)

# Now you can stack them since they are aligned
vowel_stacked_predictions = np.column_stack((vowel_svm_predictions, vowel_mfcc_lstm_predictions))

# Split the stacked predictions for training the meta-model
X_train, X_test, y_train, y_test = train_test_split(vowel_stacked_predictions, vowel_svm_y_test, test_size=0.2, random_state=42)

# Train the meta-model
vowel_meta_model = LogisticRegression()
vowel_meta_model.fit(X_train, y_train)

# Save the trained meta-model
joblib.dump(vowel_meta_model, os.path.join('models', 'vowel_meta_model.joblib'))

# Optionally, evaluate the meta-model
vowel_pred = vowel_meta_model.predict(X_test)
print(classification_report(y_test, vowel_pred))
print(vowel_pred)

## --------------- PHRASE MODEL --------------------------------
# Convert LSTM outputs to class labels if they are probabilities
phrase_mfcc_lstm_predictions = np.argmax(phrase_mfcc_lstm_predictions, axis=1)
phrase_wav2vec_lstm_predictions = np.argmax(phrase_wav2vec_lstm_predictions, axis=1)

# Stack the predictions
phrase_stacked_predictions = np.column_stack((phrase_mfcc_lstm_predictions, phrase_wav2vec_lstm_predictions))

# Split the stacked predictions for training the meta-model
X_train, X_test, y_train, y_test = train_test_split(phrase_stacked_predictions, phrase_mfcc_y_test, test_size=0.2, random_state=42)
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

# Train the meta-model
phrase_meta_model = LogisticRegression()
phrase_meta_model.fit(X_train, y_train)

# Save the trained meta-model
joblib.dump(phrase_meta_model, os.path.join('models', 'phrase_meta_model.joblib'))

# Optionally, evaluate the meta-model
phrase_pred = phrase_meta_model.predict(X_test)
print(classification_report(y_test, phrase_pred))
print(phrase_pred)

## ------------------- FINAL MODEL -------------
# Assuming vowel_probs and phrase_probs are arrays containing probability outputs from both models
# The shape of vowel_probs and phrase_probs should be (n_samples, n_classes)

# Ensure vowel_pred and phrase_pred are numpy arrays and reshape if necessary
vowel_pred = np.array(vowel_pred).reshape(-1, 1)
phrase_pred = np.array(phrase_pred).reshape(-1, 1)

# Combine the prediction outputs correctly to form a feature matrix for the final model
combined_features = np.hstack((vowel_pred, phrase_pred))

# Double check lengths and shapes
print("Combined Features Shape:", combined_features.shape)
print("Labels Shape:", y_test.shape)

# Ensure y_test is correctly shaped (1-dimensional)
y_test = np.ravel(y_test)  # This ensures y_test is 1-dimensional if not already

# Split combined features and labels for the final model
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
    combined_features, y_test, test_size=0.2, random_state=42)

# Define and train the logistic regression model
final_meta_model = LogisticRegression()
final_meta_model.fit(X_train_combined, y_train_combined)

# Save the trained final meta-model
joblib.dump(final_meta_model, os.path.join('models', 'final_meta_model.joblib'))

# Optionally, evaluate the final meta-model
final_predictions = final_meta_model.predict(X_test_combined)
print(classification_report(y_test_combined, final_predictions))