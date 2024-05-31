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

# Load feature datasets
vowel_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
phrase_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))

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

def print_metrics(y_true, predictions, model_name):
    print(f"{model_name} Classification Report")
    print(classification_report(y_true, predictions))
    print(f"{model_name} Confusion Matrix")
    cm = confusion_matrix(y_true, predictions)
    print(cm)
    return cm

def plot_confusion_matrix(cm, model_name, ax, classes):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'Confusion Matrix for {model_name}',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylim(len(cm)-0.5, -0.5)

def plot_all_confusion_matrices(cm_list, model_names, class_names):
    fig, axes = plt.subplots(nrows=1, ncols=len(cm_list), figsize=(15, 5))
    for ax, cm, name in zip(axes, cm_list, model_names):
        plot_confusion_matrix(cm, name, ax, class_names)
    plt.tight_layout()
    plt.show()

# Assume y_true for each prediction is available
cms = []
models = ['Vowel SVM', 'Vowel MFCC LSTM', 'Phrase MFCC LSTM', 'Phrase Wav2Vec LSTM']
predictions = [vowel_svm_predictions, np.argmax(vowel_mfcc_lstm_predictions, axis=1), np.argmax(phrase_mfcc_lstm_predictions, axis=1), np.argmax(phrase_wav2vec_lstm_predictions, axis=1)]
y_trues = [vowel_svm_y_test, np.argmax(vowel_mfcc_y_test, axis=1), np.argmax(phrase_mfcc_y_test, axis=1), np.argmax(np.array([to_categorical(y, num_classes=2) for y in phrase_wav2vec_y_test]), axis=1)]

# Evaluate and print metrics
for y_true, pred, model in zip(y_trues, predictions, models):
    cm = print_metrics(y_true, pred, model)
    cms.append(cm)

# Assuming all datasets have the same classes, if not adjust accordingly
class_names = ['Healthy', 'Pathological']  # adjust as needed

# Plot confusion matrices
plot_all_confusion_matrices(cms, models, class_names)

def calculate_errors(y_true, predictions):
    return y_true != predictions

def calculate_error_correlation(errors_dict):
    keys = list(errors_dict.keys())
    correlations = {}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            key_i, key_j = keys[i], keys[j]
            correlation = np.corrcoef(errors_dict[key_i], errors_dict[key_j])[0, 1]
            correlations[(key_i, key_j)] = correlation
    return correlations

errors_dict = {}

# Calculate errors for each model
for y_true, pred, model in zip(y_trues, predictions, models):
    errors = calculate_errors(y_true, np.round(pred).astype(int))  # Ensure predictions are converted to class labels if necessary
    errors_dict[model] = errors

# Calculate correlations between model errors
# error_correlations = calculate_error_correlation(errors_dict)

# Display correlations
# for pair, corr in error_correlations.items():
    # print(f"Error correlation between {pair[0]} and {pair[1]}: {corr:.2f}")