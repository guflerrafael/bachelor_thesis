import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from pipeline.classification.SVM import AudioClassificationSVM
from pipeline.classification.LSTM_mfcc import AudioClassificationLSTM
from pipeline.classification.LSTM_wav2vec import AudioClassificationWav2Vec, DataGenerator

# Load feature datasets
vowel_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
phrase_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))
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
vowel_svm_predictions = vowel_svm_model.decision_function(vowel_svm_X_test)
vowel_mfcc_lstm_predictions = vowel_mfcc_model.predict(vowel_mfcc_X_test)[:, 1]
phrase_mfcc_lstm_predictions = phrase_mfcc_model.predict(phrase_mfcc_X_test)[:, 1]
phrase_wav2vec_lstm_predictions = AudioClassificationWav2Vec.predict(phrase_wav2vec_model, wav2vec_test_generator, len(wav2vec_test_generator))[:, 1]

def plot_combined_roc(y_true_scores_list_vowels, y_true_scores_list_phrases):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for vowel ROCs
    for y_true, y_scores, label in y_true_scores_list_vowels:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        axes[0].plot(fpr, tpr, label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('(a)')
    axes[0].legend(loc="lower right")

    # Plot for phrase ROCs
    for y_true, y_scores, label in y_true_scores_list_phrases:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        axes[1].plot(fpr, tpr, label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
    axes[1].set_xlabel('False Positive Rate')
    # axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('(b)')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join('images', 'ROC_vowel_phrase.png'))

def evaluate_model(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    y_pred = (y_scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"ACC: {(accuracy * 100):.2f}, SE: {(sensitivity * 100):.2f}, SP: {(specificity * 100):.2f}, AUC: {(roc_auc * 100):.2f}, EER: {(eer * 100):.2f}")

    return fpr, tpr, roc_auc, eer, confusion_matrix(y_true, y_pred)

def plot_confusion_matrices(confusion_matrices, titles):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()

    for i, (cm, title) in enumerate(zip(confusion_matrices, titles)):
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        im = axes[i].imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set(xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=['0', '1'], yticklabels=['0', '1'],
                    title=title,
                    ylabel='True label' if i == 0 or i == 2 else None,
                    xlabel='Predicted label' if i == 2 or i == 3 else None)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm_percentage.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, format(cm_percentage[j, k], fmt) + '%',
                             ha="center", va="center",
                             color="white" if cm_percentage[j, k] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'confusion_matrices.png'))

## ------------------------------------ FIRST LAYER ---------------------------------------------------------------

# Evaluate models and collect ROC data
vowel_roc_data = [
    (vowel_svm_y_test, vowel_svm_predictions, 'SVM:'),
    (np.argmax(vowel_mfcc_y_test, axis=1), vowel_mfcc_lstm_predictions, 'MFCC LSTM:')
]
phrase_roc_data = [
    (np.argmax(phrase_mfcc_y_test, axis=1), phrase_mfcc_lstm_predictions, 'MFCC LSTM:'),
    (np.argmax(np.array([to_categorical(y, num_classes=2) for y in phrase_wav2vec_y_test]), axis=1), phrase_wav2vec_lstm_predictions, 'Wav2Vec LSTM:')
]

# Plot combined ROC curves in subplots
plot_combined_roc(vowel_roc_data, phrase_roc_data)

# Evaluate models and collect confusion matrices
_, _, _, _, cm_vowel_svm = evaluate_model(vowel_svm_y_test, vowel_svm_predictions)
_, _, _, _, cm_vowel_mfcc = evaluate_model(np.argmax(vowel_mfcc_y_test, axis=1), vowel_mfcc_lstm_predictions)
_, _, _, _, cm_phrase_mfcc = evaluate_model(np.argmax(phrase_mfcc_y_test, axis=1), phrase_mfcc_lstm_predictions)
_, _, _, _, cm_phrase_wav2vec = evaluate_model(np.argmax(np.array([to_categorical(y, num_classes=2) for y in phrase_wav2vec_y_test]), axis=1), phrase_wav2vec_lstm_predictions)

# Plot confusion matrices
confusion_matrices = [cm_vowel_svm, cm_vowel_mfcc, cm_phrase_mfcc, cm_phrase_wav2vec]
titles = ['(a)', '(b)', '(c)', '(d)']
plot_confusion_matrices(confusion_matrices, titles)

## ------------------------------------ SECOND LAYER ---------------------------------------------------------------

def plot_confusion_matrices(confusion_matrices, titles):
    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    axes = axes.ravel()

    for i, (cm, title) in enumerate(zip(confusion_matrices, titles)):
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        im = axes[i].imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
        axes[i].set(xticks=np.arange(cm.shape[1]),
                    yticks=np.arange(cm.shape[0]),
                    xticklabels=['0', '1'], yticklabels=['0', '1'],
                    title=title,
                    ylabel='True label' if i == 0 else None,
                    xlabel='Predicted label' if i == 0 or i == 1 else None)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f'
        thresh = cm_percentage.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                axes[i].text(k, j, format(cm_percentage[j, k], fmt) + '%',
                             ha="center", va="center",
                             color="white" if cm_percentage[j, k] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'confusion_matrices_second_layer.png'))

def plot_roc(y_true_scores_list):
    fig, ax = plt.subplots()  # Changed from axes to ax for clarity

    # Plot for ROCs
    for y_true, y_scores, label in y_true_scores_list:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        ax.plot(fpr, tpr, label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join('images', 'ROC_second_layer.png'))

# Assuming we have the meta models loaded and their predictions ready
# Load meta models
vowel_meta_model = joblib.load(os.path.join('models', 'vowel_meta_model.joblib'))
phrase_meta_model = joblib.load(os.path.join('models', 'phrase_meta_model.joblib'))

vowel_svm_predictions = vowel_svm_model.predict(vowel_svm_X_test)
vowel_mfcc_lstm_predictions = vowel_mfcc_model.predict(vowel_mfcc_X_test)
phrase_mfcc_lstm_predictions = phrase_mfcc_model.predict(phrase_mfcc_X_test)
phrase_wav2vec_lstm_predictions = AudioClassificationWav2Vec.predict(phrase_wav2vec_model, wav2vec_test_generator, len(wav2vec_test_generator))

# Get the vowel predictions
vowel_mfcc_lstm_predictions = np.argmax(vowel_mfcc_lstm_predictions, axis=1)
vowel_stacked_predictions = np.column_stack((vowel_svm_predictions, vowel_mfcc_lstm_predictions))
_, vowel_X_test, _, vowel_y_test = train_test_split(vowel_stacked_predictions, vowel_svm_y_test, test_size=0.2, random_state=42)

# Get the phrase predictions
phrase_mfcc_lstm_predictions = np.argmax(phrase_mfcc_lstm_predictions, axis=1)
phrase_wav2vec_lstm_predictions = np.argmax(phrase_wav2vec_lstm_predictions, axis=1)
phrase_stacked_predictions = np.column_stack((phrase_mfcc_lstm_predictions, phrase_wav2vec_lstm_predictions))
_, phrase_X_test, _, phrase_y_test = train_test_split(phrase_stacked_predictions, phrase_mfcc_y_test, test_size=0.2, random_state=42)
phrase_y_test = np.argmax(phrase_y_test, axis=1)

# Predict using meta models (assuming features are prepared as combined_features)
vowel_meta_predictions = vowel_meta_model.predict_proba(vowel_X_test)[:, 1]
phrase_meta_predictions = phrase_meta_model.predict_proba(phrase_X_test)[:, 1]

# Function to evaluate and plot meta model performances
def evaluate_and_plot_meta_models():
    # Evaluate models and collect ROC data for meta models
    first_meta_roc_data = [
        (vowel_y_test, vowel_meta_predictions, 'Vowel Meta-Model:'),
        (phrase_y_test, phrase_meta_predictions, 'Phrase Meta-Model:')
    ]

    print(vowel_meta_predictions)
    print(phrase_meta_predictions)

    # Plot combined ROC curves in subplots for meta models
    plot_roc(first_meta_roc_data)

    # Evaluate models and collect confusion matrices for meta models
    _, _, _, _, cm_vowel_meta = evaluate_model(vowel_y_test, vowel_meta_predictions)
    _, _, _, _, cm_phrase_meta = evaluate_model(phrase_y_test, phrase_meta_predictions)

    # Plot confusion matrices for meta models
    confusion_matrices = [cm_vowel_meta, cm_phrase_meta]
    titles = ['(a)', '(b)']
    plot_confusion_matrices(confusion_matrices, titles)

# Call the function to perform evaluation and plotting
evaluate_and_plot_meta_models()

## ------------------------------------ THIRD LAYER ---------------------------------------------------------------

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()  # Single subplot; using ax instead of axes

    cm_percentage = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis] * 100
    im = ax.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
            yticks=np.arange(confusion_matrix.shape[0]),
            xticklabels=['0', '1'], yticklabels=['0', '1'],
            ylabel='True label',
            xlabel='Predicted label')

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm_percentage.max() / 2.
    for j in range(confusion_matrix.shape[0]):
        for k in range(confusion_matrix.shape[1]):
            ax.text(k, j, format(cm_percentage[j, k], fmt) + '%',
                    ha="center", va="center",
                    color="white" if cm_percentage[j, k] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join('images', 'confusion_matrices_third_layer.png'))

def plot_roc(y_true_scores_list):
    fig, ax = plt.subplots()  # Changed from axes to ax for clarity

    # Plot for ROCs
    for y_true, y_scores, label in y_true_scores_list:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        ax.plot(fpr, tpr, label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join('images', 'ROC_third_layer.png'))

final_meta_model = joblib.load(os.path.join('models', 'final_meta_model.joblib'))

vowel_pred = np.array(vowel_meta_predictions).reshape(-1, 1)
phrase_pred = np.array(phrase_meta_predictions).reshape(-1, 1)

# Combine the prediction outputs correctly to form a feature matrix for the final model
combined_features = np.hstack((vowel_pred, phrase_pred))

# Ensure y_test is correctly shaped (1-dimensional)
phrase_y_test = np.ravel(phrase_y_test)  # This ensures y_test is 1-dimensional if not already

# Split combined features and labels for the final model
_, X_test_combined, _, y_test_combined = train_test_split(
    combined_features, phrase_y_test, test_size=0.2, random_state=42)

final_predictions = final_meta_model.predict_proba(X_test_combined)[:, 1]

# Function to evaluate and plot meta model performances
def evaluate_and_plot_final_meta_model():
    # Evaluate models and collect ROC data for meta models
    second_meta_roc_data = [
        (y_test_combined, final_predictions, 'Final Meta-Model:'),
    ]

    print(final_predictions)

    # Plot combined ROC curves in subplots for meta models
    plot_roc(second_meta_roc_data)
    
    # Evaluate models and collect confusion matrices for meta models
    _, _, _, _, cm_final_meta = evaluate_model(y_test_combined, final_predictions)

    # Plot confusion matrices for meta models
    plot_confusion_matrix(cm_final_meta)

evaluate_and_plot_final_meta_model()