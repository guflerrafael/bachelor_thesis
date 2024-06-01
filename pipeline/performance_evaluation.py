import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional

from pipeline.classification.first_layer.SVM import VowelSVMModel
from pipeline.classification.first_layer.LSTM_mfcc import LSTMModel
from pipeline.classification.first_layer.LSTM_wav2vec import PhraseWav2VecModel, DataGenerator
from pipeline.classification.second_layer import VowelMetaModel, PhraseMetaModel
from pipeline.classification.third_layer import FinalMetaModel

class PerformanceEvaluator:
    def __init__(self, vowel_feature_dataset: pd.DataFrame, 
                 phrase_feature_dataset: pd.DataFrame, 
                 vowel_svm_model: VowelSVMModel, 
                 vowel_mfcc_lstm_model: LSTMModel, 
                 phrase_mfcc_lstm_model: LSTMModel, 
                 phrase_wav2vec_lstm_model: PhraseWav2VecModel,
                 vowel_meta_model: Optional[VowelMetaModel] = None,
                 phrase_meta_model: Optional[PhraseMetaModel] = None,
                 final_meta_model: Optional[FinalMetaModel] = None) -> None:
        """
        Initialize the PerformanceEvaluation class.

        Parameters:
        - vowel_feature_dataset (pd.DataFrame): 
        The vowel feature dataset.
        - phrase_feature_dataset (pd.DataFrame): 
        The phrase feature dataset.
        - vowel_svm_model (AudioClassificationSVM): 
        The SVM model for vowel classification.
        - vowel_mfcc_model (AudioClassificationLSTM): 
        The LSTM model for vowel classification.
        - phrase_mfcc_model (AudioClassificationLSTM): 
        The LSTM model for phrase classification.
        - phrase_wav2vec_model (AudioClassificationWav2Vec): 
        The Wav2Vec model for phrase classification.
        - vowel_meta_model (Optional[AudioClassificationMetaModel]): 
        The meta-model for vowel classification.
        - phrase_meta_model (Optional[AudioClassificationMetaModel]): 
        The meta-model for phrase classification.
        - final_meta_model (Optional[AudioClassificationMetaModel]): 
        The final meta-model for classification.
        """
        self.vowel_feature_dataset = vowel_feature_dataset.dropna()
        self.phrase_feature_dataset = phrase_feature_dataset
        self.vowel_svm_model = vowel_svm_model
        self.vowel_mfcc_lstm_model = vowel_mfcc_lstm_model
        self.phrase_mfcc_lstm_model = phrase_mfcc_lstm_model
        self.phrase_wav2vec_lstm_model = phrase_wav2vec_lstm_model
        self.vowel_meta_model = vowel_meta_model
        self.phrase_meta_model = phrase_meta_model
        self.final_meta_model = final_meta_model

    def evaluate_model(self, 
                       y_true: np.ndarray, 
                       y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, 
                                                      float, np.ndarray]:
        """
        Evaluate the model using various performance metrics.

        Parameters:
        - y_true (np.ndarray): The true labels.
        - y_scores (np.ndarray): The predicted scores.

        Returns:
        - Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]: FPR, TPR, AUC, EER, 
        and confusion matrix.
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        y_pred = (y_scores >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(f"ACC: {(accuracy * 100):.2f}, SE: {(sensitivity * 100):.2f}, \
              SP: {(specificity * 100):.2f}, AUC: {(roc_auc * 100):.2f}, EER: {(eer * 100):.2f}")

        return fpr, tpr, roc_auc, eer, confusion_matrix(y_true, y_pred)

    def plot_vowel_phrase_roc(self, 
                                y_true_scores_list_vowels: 
                                List[Tuple[np.ndarray, np.ndarray, str]], 
                                y_true_scores_list_phrases: 
                                List[Tuple[np.ndarray, np.ndarray, str]]
                              ) -> None:
        """
        Plot the combined ROC curves for vowels and phrases.

        Parameters:
        - y_true_scores_list_vowels (List[Tuple[np.ndarray, np.ndarray, str]]): 
        The list of true labels, predicted scores, and labels for vowels.
        - y_true_scores_list_phrases (List[Tuple[np.ndarray, np.ndarray, str]]): 
        The list of true labels, predicted scores, and labels for phrases.
        """
        _, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Plot for vowel ROCs
        for y_true, y_scores, label in y_true_scores_list_vowels:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            axes[0].plot(fpr, tpr, 
                         label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('(a)')
        axes[0].legend(loc="lower right")

        # Plot for phrase ROCs
        for y_true, y_scores, label in y_true_scores_list_phrases:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            axes[1].plot(fpr, tpr, 
                         label=f'{label} AUC = {roc_auc * 100:.2f} %, EER = {eer * 100:.2f} %')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_title('(b)')
        axes[1].legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(os.path.join('images', 'ROC_vowel_phrase.png'))

    def plot_confusion_matrices(self, 
                                confusion_matrices: List[np.ndarray], 
                                titles: List[str]) -> None:
        """
        Plot the confusion matrices. Based on the number of elements, 
        the plots will be arranged accordingly.

        Parameters:
        - confusion_matrices (List[np.ndarray]): The list of confusion matrices.
        - titles (List[str]): The list of titles for the confusion matrices.
        """
        num_elements = len(confusion_matrices)
        if num_elements == 4:
            _, axes = plt.subplots(2, 2, figsize=(10, 10))
            axes = axes.ravel()
        elif num_elements == 2:
            _, axes = plt.subplots(1, 2, figsize=(10, 6))
            axes = axes.ravel()
        elif num_elements == 1:
            _, ax = plt.subplots() 
        else:
            raise ValueError('Invalid number of elements in the confusion_matrices list.')

        for i, (cm, title) in enumerate(zip(confusion_matrices, titles)):
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            if num_elements == 4 or num_elements == 2:
                _ = axes[i].imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
                axes[i].set(xticks=np.arange(cm.shape[1]),
                            yticks=np.arange(cm.shape[0]),
                            xticklabels=['0', '1'], yticklabels=['0', '1'],
                            title=title,
                            ylabel='True label' if i == 0 or i == 2 else None,
                            xlabel='Predicted label' if i == 2 or i == 3 else None)
                fmt = '.2f'
                thresh = cm_percentage.max() / 2.

                for j in range(cm.shape[0]):
                    for k in range(cm.shape[1]):
                        axes[i].text(k, j, format(cm_percentage[j, k], fmt) + '%',
                                     ha="center", va="center",
                                     color="white" if cm_percentage[j, k] > thresh else "black")                             
            elif num_elements == 1:
                cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                _ = ax.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set(xticks=np.arange(cm.shape[1]),
                        yticks=np.arange(cm.shape[0]),
                        xticklabels=['0', '1'], yticklabels=['0', '1'],
                        ylabel='True label',
                        xlabel='Predicted label')
                fmt = '.2f'
                thresh = cm_percentage.max() / 2.

                for j in range(cm.shape[0]):
                    for k in range(cm.shape[1]):
                        ax.text(k, j, format(cm_percentage[j, k], fmt) + '%',
                                ha="center", va="center",
                                color="white" if cm_percentage[j, k] > thresh else "black")
                        
        plt.tight_layout()
        if num_elements == 4:
            plt.savefig(os.path.join('images', 'confusion_matrices_first_layer.png'))
        elif num_elements == 2:
            plt.savefig(os.path.join('images', 'confusion_matrices_second_layer.png'))
        elif num_elements == 1:
            plt.savefig(os.path.join('images', 'confusion_matrices_third_layer.png'))

    def evaluate_and_plot_first_layer(self) -> None:
        """
        Evaluate and plot the first layer of models.
        """
        # Extract test features
        self.vowel_svm_X_test, _, _, self.vowel_svm_y_test = \
            VowelSVMModel.extract_features(self.vowel_feature_dataset)
        self.vowel_mfcc_X_test, _, _, self.vowel_mfcc_y_test = \
            LSTMModel.extract_features(self.vowel_feature_dataset)
        self.phrase_mfcc_X_test, _, _, self.phrase_mfcc_y_test = \
            LSTMModel.extract_features(self.phrase_feature_dataset)
        self.phrase_wav2vec_X_test, _, _, self.phrase_wav2vec_y_test = \
            PhraseWav2VecModel.extract_features(self.phrase_feature_dataset)
        self.wav2vec_test_generator = DataGenerator(self.phrase_wav2vec_X_test, 
                                                    self.phrase_wav2vec_y_test, shuffle=False)

        # Predictions from the first layer models
        self.vowel_svm_predictions = self.vowel_svm_model.decision_function(
            self.vowel_svm_X_test)
        self.vowel_mfcc_lstm_predictions = self.vowel_mfcc_lstm_model.predict(
            self.vowel_mfcc_X_test)[:, 1]
        self.phrase_mfcc_lstm_predictions = self.phrase_mfcc_lstm_model.predict(
            self.phrase_mfcc_X_test)[:, 1]
        self.phrase_wav2vec_lstm_predictions = PhraseWav2VecModel.predict(
            self.phrase_wav2vec_lstm_model, self.wav2vec_test_generator, 
            len(self.wav2vec_test_generator))[:, 1]

        # Plot the ROC curves for first layer models
        vowel_roc_data = [
            (self.vowel_svm_y_test, self.vowel_svm_predictions, 'SVM:'),
            (np.argmax(self.vowel_mfcc_y_test, axis=1), 
             self.vowel_mfcc_lstm_predictions, 'MFCC LSTM:')
        ]
        phrase_roc_data = [
            (np.argmax(self.phrase_mfcc_y_test, axis=1), self.phrase_mfcc_lstm_predictions, 'MFCC LSTM:'),
            (np.argmax(np.array([to_categorical(y, num_classes=2) \
                                 for y in self.phrase_wav2vec_y_test]), axis=1), 
             self.phrase_wav2vec_lstm_predictions, 'Wav2Vec LSTM:')
        ]
        self.plot_vowel_phrase_roc(vowel_roc_data, phrase_roc_data)

        # Evaluate the first layer models
        _, _, _, _, cm_vowel_svm = self.evaluate_model(
            self.vowel_svm_y_test, self.vowel_svm_predictions)
        _, _, _, _, cm_vowel_mfcc = self.evaluate_model(
            np.argmax(self.vowel_mfcc_y_test, axis=1), self.vowel_mfcc_lstm_predictions)
        _, _, _, _, cm_phrase_mfcc = self.evaluate_model(
            np.argmax(self.phrase_mfcc_y_test, axis=1), self.phrase_mfcc_lstm_predictions)
        _, _, _, _, cm_phrase_wav2vec = self.evaluate_model(
            np.argmax(np.array(
                    [to_categorical(y, num_classes=2) for y in self.phrase_wav2vec_y_test]), 
                axis=1), 
            self.phrase_wav2vec_lstm_predictions)
        
        # Plot the confusion matrices
        confusion_matrices = [cm_vowel_svm, cm_vowel_mfcc, cm_phrase_mfcc, cm_phrase_wav2vec]
        titles = ['(a)', '(b)', '(c)', '(d)']
        self.plot_confusion_matrices(confusion_matrices, titles)

    def evaluate_and_plot_second_layer(self, 
                                       vowel_meta_model: Optional[VowelMetaModel], 
                                       phrase_meta_model: Optional[PhraseMetaModel]) -> None:
        """
        Evaluate and plot the second layer of models.

        Parameters:
        - vowel_meta_model (Optional[VowelMetaModel]): The vowel meta model.
        - phrase_meta_model (Optional[PhraseMetaModel]): The phrase meta model.
        """
        if vowel_meta_model is not None and phrase_meta_model is not None:
            self.vowel_meta_model = vowel_meta_model
            self.phrase_meta_model = phrase_meta_model
        if self.vowel_meta_model is None or self.phrase_meta_model is None:
            raise ValueError("Meta models are not provided.")

        # Stack and split vowel predictions
        vowel_mfcc_lstm_predictions = np.argmax(self.vowel_mfcc_lstm_predictions, axis=1)
        vowel_stacked_predictions = np.column_stack((self.vowel_svm_predictions, 
                                                     vowel_mfcc_lstm_predictions))
        _, self.vowel_X_test, _, self.vowel_y_test = train_test_split(
            vowel_stacked_predictions, 
            self.vowel_svm_y_test, 
            test_size=0.2, 
            random_state=42)

        # Stack and split phrase predictions
        phrase_mfcc_lstm_predictions = np.argmax(self.phrase_mfcc_lstm_predictions, axis=1)
        phrase_wav2vec_lstm_predictions = np.argmax(self.phrase_wav2vec_lstm_predictions, axis=1)
        phrase_stacked_predictions = np.column_stack((phrase_mfcc_lstm_predictions, 
                                                      phrase_wav2vec_lstm_predictions))
        _, self.phrase_X_test, _, self.phrase_y_test = train_test_split(
            phrase_stacked_predictions, 
            self.phrase_mfcc_y_test, 
            test_size=0.2, 
            random_state=42)
        self.phrase_y_test = np.argmax(self.phrase_y_test, axis=1)

        # Predictions from the second layer models
        self.vowel_meta_predictions = self.vowel_meta_model.predict_proba(
            self.vowel_X_test)[:, 1]
        self.phrase_meta_predictions = self.phrase_meta_model.predict_proba(
            self.phrase_X_test)[:, 1]

        # Evaluate the second layer models
        _, _, _, _, cm_vowel_meta = self.evaluate_model(
            self.vowel_y_test, self.vowel_meta_predictions)
        _, _, _, _, cm_phrase_meta = self.evaluate_model(
            self.phrase_y_test, self.phrase_meta_predictions)
        
        # Plot the confusion matrices
        confusion_matrices = [cm_vowel_meta, cm_phrase_meta]
        titles = ['(a)', '(b)']
        self.plot_confusion_matrices(confusion_matrices, titles)

    def evaluate_and_plot_third_layer(self,
                                      final_meta_model: Optional[FinalMetaModel]) -> None:
        """
        Evaluate and plot the third layer of models.

        Parameters:
        - final_meta_model (Optional[FinalMetaModel]): The final meta model.
        """
        if final_meta_model is not None:
            self.final_meta_model = final_meta_model
        if self.final_meta_model is None:
            raise ValueError("Final meta model is not provided.")

        # Stack and split second layer predictions
        vowel_meta_predictions = np.array(self.vowel_meta_predictions).reshape(-1, 1)
        phrase_meta_predictions = np.array(self.phrase_meta_predictions).reshape(-1, 1)
        combined_features = np.hstack((vowel_meta_predictions, phrase_meta_predictions))
        phrase_y_test = np.ravel(self.phrase_y_test)
        _, X_test_combined, _, y_test_combined = train_test_split(
            combined_features, phrase_y_test, test_size=0.2, random_state=42)
        
        # Predictions from the final meta-model
        final_predictions = self.final_meta_model.predict_proba(X_test_combined)[:, 1]

        # Evaluate the final meta-model
        _, _, _, _, cm_final_meta = self.evaluate_model(y_test_combined, final_predictions)

        # Plot the confusion matrix
        confusion_matrices = [cm_final_meta]
        titles = ['']
        self.plot_confusion_matrices(confusion_matrices, titles)