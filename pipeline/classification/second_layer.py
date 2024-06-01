import os
import joblib

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class VowelMetaModel:
    def __init__(self):
        pass

    @staticmethod
    def stack_split_predictions(vowel_svm_predictions: np.ndarray,
                                vowel_mfcc_lstm_predictions: np.ndarray,
                                vowel_y_test: np.ndarray) -> tuple[np.ndarray, 
                                                                   np.ndarray, 
                                                                   np.ndarray, 
                                                                   np.ndarray]:
        """
        Stack and split vowel predictions.

        Parameters:
        - vowel_svm_predictions: Predictions from SVM model.
        - vowel_mfcc_lstm_predictions: Predictions from LSTM model using MFCC features.
        - vowel_y_test: True labels for vowel classification.

        Returns:
        - X_train: Training data for meta-model.
        - X_test: Testing data for meta-model.
        - y_train: Training labels for meta-model.
        - y_test: Testing labels for meta-model.
        """
        vowel_mfcc_lstm_predictions = np.argmax(vowel_mfcc_lstm_predictions, axis=1)
        vowel_stacked_predictions = np.column_stack((vowel_svm_predictions, 
                                                     vowel_mfcc_lstm_predictions))

        X_train, X_test, y_train, y_test = train_test_split(vowel_stacked_predictions, 
                                                            vowel_y_test, test_size=0.2, 
                                                            random_state=42)

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_model(vowel_svm_predictions: np.ndarray,
                    vowel_mfcc_lstm_predictions: np.ndarray,
                    vowel_y_test: np.ndarray) -> LogisticRegression:
        """
        Train vowel meta-model on vowel base models predictions.

        Parameters:
        - vowel_svm_predictions: Predictions from SVM model.
        - vowel_mfcc_lstm_predictions: Predictions from LSTM model using MFCC features.
        - vowel_y_test: True labels for vowel classification.

        Returns:
        - vowel_meta_model: Trained meta-model for vowel classification.
        """
        vowel_meta_model = LogisticRegression()
        vowel_X_train, _, vowel_y_train, _ = VowelMetaModel.stack_split_predictions(
            vowel_svm_predictions, vowel_mfcc_lstm_predictions, vowel_y_test)

        vowel_meta_model.fit(vowel_X_train, vowel_y_train)
        # joblib.dump(vowel_meta_model, os.path.join('models', 'vowel_meta_model.joblib'))

        return vowel_meta_model
    
class PhraseMetaModel:
    def __init__(self):
        pass

    @staticmethod
    def stack_split_phrase_predictions(phrase_mfcc_lstm_predictions: np.ndarray,
                                       phrase_wav2vec_lstm_predictions: np.ndarray,
                                       phrase_y_test: np.ndarray) -> tuple[np.ndarray, 
                                                                           np.ndarray, 
                                                                           np.ndarray, 
                                                                           np.ndarray]:
        """
        Stack and split phrase predictions.

        Args:
            phrase_mfcc_lstm_predictions: Predictions from LSTM model using MFCC features.
            phrase_wav2vec_lstm_predictions: Predictions from LSTM model using Wav2Vec features.
            phrase_y_test: True labels for phrase classification.

        Returns:
            X_train: Training data for meta-model.
            X_test: Testing data for meta-model.
            y_train: Training labels for meta-model.
            y_test: Testing labels for meta-model.
        """
        phrase_mfcc_lstm_predictions = np.argmax(phrase_mfcc_lstm_predictions, axis=1)
        phrase_wav2vec_lstm_predictions = np.argmax(phrase_wav2vec_lstm_predictions, axis=1)

        phrase_stacked_predictions = np.column_stack((phrase_mfcc_lstm_predictions, 
                                                      phrase_wav2vec_lstm_predictions))

        X_train, X_test, y_train, y_test = train_test_split(phrase_stacked_predictions, 
                                                            phrase_y_test, test_size=0.2, 
                                                            random_state=42)
        y_train = np.argmax(y_train, axis=1)
        y_test = np.argmax(y_test, axis=1)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_model(phrase_mfcc_lstm_predictions: np.ndarray,
                    phrase_wav2vec_lstm_predictions: np.ndarray,
                    phrase_y_test: np.ndarray) -> LogisticRegression:
        """
        Train phrase meta-model on phrase base models predictions.

        Parameters:
        - phrase_mfcc_lstm_predictions: Predictions from LSTM model using MFCC features.
        - phrase_wav2vec_lstm_predictions: Predictions from LSTM model using Wav2Vec features.
        - phrase_y_test: True labels for phrase classification.

        Returns:
        - phrase_meta_model: Trained meta-model for phrase classification.
        """
        phrase_meta_model = LogisticRegression()
        phrase_X_train, _, phrase_y_train, _ = PhraseMetaModel.stack_split_phrase_predictions(
            phrase_mfcc_lstm_predictions, phrase_wav2vec_lstm_predictions, phrase_y_test)

        phrase_meta_model.fit(phrase_X_train, phrase_y_train)
        joblib.dump(phrase_meta_model, os.path.join('models', 'phrase_meta_model.joblib'))

        return phrase_meta_model