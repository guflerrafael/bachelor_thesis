import os
import joblib

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

class FinalMetaModel:
    def __init__(self):
        pass

    @staticmethod
    def stack_split_predictions(vowel_MM_predictions: np.ndarray, 
                                phrase_MM_predictions: np.ndarray, 
                                y_test: np.ndarray) -> tuple[np.ndarray, 
                                                             np.ndarray, 
                                                             np.ndarray, 
                                                             np.ndarray]:
        """ 
        Stack and split vowel predictions from the vowel and phrase meta models.

        Parameters:
        - vowel_MM_predictions (np.ndarray): Predictions from the vowel meta model.
        - phrase_MM_predictions (np.ndarray): Predictions from the phrase meta model.
        - y_test (np.ndarray): Ground truth labels.

        Returns:
        - tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The train and test sets.
        """
        vowel_MM_predictions = np.array(vowel_MM_predictions).reshape(-1, 1)
        phrase_MM_predictions = np.array(phrase_MM_predictions).reshape(-1, 1)
        combined_features = np.hstack((vowel_MM_predictions, phrase_MM_predictions))
        y_test = np.ravel(y_test)  # Ensures y_test is 1-dimensional if not already

        X_train, X_test, y_train, y_test = train_test_split(combined_features, y_test, 
                                                            test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def train_model(vowel_MM_predictions: np.ndarray, 
                    phrase_MM_predictions: np.ndarray, 
                    y_test: np.ndarray) -> LogisticRegression:
        """
        Train the final meta-model on vowel and phrase meta-models predictions.

        Parameters:
        - vowel_MM_predictions (np.ndarray): Predictions from the vowel meta-model.
        - phrase_MM_predictions (np.ndarray): Predictions from the phrase meta-model.
        - y_test (np.ndarray): Ground truth labels.

        Returns:
        - LogisticRegression: The trained final meta-model.
        """
        final_meta_model = LogisticRegression()
        X_train, _, y_train, _ = FinalMetaModel.stack_split_predictions(vowel_MM_predictions, 
                                                                        phrase_MM_predictions, 
                                                                        y_test)

        final_meta_model.fit(X_train, y_train)
        joblib.dump(final_meta_model, os.path.join('models', 'final_meta_model.joblib'))

        return final_meta_model