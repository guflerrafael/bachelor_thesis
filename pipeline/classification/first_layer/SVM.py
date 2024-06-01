import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class VowelSVMModel():
    def __init__(self):
        pass

    def _balance_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by upsampling the minority class.

        Parameters:
        - dataset (pd.DataFrame): The dataset to balance.

        Returns:
        - pd.DataFrame: The balanced dataset.
        """
        class_0 = dataset[dataset['diagnosis'] == 0]
        class_1 = dataset[dataset['diagnosis'] == 1]
        
        if len(class_0) > len(class_1):
            class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), 
                                         random_state=42)
            balanced_dataset = pd.concat([class_0, class_1_upsampled])
        else:
            class_0_upsampled = resample(class_0, replace=True, n_samples=len(class_1), 
                                         random_state=42)
            balanced_dataset = pd.concat([class_0_upsampled, class_1])
        
        return balanced_dataset

    def extract_features(dataset: pd.DataFrame, 
                         test_size: float = 0.2, 
                         random_state: int = 42) -> tuple:
        """
        Extract features from the dataset and split it into training and testing sets.

        Parameters:
        - dataset (pd.DataFrame): The dataset to extract features from.
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - random_state (int, optional): The seed used by the random number generator. Defaults to 42.

        Returns:
        - tuple: A tuple containing the training and testing features and labels.
        """
        dataset_clean = dataset.dropna(subset=dataset.columns.difference(['mfcc', 'diagnosis']))
        X = dataset_clean.drop(columns=['mfcc', 'diagnosis'])
        y = dataset_clean['diagnosis']  
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        k = min(10, X.shape[1])  # Select up to 10 features or the number of features available
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        X = pd.DataFrame(X_selected)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def train_model(dataset: pd.DataFrame, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> SVC:
        """
        Train the vowel SVM model on the dataset.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the model on.
        - test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        - random_state (int, optional): The seed used by the random number generator. Defaults to 42.

        Returns:
        - SVC: The trained Support Vector Machine model.
        """
        dataset = VowelSVMModel._balance_dataset(dataset)
        X_train, _, y_train, _ = VowelSVMModel.extract_features(dataset, test_size=test_size, 
                                                                random_state=random_state)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        return best_model

    @staticmethod
    def predict(model: SVC, features: pd.DataFrame) -> np.ndarray:
        """
        Predict the labels for the given features using the SVM model.

        Parameters:
        - model (SVC): The trained Support Vector Machine model.
        - features (pd.DataFrame): The features to make predictions on.

        Returns:
        - np.ndarray: The predicted labels.
        """
        if isinstance(features, pd.DataFrame):
            return model.predict(features)
        else:
            raise ValueError("Input features should be a pandas DataFrame.")