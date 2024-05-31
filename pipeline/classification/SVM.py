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

class AudioClassificationSVM():
    def __init__(self):
        pass

    def _check_missing_values(dataset):
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)

    def _balance_dataset(dataset):
        class_0 = dataset[dataset['diagnosis'] == 0]
        class_1 = dataset[dataset['diagnosis'] == 1]
        
        if len(class_0) > len(class_1):
            class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
            balanced_dataset = pd.concat([class_0, class_1_upsampled])
        else:
            class_0_upsampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
            balanced_dataset = pd.concat([class_0_upsampled, class_1])
        
        return balanced_dataset

    def _extract_svm_features(dataset, test_size=0.2, random_state=42):
        dataset_clean = dataset.dropna(subset=dataset.columns.difference(['mfcc', 'diagnosis']))
        X = dataset_clean.drop(columns=['mfcc', 'diagnosis'])
        y = dataset_clean['diagnosis']  
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        k = min(10, X.shape[1])  # Select up to 10 features or the number of features available
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        X = pd.DataFrame(X_selected)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def train_svm_model(dataset, test_size=0.2, random_state=42,  visualize_performance=True):
        AudioClassificationSVM._check_missing_values(dataset)
        dataset = AudioClassificationSVM._balance_dataset(dataset)
        X_train, X_test, y_train, y_test = AudioClassificationSVM._extract_svm_features(dataset, test_size=test_size, random_state=random_state)

        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
        grid.fit(X_train, y_train)
        
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        if visualize_performance:
            AudioClassificationSVM.visualize_performance(y_test, y_pred)

        return best_model

    @staticmethod
    def predict(model, features):
        if isinstance(features, pd.DataFrame):
            return model.predict(features)
        else:
            raise ValueError("Input features should be a pandas DataFrame.")
        
    @staticmethod
    def visualize_performance(y_test, y_pred):
        """
        Visualize the performance of the LSTM model using a confusion matrix and classification report.

        Parameters:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        """
        # Accuracy                
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join('images', 'vowel_svm_model_confusion.png'))

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
        print("Classification Report:")
        print(report)