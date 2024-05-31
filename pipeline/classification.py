import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AudioClassification():
    def __init__(self):
        pass

    def _check_missing_values(dataset):
        """
        Check for missing values in the dataset and print the count of missing values for each column.

        Parameters:
        - dataset (pd.DataFrame): The dataset to check for missing values.
        """
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)

    def _extract_svm_features(dataset):
        """
        Extract features and labels for SVM classification and standardize the feature values.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing features and labels.

        Returns:
        - tuple: A tuple containing the standardized features (X) and labels (y).
        """
        X = dataset.drop(columns=['mfcc', 'diagnosis'])
        y = dataset['diagnosis']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)
        
        return X, y
    
    def train_svm_model(dataset, test_size=0.2, random_state=42):
        """
        Train an SVM model using the provided dataset, evaluate its performance, and visualize the results.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the SVM model.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.
        """
        dataset = dataset.dropna()
        AudioClassification._check_missing_values(dataset)
        X, y = AudioClassification._extract_svm_features(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        AudioClassification.visualize_performance(y_test, y_pred)

        return model

    def predict(model, features):
        """
        Predict the class labels for the given features using the trained SVM model.

        Parameters:
        - model (SVC): The trained SVM model.
        - features (pd.DataFrame): The features to predict.

        Returns:
        - np.ndarray: The predicted class labels.
        
        Raises:
        - ValueError: If the input features are not a pandas DataFrame.
        """
        if isinstance(features, pd.DataFrame):
            return model.predict(features)
        else:
            raise ValueError("Input features should be a pandas DataFrame.")
        
    def visualize_performance(y_test, y_pred):
        """
        Visualize the performance of the SVM model using a confusion matrix and classification report.

        Parameters:
        y_test (pd.Series): The true labels.
        y_pred (np.ndarray): The predicted labels.
        """
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
        print("Classification Report:")
        print(report)    