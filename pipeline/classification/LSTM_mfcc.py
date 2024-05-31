import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical, plot_model, model_to_dot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydot import graph_from_dot_data
import matplotlib.pyplot as plt
import seaborn as sns

class AudioClassificationLSTM():
    def __init__(self):
        pass

    @staticmethod
    def _check_missing_values(dataset):
        """
        Check for missing values in the dataset and print the count of missing values for each column.

        Parameters:
        - dataset (pd.DataFrame): The dataset to check for missing values.
        """
        missing_values = dataset.isnull().sum()
        print("Missing values in each column:")
        print(missing_values)

    @staticmethod
    def _extract_lstm_features(dataset, test_size=0.2, random_state=42):
        """
        Extract MFCC features and labels for LSTM classification.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing MFCC features and labels.

        Returns:
        - tuple: A tuple containing the features (X) and labels (y).
        """
        X = dataset['mfcc'].values
        y = dataset['diagnosis'].values
        y = to_categorical(y)
        X = np.array([x for x in X])
        y = np.array([x for x in y])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_lstm_model(dataset, test_size=0.2, random_state=42, epochs=50, batch_size=32, visualize_performance=True):
        """
        Train an LSTM model using the provided dataset, evaluate its performance, and visualize the results.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the LSTM model.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of samples per gradient update.
        """
        X_train, X_test, y_train, y_test = AudioClassificationLSTM._extract_lstm_features(dataset, test_size, random_state)

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # plot_model(model, to_file=os.path.join('images', 'LSTM_summary.png'), show_shapes=True, show_layer_names=False)
        
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr])
        y_pred = model.predict(X_test)
       
        if visualize_performance: 
            AudioClassificationLSTM.visualize_performance(y_test, y_pred)

        return model, history

    @staticmethod
    def predict(model, features):
        """
        Predict the class labels for the given features using the trained LSTM model.

        Parameters:
        - model (Sequential): The trained LSTM model.
        - features (np.ndarray): The features to predict.

        Returns:
        - np.ndarray: The predicted class labels.
        
        Raises:
        - ValueError: If the input features are not a numpy ndarray.
        """
        if isinstance(features, np.ndarray):
            features_padded = pad_sequences(features, padding='post', dtype='float32')
            predictions = model.predict(features_padded)
            return np.argmax(predictions, axis=1)
        else:
            raise ValueError("Input features should be a numpy ndarray.")
        
    @staticmethod
    def visualize_performance(y_test, y_pred):
        """
        Visualize the performance of the LSTM model using a confusion matrix and classification report.

        Parameters:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        """
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        # Accuracy
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')

        # Confusion Matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join('images', 'phrase_mfcc_lstm_model_confusion.png'))

        # Classification Report
        report = classification_report(y_test_classes, y_pred_classes, target_names=['Class 0', 'Class 1'])
        print("Classification Report:")
        print(report)

# Example usage
# df = pd.read_csv('path_to_your_dataframe.csv') # Load your DataFrame
# lstm_classifier = AudioClassificationLSTM()
# model = lstm_classifier.train_lstm_model(df)