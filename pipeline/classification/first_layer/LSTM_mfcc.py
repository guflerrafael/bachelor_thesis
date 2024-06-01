
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LSTMModel():
    def __init__(self):
        pass

    @staticmethod
    def extract_features(dataset: pd.DataFrame, 
                         test_size: float = 0.2, 
                         random_state: int = 42) -> tuple:
        """
        Extract and split features and labels for LSTM classification.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing MFCC features and labels.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.

        Returns:
        - tuple: A tuple containing the features (X) and labels (y).
        """
        X = dataset['mfcc'].values
        y = dataset['diagnosis'].values
        y = to_categorical(y)
        X = np.array([x for x in X])
        y = np.array([x for x in y])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                            random_state=random_state)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def train_model(dataset: pd.DataFrame, 
                    test_size: float = 0.2, 
                    random_state: int = 42, 
                    epochs: int = 50, 
                    batch_size: int = 32) -> tuple:
        """
        Train a LSTM model using the provided dataset.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the LSTM model.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of samples per gradient update.

        Returns:
        - tuple: A tuple containing the trained model and the training history.
        """
        X_train, _, y_train, _ = LSTMModel.extract_features(dataset, test_size, random_state)

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
                
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                            validation_split=0.2,
                            callbacks=[early_stopping, reduce_lr])

        return model, history

    @staticmethod
    def predict(model: Sequential, features: np.ndarray) -> np.ndarray:
        """
        Predict the labels for the given features using the trained LSTM model.

        Parameters:
        - model (Sequential): The trained LSTM model.
        - features (np.ndarray): The features to predict.

        Returns:
        - np.ndarray: The predicted class labels.
        """
        if isinstance(features, np.ndarray):
            features_padded = pad_sequences(features, padding='post', dtype='float32')
            predictions = model.predict(features_padded)
            return np.argmax(predictions, axis=1)
        else:
            raise ValueError("Input features should be a numpy ndarray.")