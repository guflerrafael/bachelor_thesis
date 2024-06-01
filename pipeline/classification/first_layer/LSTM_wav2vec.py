import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical

from typing import List, Tuple

class DataGenerator(Sequence):
    def __init__(self, 
                 X: List[np.ndarray], 
                 y: List[int], 
                 batch_size: int = 32, 
                 max_len: int = 320, 
                 shuffle: bool = True):
        """
        Initialize the DataGenerator.

        Parameters:
        - X (List[np.ndarray]): The input features.
        - y (List[int]): The target labels.
        - batch_size (int): The batch size.
        - max_len (int): The maximum length of the input features.
        - shuffle (bool): Whether to shuffle the data after each epoch.
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """
        Get the number of batches in the generator.

        Returns:
        - int: The number of batches.
        """
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a batch of data.

        Parameters:
        - index (int): The index of the batch.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: The input features and target labels for the batch.
        """
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = [self.X[i] for i in indices]
        X_batch = np.array(X_batch)
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[2], 
                                   (X_batch.shape[1] * X_batch.shape[3])))
        y_batch = [self.y[i] for i in indices]
        y_batch = to_categorical(y_batch, num_classes=2)

        return X_batch, y_batch

    def on_epoch_end(self):
        """
        Shuffle the data after each epoch.
        """
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

class PhraseWav2VecModel():
    def __init__(self):
        pass

    @staticmethod
    def extract_features(dataset: pd.DataFrame, 
                         test_size: float = 0.2, 
                         random_state: int = 42) -> tuple:
        """
        Extract and split wav2vec features and labels for LSTM classification.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing wav2vec features and labels.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.

        Returns:
        - tuple: A tuple containing the features (X) and labels (y).
        """
        X = dataset['wav2vec2'].to_list()
        y = dataset['diagnosis'].to_list()

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
        Train the wav2vec LSTM model using the provided dataset.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the LSTM model.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of samples per gradient update.

        Returns:
        - tuple: A tuple containing the trained model and the training history.
        """
        X_train, X_test, y_train, y_test = PhraseWav2VecModel.extract_features(dataset, 
                                                                               test_size, 
                                                                               random_state)
        train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
        test_generator = DataGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)

        model = Sequential()
        model.add(Input(shape=(320, 9216)))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        history = model.fit(train_generator, epochs=epochs, validation_data=test_generator,
                        callbacks=[early_stopping, reduce_lr])

        return model, history

    @staticmethod
    def predict(model: Sequential, generator: DataGenerator, steps: int) -> np.ndarray:
        """
        Predict the labels for the given features using the trained LSTM model.

        Parameters:
        - model (Sequential): The trained LSTM model.
        - generator (DataGenerator): The data generator for generating batches of data.
        - steps (int): The number of steps to iterate over the generator.

        Returns:
        - np.ndarray: The predicted class labels.
        """
        predictions = []
        for _ in range(steps):
            x_batch, _ = generator.__getitem__(_)
            batch_pred = model.predict(x_batch)
            predictions.append(batch_pred)
        predictions = np.concatenate(predictions)

        return predictions[:len(generator.X)]