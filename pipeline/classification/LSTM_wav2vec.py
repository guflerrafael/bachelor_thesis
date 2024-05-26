import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size=32, max_len=320, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_len = max_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = [self.X[i] for i in indices]
        X_batch = np.array(X_batch)
        # Shape is (32, 12, 320, 768)
        X_batch = X_batch.reshape((X_batch.shape[0], X_batch.shape[2], (X_batch.shape[1] * X_batch.shape[3])))  # Reshape to (batch_size, 320, 12 * 768)
        y_batch = [self.y[i] for i in indices]
        y_batch = to_categorical(y_batch, num_classes=2)
        return X_batch, y_batch

    def on_epoch_end(self):
        self.indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indices)

class AudioClassificationWav2Vec():
    def __init__(self):
        pass

    @staticmethod
    def check_array_shapes(dataset):
        """
        Check if all arrays in the dataset have the same shape.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing wav2vec features.
        """
        shapes = [np.array(x).shape for x in dataset['wav2vec2']]
        unique_shapes = set(shapes)
        
        if len(unique_shapes) == 1:
            print("All arrays have the same shape:", unique_shapes)
        else:
            print("Arrays have different shapes:", unique_shapes)

        # Optionally, print each shape for detailed inspection
        # for i, shape in enumerate(shapes):
        #     print(f"Array {i} shape: {shape}")
    
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
    def _extract_wav2vec_features(dataset):
        """
        Extract wav2vec features and labels for LSTM classification.

        Parameters:
        - dataset (pd.DataFrame): The dataset containing wav2vec features and labels.

        Returns:
        - tuple: A tuple containing the features (X) and labels (y).
        """
        X = dataset['wav2vec2'].to_list()
        y = dataset['diagnosis'].to_list()

        return X, y

    @staticmethod
    def train_lstm_model(dataset, test_size=0.2, random_state=42, epochs=50, batch_size=32):
        """
        Train an LSTM model using the provided dataset, evaluate its performance, and visualize the results.

        Parameters:
        - dataset (pd.DataFrame): The dataset to train the LSTM model.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used by the random number generator.
        - epochs (int): Number of epochs to train the model.
        - batch_size (int): Number of samples per gradient update.
        """
        # AudioClassificationWav2Vec.check_array_shapes(dataset)
        AudioClassificationWav2Vec._check_missing_values(dataset)
        X, y = AudioClassificationWav2Vec._extract_wav2vec_features(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        train_generator = DataGenerator(X_train, y_train, batch_size=batch_size)
        test_generator = DataGenerator(X_test, y_test, batch_size=batch_size, shuffle=False)

        model = Sequential()
        model.add(Input(shape=(320, 9216)))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        history = model.fit(train_generator, epochs=epochs, validation_data=test_generator,
                        callbacks=[early_stopping, reduce_lr])

        y_pred = AudioClassificationWav2Vec.predict(model, test_generator, len(test_generator))
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(np.array([to_categorical(y, num_classes=2) for y in y_test]), axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        print(f'Model Accuracy: {accuracy * 100:.2f}%')
        AudioClassificationWav2Vec.visualize_performance(y_test_classes, y_pred_classes)

        return model

    @staticmethod
    def predict(model, generator, steps):
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
        predictions = []
        for _ in range(steps):
            x_batch, _ = generator.__getitem__(_)
            batch_pred = model.predict(x_batch)
            predictions.append(batch_pred)
        predictions = np.concatenate(predictions)
        return predictions[:len(generator.X)]
        
    @staticmethod
    def visualize_performance(y_test, y_pred):
        """
        Visualize the performance of the LSTM model using a confusion matrix and classification report.

        Parameters:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        """
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join('images', 'phrase_wav2vec_lstm_model.png'))

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
        print("Classification Report:")
        print(report)

# Example usage
# df = pd.read_csv('path_to_your_dataframe.csv') # Load your DataFrame
# lstm_classifier = AudioClassificationWav2Vec()
# model = lstm_classifier.train_lstm_model(df)