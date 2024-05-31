import joblib
import os 
import pandas as pd
from pipeline.classification.SVM import AudioClassificationSVM
from pipeline.classification.LSTM_mfcc import AudioClassificationLSTM
from pipeline.classification.LSTM_wav2vec import AudioClassificationWav2Vec, DataGenerator

vowel_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
phrase_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))

# Train the models
# vowel_svm_model = AudioClassificationSVM.train_svm_model(vowel_feature_dataset)
vowel_mfcc_lstm_model, _ = AudioClassificationLSTM.train_lstm_model(vowel_feature_dataset)
# phrase_mfcc_lstm_model, _ = AudioClassificationLSTM.train_lstm_model(phrase_feature_dataset)
# phrase_wav2vec_lstm_model, _ = AudioClassificationWav2Vec.train_lstm_model(phrase_feature_dataset)

# Save the models
# joblib.dump(vowel_svm_model, os.path.join('models', 'vowel_svm_model.joblib'))
# vowel_mfcc_lstm_model.save(os.path.join('models', 'vowel_mfcc_lstm_model.keras'))
# phrase_mfcc_lstm_model.save(os.path.join('models', 'phrase_mfcc_lstm_model.keras'))
# phrase_wav2vec_lstm_model.save(os.path.join('models', 'phrase_wav2vec_lstm_model.keras'))