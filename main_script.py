import os
import joblib

import pandas as pd

from utilities.utilities import extract_zip, round_to_tenth
from pipeline.data_processing import DatasetManager, AudioProcessing
from pipeline.data_augmentation import DataAugmentation
from pipeline.feature_extraction import FeatureDatasetManager
from pipeline.classification.first_layer.SVM import VowelSVMModel
from pipeline.classification.first_layer.LSTM_mfcc import LSTMModel
from pipeline.classification.first_layer.LSTM_wav2vec import PhraseWav2VecModel, DataGenerator
from pipeline.classification.second_layer import VowelMetaModel, PhraseMetaModel
from pipeline.classification.third_layer import FinalMetaModel
from pipeline.performance_evaluation import PerformanceEvaluator

BASE_DATA_DIR = 'data'
SAMPLE_RATE = 16000

## -----------------------------------------------------------------------------------------------
## ---------- DATA PROCESSING --------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\nProcessing dataset...\n')

# Extract wav files from zip file
extract_zip(os.path.join('data', 'svd_original.zip'), BASE_DATA_DIR)

# Create dataset from folder structure
dataset_manager = DatasetManager(os.path.join(BASE_DATA_DIR, 'svd'))
vowel_dataset, phrase_dataset = dataset_manager.create_dataset()
dataset_manager.plot_dataset_distribution(vowel_dataset, 
                                          os.path.join('data', 'distribution_pathologies.json'), 
                                          os.path.join('images', 'dataset_distribution.png'))

# Align dataset 
vowel_dataset, phrase_dataset = dataset_manager.align_dataset(vowel_dataset, phrase_dataset)

# Resample audios
vowel_dataset['audio_data'] = AudioProcessing.resample_audios(vowel_dataset['audio_data'], 
                                                              vowel_dataset['sample_rate'], 
                                                              SAMPLE_RATE)
phrase_dataset['audio_data'] = AudioProcessing.resample_audios(phrase_dataset['audio_data'], 
                                                               phrase_dataset['sample_rate'], 
                                                               SAMPLE_RATE)
vowel_dataset['sample_rate'] = SAMPLE_RATE
phrase_dataset['sample_rate'] = SAMPLE_RATE

# Pad audios 
vowel_dataset_stats = DatasetManager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = DatasetManager.analyse_dataset(phrase_dataset)
vowel_target_duration = round_to_tenth(max(vowel_dataset_stats['max_duration']))
phrase_target_duration = round_to_tenth(max(phrase_dataset_stats['max_duration']))
vowel_dataset['audio_data'] = AudioProcessing.pad_audios(vowel_dataset['audio_data'], 
                                                         vowel_dataset['sample_rate'], 
                                                         vowel_target_duration)
phrase_dataset['audio_data'] = AudioProcessing.pad_audios(phrase_dataset['audio_data'], 
                                                          phrase_dataset['sample_rate'], 
                                                          phrase_target_duration)

# Save padded audios
DatasetManager.save_all_audios(vowel_dataset['audio_data'], 
                               vowel_dataset['sample_rate'], 
                               vowel_dataset['audio_path'])
DatasetManager.save_all_audios(phrase_dataset['audio_data'], 
                               phrase_dataset['sample_rate'], 
                               phrase_dataset['audio_path'])

# Save dataset
DatasetManager.save_dataset(vowel_dataset, 
                            os.path.join('datasets', 'processed', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_dataset, 
                            os.path.join('datasets', 'processed', 'phrase_dataset.pkl'))

# Extract statistics from dataset
vowel_dataset_stats = dataset_manager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dataset_manager.analyse_dataset(phrase_dataset)
# print('\nVowel dataset stats:\n')
# print(vowel_dataset_stats)
# print('\nPhrase dataset stats:\n')
# print(phrase_dataset_stats)

## -----------------------------------------------------------------------------------------------
## ---------- DATA AUGMENTATION ------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Augmenting dataset...\n')

"""
As the dataset shows inherent imbalances, this section will augment audio data 
to ensure a balances set. This is done by two methods:
- Adding white noise
- Time strechting the audio
"""

# Health male -> 498 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') 
                                & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') 
                                  & (vowel_dataset['diagnosis'] == 'health')],
    n_augmentations=498
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Health female -> 368 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'female') 
                                & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'female') 
                                  & (vowel_dataset['diagnosis'] == 'health')],
    n_augmentations=368
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Path male -> 113 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') 
                                & (vowel_dataset['diagnosis'] == 'path')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') 
                                  & (vowel_dataset['diagnosis'] == 'path')],
    n_augmentations=123
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Path female -> 24 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'female') 
                                & (vowel_dataset['diagnosis'] == 'path')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'female') 
                                  & (vowel_dataset['diagnosis'] == 'path')],
    n_augmentations=24
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Save dataset
DatasetManager.save_dataset(vowel_dataset, 
                            os.path.join('datasets', 'augmented', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_dataset, 
                            os.path.join('datasets', 'augmented', 'phrase_dataset.pkl'))

# Extract statistics from dataset
vowel_dataset_stats = dataset_manager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dataset_manager.analyse_dataset(phrase_dataset)
# print('\nVowel dataset stats:\n')
# print(vowel_dataset_stats)
# print('\nPhrase dataset stats:\n')
# print(phrase_dataset_stats)

## -----------------------------------------------------------------------------------------------
## ---------- FEATURE EXTRACTION -----------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Extracting features from dataset...\n')

# Calculate padsize 
wav2vec2_window_size = 20
mfcc_window_size = 20
vowel_padsize_mfcc = round((vowel_dataset_stats['mean_duration'][0] * 1000) / mfcc_window_size) 
phrase_padsize_mfcc = round((phrase_dataset_stats['mean_duration'][0] * 1000) / mfcc_window_size) 
padsize_wav2vec2 = round((phrase_dataset_stats['mean_duration'][0] * 1000) / wav2vec2_window_size) 

# Extract features
vowel_feature_dataset, phrase_feature_dataset = FeatureDatasetManager.create_dataset(
    vowel_dataset=vowel_dataset,
    phrase_dataset=phrase_dataset,
    vowel_padsize_mfcc=vowel_padsize_mfcc, 
    phrase_padsize_mfcc=phrase_padsize_mfcc,
    padsize_wav2vec2=padsize_wav2vec2
)

# Save dataset
DatasetManager.save_dataset(vowel_feature_dataset, 
                            os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_feature_dataset, 
                            os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))

## -----------------------------------------------------------------------------------------------
## ---------- BASE MODEL TRAINING ----------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Training first-layer models...\n')

vowel_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 
                                                    'vowel_dataset.pkl'))
phrase_feature_dataset = pd.read_pickle(os.path.join('datasets', 'feature_extracted', 
                                                     'phrase_dataset.pkl'))

# Train base models
vowel_svm_model = VowelSVMModel.train_model(vowel_feature_dataset)
vowel_mfcc_lstm_model = LSTMModel.train_model(vowel_feature_dataset)
phrase_mfcc_lstm_model = LSTMModel.train_model(phrase_feature_dataset)
phrase_wav2vec_lstm_model = PhraseWav2VecModel.train_model(phrase_feature_dataset)

# Save base models
joblib.dump(vowel_svm_model, os.path.join('models', 'vowel_svm_model.joblib'))
vowel_mfcc_lstm_model.save(os.path.join('models', 'vowel_mfcc_lstm_model.h5'))
phrase_mfcc_lstm_model.save(os.path.join('models', 'phrase_mfcc_lstm_model.h5'))
phrase_wav2vec_lstm_model.save(os.path.join('models', 'phrase_wav2vec_lstm_model.h5'))

## -----------------------------------------------------------------------------------------------
## ---------- SECOND LAYER META MODEL TRAINING ---------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Training second-layer models...\n')

# Extract test set from featuresets
vowel_feature_dataset = vowel_feature_dataset.dropna()
_, vowel_svm_X_test, _, vowel_svm_y_test = VowelSVMModel.extract_features(vowel_feature_dataset)
_, vowel_mfcc_X_test, _, vowel_mfcc_y_test = LSTMModel.extract_features(vowel_feature_dataset)
_, phrase_mfcc_X_test, _, phrase_mfcc_y_test = LSTMModel.extract_features(phrase_feature_dataset)
_, phrase_wav2vec_X_test, _, phrase_wav2vec_y_test = PhraseWav2VecModel.extract_features(
    phrase_feature_dataset)
wav2vec_test_generator = DataGenerator(phrase_wav2vec_X_test, phrase_wav2vec_y_test, 
                                       shuffle=False)

# Predictions
vowel_svm_predictions = vowel_svm_model.predict(vowel_svm_X_test)
vowel_mfcc_lstm_predictions = vowel_mfcc_lstm_model.predict(vowel_mfcc_X_test)
phrase_mfcc_lstm_predictions = phrase_mfcc_lstm_model.predict(phrase_mfcc_X_test)
phrase_wav2vec_lstm_predictions = PhraseWav2VecModel.predict(phrase_wav2vec_lstm_model, 
                                                             wav2vec_test_generator, 
                                                             len(wav2vec_test_generator))

# Train meta-models
vowel_meta_model = VowelMetaModel.train_model(vowel_svm_predictions, vowel_mfcc_lstm_predictions, 
                                              vowel_svm_y_test)
phrase_meta_model = PhraseMetaModel.train_model(phrase_mfcc_lstm_predictions, 
                                                phrase_wav2vec_lstm_predictions, 
                                                phrase_mfcc_y_test)

# Save meta-models
joblib.dump(vowel_meta_model, os.path.join('models', 'vowel_meta_model.joblib'))
joblib.dump(phrase_meta_model, os.path.join('models', 'phrase_meta_model.joblib'))

## -----------------------------------------------------------------------------------------------
## ---------- THIRD LAYER META MODEL TRAINING ----------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Training third-layer models...\n')

# Extract test set from predictions
_, vowel_MM_X_test, _, vowel_MM_y_test = VowelMetaModel.stack_split_predictions(
    vowel_svm_predictions,
    vowel_mfcc_lstm_predictions, 
    vowel_svm_y_test
)
_, phrase_MM_X_test, _, phrase_MM_y_test = PhraseMetaModel.stack_split_phrase_predictions(
    phrase_mfcc_lstm_predictions,
    phrase_wav2vec_lstm_predictions, 
    phrase_mfcc_y_test
)

# Predictions
vowel_MM_predictions = vowel_meta_model.predict(vowel_MM_X_test)
phrase_MM_predictions = phrase_meta_model.predict(phrase_MM_X_test)

# Train final meta-model
final_meta_model = FinalMetaModel.train_model(vowel_MM_predictions, 
                                              phrase_MM_predictions,
                                              vowel_MM_y_test)

# Save final meta-model
joblib.dump(final_meta_model, os.path.join('models', 'final_meta_model.joblib'))

## -----------------------------------------------------------------------------------------------
## ---------- MODEL EVALUATION -------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------

print('\Evaluating models...\n')

# Initialize performance evaluator with models
performance_evaluator = PerformanceEvaluator(vowel_feature_dataset=vowel_feature_dataset,
                                             phrase_feature_dataset=phrase_feature_dataset,
                                             vowel_svm_model=vowel_svm_model,
                                             vowel_mfcc_lstm_model=vowel_mfcc_lstm_model,
                                             phrase_mfcc_lstm_model=phrase_mfcc_lstm_model,
                                             phrase_wav2vec_lstm_model=phrase_wav2vec_lstm_model,
                                             vowel_meta_model=vowel_meta_model,
                                             phrase_meta_model=phrase_meta_model,
                                             final_meta_model=final_meta_model)

# Evaluate models
performance_evaluator.evaluate_and_plot_first_layer()
performance_evaluator.evaluate_and_plot_second_layer()
performance_evaluator.evaluate_and_plot_third_layer()