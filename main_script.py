import os
import pandas as pd
from pipeline.data_processing import DatasetManager, AudioProcessing
from pipeline.data_augmentation import DataAugmentation
from pipeline.feature_extraction import FeatureDatasetManager
from utilities.utilities import extract_zip, round_up_to_half

BASE_DATA_DIR = 'data'

# ------------- DATA PROCESSING -----------------------------------------------------

# Extract wav files from zip file
extract_zip(os.path.join('data', 'svd_original.zip'), BASE_DATA_DIR)

# Create dataset from folder structure
dataset_manager = DatasetManager(os.path.join(BASE_DATA_DIR, 'svd'))
vowel_dataset, phrase_dataset = dataset_manager.create_dataset()
dataset_manager.plot_dataset_distribution(vowel_dataset, os.path.join('data', 'distribution_pathologies.json'), os.path.join('images', 'dataset_distribution.png'))

# Extract statistics from dataset
vowel_dataset_stats = dataset_manager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dataset_manager.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)

# Align dataset 
vowel_dataset, phrase_dataset = dataset_manager.align_dataset(vowel_dataset, phrase_dataset)

# Resample audios
target_sample_rate = 16000
vowel_dataset['audio_data'] = AudioProcessing.resample_all_audios(vowel_dataset['audio_data'], vowel_dataset['sample_rate'], target_sample_rate)
phrase_dataset['audio_data'] = AudioProcessing.resample_all_audios(phrase_dataset['audio_data'], phrase_dataset['sample_rate'], target_sample_rate)
vowel_dataset['sample_rate'] = target_sample_rate
phrase_dataset['sample_rate'] = target_sample_rate

# Pad audios 
vowel_dataset_stats = DatasetManager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = DatasetManager.analyse_dataset(phrase_dataset)
vowel_target_duration = round_up_to_half(max(vowel_dataset_stats['max_duration']))
phrase_target_duration = round_up_to_half(max(phrase_dataset_stats['max_duration']))
vowel_dataset['audio_data'] = AudioProcessing.pad_audios(vowel_dataset['audio_data'], vowel_dataset['sample_rate'], vowel_target_duration)
phrase_dataset['audio_data'] = AudioProcessing.pad_audios(phrase_dataset['audio_data'], phrase_dataset['sample_rate'], phrase_target_duration)

# Save padded audios
DatasetManager.save_all_audios(vowel_dataset['audio_data'], vowel_dataset['sample_rate'], vowel_dataset['audio_path'])
DatasetManager.save_all_audios(phrase_dataset['audio_data'], phrase_dataset['sample_rate'], phrase_dataset['audio_path'])

# Save dataset
DatasetManager.save_dataset(vowel_dataset, os.path.join('datasets', 'processed', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_dataset, os.path.join('datasets', 'processed', 'phrase_dataset.pkl'))

# Extract statistics from dataset
vowel_dataset_stats = dataset_manager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dataset_manager.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)

print(vowel_dataset)
print(phrase_dataset)

# ------------- DATA AUGMENTATION -----------------------------------------------------

'''
As the dataset shows inherent imbalances, this section will augment audio data to ensure a balances set.
This is done by one of three methods:
- Adding white noise
- Pitch shifting the audio
- Time strechting the audio

As vowel and phrase entries are already alignes, either one of these options creates a new dataset entry in both the vowel and phrase dataset. 
To ensure a good overview, the augmented data will be markes via the additon of "_augmented" at the end of the audio path.

For each augmentation, the passed augmentation type is applied to a randomly selected audio from the passed datasets. Not the full datasets are passed, 
but only the category and/or sex to be augmented. This way, the dataset can be balanced out, while still giving the user enough control and overviwe of the process.

Another option to consider would be the following:
The user can not specifically decide which augmentation type is used, but the augmentation parameters can be set. 
The augmentation type will be selected by taking turns, ensuring each augmentation type is represented equally.
'''

## Health male -> 
# 498 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'health')],
    n_augmentations=498
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

##  Health female
# -> 368 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'health')],
    n_augmentations=368
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

## Path male 
# 113 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'path')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'path')],
    n_augmentations=123
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

## Path male 
# 24 samples to augment
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.random_augmentation(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'path')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'path')],
    n_augmentations=24
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Save dataset
DatasetManager.save_dataset(vowel_dataset, os.path.join('datasets', 'augmented', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_dataset, os.path.join('datasets', 'augmented', 'phrase_dataset.pkl'))

# Extract statistics from dataset
vowel_dataset_stats = dataset_manager.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dataset_manager.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)

# ------------- FEATURE EXTRACTION -----------------------------------------------------

# TODO: Calculate padsize 

vowel_feature_dataset, phrase_feature_dataset = FeatureDatasetManager.create_dataset(
    vowel_dataset=vowel_dataset,
    phrase_dataset=phrase_dataset,
    padsize_mfcc=350,
    padsize_wav2vec2=350
)

# Save dataset
DatasetManager.save_dataset(vowel_feature_dataset, os.path.join('datasets', 'feature_extracted', 'vowel_dataset.pkl'))
DatasetManager.save_dataset(phrase_feature_dataset, os.path.join('datasets', 'feature_extracted', 'phrase_dataset.pkl'))

print(vowel_feature_dataset)
print(phrase_feature_dataset)

# ------------- CLASSIFICATION -----------------------------------------------------

