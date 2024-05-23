import os
import math
import pandas as pd
from pipeline.data_processing import DatasetManager, AudioProcessing
from pipeline.data_augmentation import DataAugmentation
from pipeline.feature_extraction import FeatureDatasetManager

# ------------- DATA PROCESSING -----------------------------------------------------
# Create dataset from folder structure
dm = DatasetManager('data/svd')
vowel_dataset, phrase_dataset = dm.create_dataset()
dm.plot_dataset_distribution(vowel_dataset, os.path.join('data', 'distribution_pathologies.json'), os.path.join('images', 'dataset_distribution.png'))

# Extract statistics from dataset
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)

# Extract statistics from dataset
'''
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)
'''

# Align dataset 
vowel_dataset, phrase_dataset = dm.align_dataset(vowel_dataset, phrase_dataset)

# Extract statistics from dataset
'''
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)
'''

# Resample audios
target_sample_rate = 16000
vowel_dataset.loc[:, 'audio_data'] = AudioProcessing.resample_all_audios(vowel_dataset['audio_data'], vowel_dataset['sample_rate'], target_sample_rate)
phrase_dataset.loc[:, 'audio_data'] = AudioProcessing.resample_all_audios(phrase_dataset['audio_data'], phrase_dataset['sample_rate'], target_sample_rate)
vowel_dataset.loc[:, 'sample_rate'] = target_sample_rate
phrase_dataset.loc[:, 'sample_rate'] = target_sample_rate

# Might be moved to utilites class
def round_up_to_half(number):
    """
    Rounds a float up to the nearest multiple of 0.5.
    
    Args:
    number (float): The number to round.

    Returns:
    float: The number rounded up to the nearest 0.5.
    """
    return math.ceil(number * 2) / 2

# Pad audios 
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
vowel_target_duration = round_up_to_half(max(vowel_dataset_stats['max_duration']))
phrase_target_duration = round_up_to_half(max(phrase_dataset_stats['max_duration']))
vowel_dataset.loc[:, 'audio_data'] = AudioProcessing.pad_audios(vowel_dataset['audio_data'], vowel_dataset['sample_rate'], vowel_target_duration)
phrase_dataset.loc[:, 'audio_data'] = AudioProcessing.pad_audios(phrase_dataset['audio_data'], phrase_dataset['sample_rate'], phrase_target_duration)
# dm.save_dataset(vowel_dataset, os.path.join('datasets', 'processed', 'vowel_dataset.csv'))
# dm.save_dataset(phrase_dataset, os.path.join('datasets', 'processed', 'phrase_dataset.csv'))

# Extract statistics from dataset
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
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

'''
# Health male -> 498 samples
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.augment_audio(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'health')],
    aug_type='noise',
    n_audios=100,
    noise_level=0.05
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)

# Health female -> 368 samples 

# Path male -> 113 samples
aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.augment_audio(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'path')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'male') & (vowel_dataset['diagnosis'] == 'path')],
    vowel_stats=vowel_dataset_stats,
    phrase_stats=phrase_dataset_stats,
    aug_type='stretch',
    n_audios=113,
    stretch_rate=0.9
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)
'''

# Path female -> 24 samples

'''
## PROBLEM
# Problem with pitch shift: If labels are updated (in chase the pitch afterwards sounds more like another sex), 
# the sit will very likely be inbalanced as well. This would necessitate a sort of regularization algorithm, 
# to delete augmented audios in the sex subcategory to which the pitch shifted audio is moved.
# Probably to eleborate for now, so pitch_shift will not be supported for now.

aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.augment_audio(
    vowel_dataset=vowel_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'health')],
    phrase_dataset=phrase_dataset[(vowel_dataset['sex'] == 'female') & (vowel_dataset['diagnosis'] == 'health')],
    vowel_stats=vowel_dataset_stats,
    phrase_stats=phrase_dataset_stats,
    aug_type='pitch',
    n_audios=100,
    shift_steps=2.0
)
vowel_dataset = pd.concat([vowel_dataset, aug_vowel_dataset], ignore_index=True)
phrase_dataset = pd.concat([phrase_dataset, aug_phrase_dataset], ignore_index=True)
'''


# Extract statistics from dataset
vowel_dataset_stats = dm.analyse_dataset(vowel_dataset)
phrase_dataset_stats = dm.analyse_dataset(phrase_dataset)
print('\nVowel dataset stats:\n')
print(vowel_dataset_stats)
print('\nPhrase dataset stats:\n')
print(phrase_dataset_stats)

print(vowel_dataset)
print(phrase_dataset)

# ------------- FEATURE EXTRACTION -----------------------------------------------------

# TODO: Calculate padsize 

vowel_feature_dataset, phrase_feature_dataset = FeatureDatasetManager.create_dataset(
    vowel_dataset=vowel_dataset,
    phrase_dataset=phrase_dataset,
    padsize_mfcc=500,
    padsize_wav2vec2=500
)

print(vowel_feature_dataset)
print(phrase_feature_dataset)

print()

# ------------- CLASSIFICATION -----------------------------------------------------

