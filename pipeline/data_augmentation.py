import librosa
import torchaudio.transforms as T
import torch
import numpy as np
import random
import os
import pandas as pd
import soundfile as sf
from pipeline.data_processing import AudioProcessing

SAMPLE_RATE = 16000

class DataAugmentation:
    def __init__(self):
        pass
    
    @staticmethod
    def add_white_noise(audio, noise_level=0.05):
        """
        Adds white noise to the audio sample.

        Parameters:
        - audio (ndarray): The input audio signal.
        - noise_level (float): The amplitude of the noise.

        Returns:
        - The white-noise-added audio array.
        """
        noise = np.random.randn(len(audio))
        audio_noisy = audio + noise_level * noise
        return audio_noisy

    @staticmethod
    def pitch_shift(audio, sample_rate, shift_steps=2.5):
        """
        Shifts the pitch of the audio.
 
        Parameters:
        - audio (ndarray): The input audio signal.
        - sample_rate (int): The sample rate of the audio.
        - n_steps (float): How many steps to shift the audio signal.

        Returns:
        - The pitch-shifted audio array.
        """
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=shift_steps)

    @staticmethod
    def time_stretch(audio, stretch_rate=0.5):
        """
        Time-stretches the audio by a fixed rate.

        Parameters:
        - audio (ndarray): The input audio signal.
        - stretch_rate (float): Factor by which the audio will be sped up (>1) or slowed down (<1).

        Returns:
        - The time-strechted audio array.
        """
        return librosa.effects.time_stretch(y=audio, rate=stretch_rate)

    @staticmethod
    def augment_audio(vowel_dataset, phrase_dataset, vowel_stats, phrase_stats, aug_type, n_audios, noise_level=0.05, shift_steps=2.5, stretch_rate=0.9):
        """
        Augments a given dataset using the specified augmentation type.

        Not the full datasets are passed, but only the parts that are to be augmented. For example, if male, healthy individuals are to be augmented,
        the passed datasets only contains said entries. For each sample type, a empty dataset will be created appon calling the method.
        These datasets contain the augmented audios, which will be returned to the parent method after data augmentation is complete.

        A second version of the program could split each augmentation type, giving the user more flexibility in choosing the desired augmentation method.
        This would also make more sense regarding the additional parameters such as noise_level, shift_steps and stretch_rate. 
    
        In the current version, the aug_type chooses the desired type of augmentation.

        Parameters:
        - vowel_dataset (DataFrame): DataFrame containing vowel audio samples to be augmented.
        - phrase_dataset (DataFrame): DataFrame containing phrase audio samples to be augmented.
        - aug_type (str): The type of augmentation to apply ('noise', 'pitch', 'stretch').
        - n_audios (int): The number of audio samples to augment.
        - noise_level (float): The amplitude of the noise to be added (only used if aug_type is 'noise').
        - shift_steps (float): The number of steps to shift the pitch (only used if aug_type is 'pitch').
        - stretch_rate (float): The factor by which to stretch the time (only used if aug_type is 'stretch').

        Returns:
        - Tuple containing the augmented vowel dataset and augmented phrase dataset.
        """

        """
        TODO: Idea is to make the parameter value random as well. The augmentation args can be given,
        but the default takes a random value within a defined range. This ensures a more well-distributed dataset.
        """

        aug_vowel_dataset = pd.DataFrame(columns=['speaker_id', 'audio_path', 'audio_data', 'sample_rate', 'sex', 'diagnosis'])
        aug_phrase_dataset = pd.DataFrame(columns=['speaker_id', 'audio_path', 'audio_data', 'sample_rate', 'sex', 'diagnosis'])
        used_indices = set()

        while len(used_indices) < n_audios:
            idx = random.randint(0, len(vowel_dataset) - 1)
            if idx in used_indices:
                continue
            used_indices.add(idx)

            vowel_entry = vowel_dataset.iloc[idx]
            phrase_entry = phrase_dataset.iloc[idx]
            vowel_audio = vowel_entry['audio_data']
            phrase_audio = phrase_entry['audio_data']
            vowel_sample_rate = vowel_entry['sample_rate']
            phrase_sample_rate = phrase_entry['sample_rate']
            vowel_original_length = len(vowel_audio)
            phrase_original_length = len(phrase_audio)

            if aug_type == 'noise':
                aug_vowel_audio_data = DataAugmentation.add_white_noise(vowel_audio, noise_level)
                aug_phrase_audio_data = DataAugmentation.add_white_noise(phrase_audio, noise_level)
                aug_vowel_audio_path = DataAugmentation._save_audio(aug_vowel_audio_data, vowel_sample_rate, vowel_entry['audio_path'])
                aug_phrase_audio_path = DataAugmentation._save_audio(aug_phrase_audio_data, phrase_sample_rate, phrase_entry['audio_path'])
            elif aug_type == 'stretch':
                aug_vowel_audio_data = DataAugmentation.time_stretch(vowel_audio, stretch_rate)
                aug_phrase_audio_data = DataAugmentation.time_stretch(phrase_audio, stretch_rate)
                aug_vowel_audio_path = DataAugmentation._save_audio(aug_vowel_audio_data, vowel_sample_rate, vowel_entry['audio_path'])
                aug_phrase_audio_path = DataAugmentation._save_audio(aug_phrase_audio_data, phrase_sample_rate, phrase_entry['audio_path'])
            else:
                raise ValueError(f"{aug_type} is not supported. Select from: 'noise' and 'stretch'.")
            '''
            elif aug_type == 'pitch':
                aug_vowel_audio_data = DataAugmentation.pitch_shift(vowel_audio, vowel_sample_rate, shift_steps)
                aug_phrase_audio_data = DataAugmentation.pitch_shift(phrase_audio, phrase_sample_rate, shift_steps)
                new_vowel_pitch = AudioProcessing.calculate_pitch(aug_vowel_audio_data, vowel_sample_rate)
                new_phrase_pitch = AudioProcessing.calculate_pitch(aug_phrase_audio_data, phrase_sample_rate)
                pitch_shifted_vowel_audio_path = DataAugmentation._update_labels(vowel_stats, new_vowel_pitch, vowel_entry['audio_path'])
                pitch_shifted_phrase_audio_path = DataAugmentation._update_labels(phrase_stats, new_phrase_pitch, phrase_entry['audio_path'])
                aug_vowel_audio_path = DataAugmentation._save_audio(aug_vowel_audio_data, vowel_sample_rate, pitch_shifted_vowel_audio_path)
                aug_phrase_audio_path = DataAugmentation._save_audio(aug_phrase_audio_data, phrase_sample_rate, pitch_shifted_phrase_audio_path)
            '''

            # Check if length was altered and if so, fix it to original length
            aug_vowel_audio_data = DataAugmentation._check_fix_length(aug_vowel_audio_data, vowel_original_length)
            aug_phrase_audio_data = DataAugmentation._check_fix_length(aug_phrase_audio_data, phrase_original_length)
            
            # Add the new entry to the augmented datasets
            DataAugmentation._add_entry_dataframe(aug_vowel_dataset, vowel_entry['speaker_id'], aug_vowel_audio_path, aug_vowel_audio_data, vowel_sample_rate)
            DataAugmentation._add_entry_dataframe(aug_phrase_dataset, phrase_entry['speaker_id'], aug_phrase_audio_path, aug_phrase_audio_data, phrase_sample_rate)

        return aug_vowel_dataset, aug_phrase_dataset
    
    @staticmethod
    def random_augmentation(vowel_dataset, phrase_dataset, vowel_stats, phrase_stats, n_augmentations):
        """
        Performs random augmentations on the datasets.

        Parameters:
        - vowel_dataset (DataFrame): The vowel dataset to augment.
        - phrase_dataset (DataFrame): The phrase dataset to augment.
        - vowel_stats (DataFrame): Statistics of the vowel dataset.
        - phrase_stats (DataFrame): Statistics of the phrase dataset.
        - n_augmentations (int): Total number of augmentations to perform.

        Returns:
        - Augmented vowel and phrase datasets.
        """
        augmentation_types = ['noise', 'stretch']
        noise_level_range = (0.02, 0.08)
        stretch_rate_range = (0.8, 1.2)
        aug_vowel_datasets = []
        aug_phrase_datasets = []

        for _ in range(n_augmentations):
            aug_type = random.choice(augmentation_types)
            if aug_type == 'noise':
                noise_level = random.uniform(*noise_level_range)
                aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.augment_audio(
                    vowel_dataset=vowel_dataset,
                    phrase_dataset=phrase_dataset,
                    vowel_stats=vowel_stats,
                    phrase_stats=phrase_stats,
                    aug_type=aug_type,
                    n_audios=1,
                    noise_level=noise_level
                )
            elif aug_type == 'stretch':
                stretch_rate = random.uniform(*stretch_rate_range)
                aug_vowel_dataset, aug_phrase_dataset = DataAugmentation.augment_audio(
                    vowel_dataset=vowel_dataset,
                    phrase_dataset=phrase_dataset,
                    vowel_stats=vowel_stats,
                    phrase_stats=phrase_stats,
                    aug_type=aug_type,
                    n_audios=1,
                    stretch_rate=stretch_rate
                )

            aug_vowel_datasets.append(aug_vowel_dataset)
            aug_phrase_datasets.append(aug_phrase_dataset)

        combined_vowel_dataset = pd.concat(aug_vowel_datasets, ignore_index=True)
        combined_phrase_dataset = pd.concat(aug_phrase_datasets, ignore_index=True)

        return combined_vowel_dataset, combined_phrase_dataset

    @staticmethod
    def _check_fix_length(audio, original_length):
        """
        Ensures the audio length matches the original length.

        Parameters:
        - audio (ndarray): The input audio signal.
        - original_length (int): The original length of the audio signal.

        Returns:
        - The audio array with the fixed length.
        """
        return librosa.util.fix_length(data=audio, size=original_length) if len(audio) != original_length else audio
    
    @staticmethod
    def _save_audio(audio, sample_rate, original_path):
        """
        Saves the augmented audio to a new file.

        Parameters:
        - audio (ndarray): The augmented audio signal.
        - sample_rate (int): The sample rate of the audio.
        - original_path (str): The original path of the audio file.

        Returns:
        - The new path of the saved augmented audio file.
        """
        base, ext = os.path.splitext(original_path)
        new_path = f"{base}_aug{ext}"
        sf.write(new_path, audio, sample_rate)  # Assuming fixed sample rate for simplicity

        return new_path

    @staticmethod
    def _add_entry_dataframe(dataset, speaker_id, audio_path, audio_data, sample_rate):
        """
        Adds a new entry to the dataset DataFrame.

        Parameters:
        - dataset (DataFrame): The dataset to which the entry will be added.
        - speaker_id (int): The speaker ID.
        - audio_path (str): The path of the audio file.
        - audio_data (ndarray): The audio signal data.
        - sample_rate (int): The sample rate of the audio signal.

        Returns:
        - The updated dataset DataFrame.
        """
        audio_path_parts = audio_path.split(os.path.sep)
        diagnosis = audio_path_parts[2]
        sex = audio_path_parts[3]

        new_entry = pd.DataFrame({
            'speaker_id': [speaker_id],
            'audio_path': [audio_path],
            'audio_data': [audio_data],
            'sample_rate': [sample_rate],
            'sex': [sex],
            'diagnosis': [diagnosis]
        })
        dataset = pd.concat([dataset, new_entry], ignore_index=True)

        return dataset

    @staticmethod
    def _update_labels(stats, new_pitch, original_audio_path):
        """
        Updates the labels based on the closest matching pitch statistics.

        Parameters:
        - stats (DataFrame): DataFrame containing pitch statistics.
        - new_pitch (float): The new pitch of the audio signal.
        - original_audio_path (str): The original path of the audio file.

        Returns:
        - The new audio path with updated labels.
        """
        closest = None
        new_audio_path = ''
        min_distance = float('inf')
        audio_path_parts = original_audio_path.split(os.path.sep)
        base = os.path.sep.join(audio_path_parts[:2])
        diagnosis = audio_path_parts[2]
        sample_type = audio_path_parts[4]
        file_name = audio_path_parts[5]

        for _, row in stats[stats['diagnosis'] == diagnosis].iterrows():
            mean_pitch = row['mean_pitch']
            std_pitch = row['std_pitch']
            distance = abs(new_pitch - mean_pitch) / std_pitch # Calculate the distance considering both mean and standard deviation
            if distance < min_distance:
                min_distance = distance
                closest = row

        if closest is not None:
            category = 'health' if closest['diagnosis'] == 'healthy' else 'path'
            sex = 'male' if closest['sex'] == 'M' else 'female'
            new_audio_path = os.path.join(base, category, sex, sample_type, file_name)

        return new_audio_path