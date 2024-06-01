import os
import librosa
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

from typing import List, Tuple, Set
from scipy.signal import butter, lfilter
from matplotlib.patches import Patch

class DataProcessing:
    def __init__(self, base_path: str) -> None:
        """
        Initialize the DataProcessing class with a base path for data handling.

        Parameters:
        - base_path (str): The root directory for data processing.
        """
        self.base_path = base_path

class DatasetManager(DataProcessing):
    def __init__(self, base_path: str) -> None:
        """
        Initialize the DatasetManager class, setting up categories, sexes, sample types, and datasets.

        Parameters:
        - base_path (str): The base path for dataset files.
        """
        super().__init__(base_path)
        self._categories = ['health', 'path']
        self._sexs = ['male', 'female']
        self._sample_types = ['a_n', 'phrase']
        self._vowel_dataset = pd.DataFrame(columns=['speaker_id', 
                                                    'audio_path', 
                                                    'audio_data', 
                                                    'sample_rate', 
                                                    'sex', 
                                                    'diagnosis'])
        self._phrase_dataset = pd.DataFrame(columns=['speaker_id', 
                                                     'audio_path', 
                                                     'audio_data', 
                                                     'sample_rate', 
                                                     'sex', 
                                                     'diagnosis'])
    
    @staticmethod
    def load_wav(path: str) -> Tuple[np.ndarray, int]:
        """
        Load an audio file from the specified path without changing its sampling rate.

        Parameters:
        - path (str): The path to the audio file.

        Returns:
        - Tuple containing audio time series and sampling rate as provided by librosa.load.
        """
        return librosa.load(path, sr=None)
    
    @staticmethod
    def load_all_wav(directory: str) -> List[Tuple[np.ndarray, int]]:
        """
        Loads all `.wav` files from the specified directory.

        Parameters:
        - directory (str): The directory from which to load audio files.

        Returns:
        - List of tuples, each containing audio data and sampling rate of a file.
        """
        return [DatasetManager.load_wav(os.path.join(directory, file)) \
                for file in os.listdir(directory) if file.endswith('.wav')]
        
    @staticmethod
    def save_all_audios(audios: List[np.ndarray], sample_rates: List[int], paths: List[str]) -> None:
        """
        Saves the audios to wav files..

        Parameters:
        - audios (list of ndarray): The audio signals.
        - sample_rates (list of int): The sample rate of the audios.
        - paths (list of str): The path where to save the audio files.
        """
        for audio, sample_rate, path in zip(audios, sample_rates, paths):
            sf.write(path, audio, sample_rate)

    @staticmethod
    def analyse_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Analyzes the dataset by calculating various statistics related to audio duration.

        Parameters:
        - dataset (DataFrame): Pandas dataframe to be analysed.

        Returns: 
        - A DataFrame containing various statistics about the dataframe.
        """
        durations = dataset.apply(
            lambda row: AudioProcessing.calculate_audio_duration(row['audio_data'], 
                                                                 row['sample_rate']), axis=1
        )
        stats_dataset = pd.DataFrame({
            'duration': durations,
            'diagnosis': dataset['diagnosis'],
            'sex': dataset['sex']
        })
        stats = stats_dataset.groupby(['diagnosis', 'sex']).agg({
            'duration': ['mean', 'std', 'max', 'count'],
        })
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.rename(columns={
            'duration_mean': 'mean_duration',
            'duration_std': 'std_duration',
            'duration_max': 'max_duration',
            'duration_count': 'sample_count',
        })
        stats = stats.reset_index()

        return stats
        
    @staticmethod
    def plot_dataset_distribution(dataset: pd.DataFrame, distribution_pathologies_path: str, plot_save_path: str) -> None:
        """
        Plots a summary of the dataset showing health status and the distribution of pathologies.

        Parameters:
        - dataset (DataFrame): Pandas dataframe to be plotted.
        - distribution_pathologies_path (str): Path to the json file containing the pathologies distribution.
        - plot_save_path (str): Path were the plot will be saved.
        """
        grouped_vowel_dataset = dataset.groupby(['diagnosis', 'sex']).size().unstack(fill_value=0)
        grouped_vowel_dataset.index = grouped_vowel_dataset.index.map({'health': 'Healthy',
                                                                        'path': 'Pathological'})
        grouped_vowel_dataset.columns = grouped_vowel_dataset.columns.map({'female': 'Female', 
                                                                           'male': 'Male'})

        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        grouped_vowel_dataset.plot(kind='barh', stacked=False, ax=ax1, rot=0)
        ax1.set_title('(a)')
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Diagnosis')
        legend_handles = [
            Patch(facecolor=default_colors[1], 
                  label=f'Male ({grouped_vowel_dataset["Male"].sum()} samples)'),
            Patch(facecolor=default_colors[0], 
                  label=f'Female ({grouped_vowel_dataset["Female"].sum()} samples)')
        ]
        ax1.legend(handles=legend_handles, loc='lower right', title=None)

        with open(distribution_pathologies_path, 'r') as file:
            distribution_pathologies = json.load(file)
        distribution_pathologies = dict(sorted(distribution_pathologies.items(), 
                                               key=lambda x: x[1], 
                                               reverse=True))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = [default_colors[1] if key == 'Healthy' else default_colors[0] 
                  for key in distribution_pathologies.keys()]
        fig2 = ax2.barh(list(distribution_pathologies.keys()), 
                        list(distribution_pathologies.values()), 
                        color=colors)
        ax2.set_title('(b)')
        ax2.set_xlabel('Time / h')
        healthy_hours = distribution_pathologies.get('Healthy', 0)
        path_hours = sum(val for key, val in distribution_pathologies.items() if key != 'Healthy')
        legend2_labels = [f'Healthy ({healthy_hours} h)', f'Pathological ({path_hours} h)']
        ax2.legend(fig2.patches[:2], legend2_labels, loc='upper right', title=None)

        plt.tight_layout()
        plt.savefig(plot_save_path)

    def align_dataset(self, vowel_dataset: pd.DataFrame, phrase_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Checks and aligns datasets for missing samples across vowel and phrase types 
        for given category and sex. This ensures that each speaker included in the dataset 
        has both a vowel and phrase recording.

        Parameters:
        - vowel_dataset (DataFrame): The pandas dataset containing the vowel information.
        - phrase_dataset (DataFrame): The pandas dataset containing the phrase information.

        Returns:
        - The aligned dataset containing the vowel information.
        - The aligned dataset containing the phrase information.
        """
        vowel_speaker_ids = set(vowel_dataset['speaker_id'])
        phrase_speaker_ids = set(phrase_dataset['speaker_id'])
        missing_vowel_speakers = phrase_speaker_ids - vowel_speaker_ids
        missing_phrase_speakers = vowel_speaker_ids - phrase_speaker_ids

        if missing_vowel_speakers or missing_phrase_speakers:
            for category in self._categories:
                for sex in self._sexs:
                    if missing_vowel_speakers:
                        source_directory = os.path.join(self.base_path, 
                                                        category, 
                                                        sex, 
                                                        self._sample_types[1])
                        AudioFileManager._move_unmatched_files(self, 
                                                               source_directory, 
                                                               missing_vowel_speakers, 
                                                               category, 
                                                               sex, 
                                                               self._sample_types[1])
                    if missing_phrase_speakers:
                        source_directory = os.path.join(self.base_path, 
                                                        category, 
                                                        sex, 
                                                        self._sample_types[0])
                        AudioFileManager._move_unmatched_files(self, 
                                                               source_directory, 
                                                               missing_phrase_speakers, 
                                                               category, 
                                                               sex, 
                                                               self._sample_types[0])

            vowel_mask = ~vowel_dataset['speaker_id'].isin(missing_phrase_speakers)
            phrase_mask = ~phrase_dataset['speaker_id'].isin(missing_vowel_speakers)
            self._vowel_dataset = vowel_dataset[vowel_mask]
            self._phrase_dataset = phrase_dataset[phrase_mask]
            self._vowel_dataset = self._vowel_dataset.reset_index(drop=True)
            self._phrase_dataset = self._phrase_dataset.reset_index(drop=True)

        return self._vowel_dataset, self._phrase_dataset
    
    def create_dataset(self, align_dataset: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Creates the two dataframes for vowel and phrase. If align_dataset 'True',
        the dataset will be aligned (i.e. missing speaker samples will be moved). 
        
        Parameters:
        - align_dataset (bool, optional): If True, the dataset will be aligned.

        Returns:
        - _vowel_dataset: Dataframe containing the speaker id, path to the vowel audio, sex of the speaker and the diagnosis. 
        - _phrase_dataset: Dataframe containing the speaker id, path to the phrase audio, sex of the speaker and the diagnosis.
        """
        for category in self._categories:
            for sex in self._sexs:
                for sample_type in self._sample_types:
                    path = os.path.join(self.base_path, category, sex, sample_type)
                    if os.path.exists(path):
                        audio_files = os.listdir(path)
                        for file in audio_files:
                            if file.endswith('.wav'):
                                speaker_id = AudioFileManager.get_speaker_id(file)
                                audio_path = os.path.join(path, file)
                                audio_data, sample_rate = self.load_wav(audio_path)
                                new_row = pd.DataFrame({
                                    'speaker_id': [speaker_id],
                                    'audio_path': [audio_path],
                                    'audio_data': [audio_data],
                                    'sample_rate': [sample_rate],
                                    'sex': [sex],
                                    'diagnosis': [category]
                                })
                                if sample_type == 'a_n':
                                    self._vowel_dataset = pd.concat([self._vowel_dataset, 
                                                                     new_row], 
                                                                    ignore_index=True)
                                elif sample_type == 'phrase':
                                    self._phrase_dataset = pd.concat([self._phrase_dataset, 
                                                                      new_row], 
                                                                     ignore_index=True)

        self._vowel_dataset = self._vowel_dataset.sort_values('speaker_id')
        self._phrase_dataset = self._phrase_dataset.sort_values('speaker_id')
        self._vowel_dataset = self._vowel_dataset.reset_index(drop=True)
        self._phrase_dataset = self._phrase_dataset.reset_index(drop=True)

        if align_dataset:
            self._vowel_dataset, self._phrase_dataset = self.align_dataset(self._vowel_dataset)

        return self._vowel_dataset, self._phrase_dataset
    
    @staticmethod
    def save_dataset(dataset: pd.DataFrame, file_path: str) -> None:
        """
        Saves the DataFrame to the specified path. The file format specified by the file_path
        will be used to encode the data. The format can either be '.csv', '.json' or '.pkl'.
        
        Parameters:
        - dataset (DataFrame): Dataframe containing the data. 
        - file_path (str): Path to where the file will be saved. File type has to be specified.
        """
        try:
            if file_path.endswith('.csv'):
                dataset_copy = dataset.copy()
                dataset_copy['audio_data'] = dataset_copy['audio_data'].apply(
                    lambda audio_data: json.dumps(audio_data.tolist())
                )
                dataset_copy.to_csv(file_path, index=False)
            elif file_path.endswith('.json'):
                dataset_copy = dataset.copy()
                dataset_copy['audio_data'] = dataset_copy['audio_data'].apply(
                    lambda audio_data: json.dumps(audio_data.tolist())
                )
                dataset_copy.to_json(file_path, orient='records', lines=True)
            elif file_path.endswith('.pkl'):
                dataset.to_pickle(file_path)
            else:
                raise ValueError("Unsupported file format. Please use .csv, .json, or .pkl")
        except Exception as e:
            print(f"Failed to save dataset to {file_path}: {e}")


class AudioFileManager(DataProcessing):
    @ staticmethod
    def get_speaker_id(filename: str) -> int:
        """
        Extracts the speaker ID from the filename.

        Parameters:
        - filename (str): The name of the file.

        Returns:
        - The speaker ID as an integer.
        """
        return int(filename.split('-')[0])

    @staticmethod
    def _list_files(directory: str) -> List[str]:
        """
        Lists `.wav` files in the specified directory.

        Parameters:
        - directory (str): The directory to list files from.

        Returns:
        - List of file names.
        """
        return [file for file in os.listdir(directory) if file.endswith('.wav')]

    @staticmethod
    def _count_files(directory: str) -> int:
        return len(AudioFileManager._list_files(directory))

    @staticmethod
    def _move_unmatched_files(self, source_directory: str, unmatched_ids: Set[int], category: str, sex: str, sample_type: str) -> None:
        """
        Moves files from the source directory to a designated missing samples directory 
        which will be created, if their IDs match the unmatched IDs list.

        Parameters:
        - source_directory (str): Directory from which files are moved.
        - unmatched_ids (set of int): IDs of unmatched speakers.
        - category (str): Health category.
        - gender (str): Gender category.
        - sample_type (str): Type of sample, e.g., 'a_n' or 'phrase'.
        """
        for file in os.listdir(source_directory):
            speaker_id = AudioFileManager.get_speaker_id(file)
            if speaker_id in unmatched_ids:
                target_directory = os.path.join(self.base_path, 
                                                'missing_samples', 
                                                category, 
                                                sex, 
                                                sample_type)
                if not os.path.exists(target_directory):
                    os.makedirs(target_directory)
                source_file_path = os.path.join(source_directory, file)
                target_file_path = os.path.join(target_directory, file)
                shutil.move(source_file_path, target_file_path)

class AudioProcessing(DataProcessing):
    @staticmethod
    def calculate_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculates the duration of an audio file based on its audio data and sample rate.

        Parameters:
        - audio (numpy array): Audio data.
        - sample_rate (int): Sample rate of the audio.

        Returns:
        - Duration of the audio in seconds.
        """
        return librosa.get_duration(y=audio, sr=sample_rate)
    
    @staticmethod
    def calculate_pitch(audio_data: np.ndarray, sample_rate: int) -> float:
        """
        Calculate the predominant pitch of an audio signal using librosa's piptrack method.

        Parameters:
        - audio_data (array): The audio time series data.
        - sample_rate (int): The sampling rate of the audio data.

        Returns:
        - float: The predominant pitch in Hz. If no significant pitch is detected, 0 is returned.
        """
        # Since 75-1500 Hz cover human voice range (with sufficient padding), 
        # this is set as fmin and fmax.
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate, fmin=75, fmax=1500) 
        index_of_maximums = magnitudes.argmax(axis=0)
        pitches_with_high_magnitudes = pitches[index_of_maximums, range(pitches.shape[1])]
        pitches_with_high_magnitudes = pitches_with_high_magnitudes[magnitudes.max(axis=0) 
                                                                    > np.mean(magnitudes)]

        if len(pitches_with_high_magnitudes) == 0:
            return 0.0  # No significant pitch detected

        return np.median(pitches_with_high_magnitudes)
    
    @staticmethod
    def pad_audio(audio: np.ndarray, sample_rate: int, desired_length: float) -> np.ndarray:
        """
        Center pads the audio data to the desired length in seconds.

        Parameters:
        - audio (numpy array): Audio data to be padded.
        - sample_rate (int): Sample rate of the audio.
        - desired_length (float): Desired duration in seconds.
        
        Returns:
        - The padded audio data array.
        """
        current_length = len(audio)
        desired_length_samples = int(desired_length * sample_rate)

        if current_length > desired_length_samples:
            return audio[:desired_length_samples]
            
        total_padding = desired_length_samples - current_length
        padding_before = total_padding // 2
        padding_after = total_padding - padding_before
        
        return np.pad(audio, (padding_before, padding_after), 'constant')
        
    @staticmethod
    def pad_audios(audios: List[np.ndarray], sample_rates: List[int], desired_length: float) -> List[np.ndarray]:
        """
        Pads all the passed audios to the desired length in seconds.

        Parameters:
        - audios ([numpy array]): Audios to pad.
        - sample_rates ([int]): Sample rates of the audios.
        - desired_length (float): Desired duration in seconds.
        
        Returns:
        - The padded list of audio arrays.
        """
        return [AudioProcessing.pad_audio(audio, sample_rate, desired_length) \
                for audio, sample_rate in zip(audios, sample_rates)]
    
    @staticmethod
    def filter_audio(audio: np.ndarray, sample_rate: int, type: str = 'low', cutoff: int = 200) -> np.ndarray:
        """
        Applies a Butterworth filter to the audio data.

        Parameters:
        - audio (numpy array): Audio data to filter.
        - sample_rate (int): Sample rate of the audio.
        - type (str, optional): Filter type ('low', 'high', 'bandpass', etc.).
        - cutoff (int, optional): Cutoff frequency for the filter.

        Returns:
        - The filtered audio data array.
        """
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff / nyquist
        b, a = butter(5, normal_cutoff, btype=type, analog=False)
        filtered_audio = lfilter(b, a, audio)
        
        return filtered_audio

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sample_rate: int, target_sample_rate: int) -> np.ndarray:
        """
        Resamples audio from the original to the target sampling rate.

        Parameters:
        - audio (numpy array): Audio to resample.
        - orig_sr (int): Original sampling rate.
        - target_sr (int): Target sampling rate.

        Returns:
        - The resampled audio data array.
        """
        return librosa.resample(audio, orig_sr=orig_sample_rate, target_sr=target_sample_rate)
    
    @staticmethod
    def resample_audios(audios: List[np.ndarray], orig_sample_rates: List[int], target_sample_rate: int) -> List[np.ndarray]:
        """
        Resamples audio from the original to the target sampling rate.

        Parameters:
        - audios ([numpy arrays]): Audios to resample.
        - orig_sample_rates ([int]): Original sampling rate.
        - target_sample_rate (int): Target sampling rate.

        Returns:
        - The list of resampled audio arrays.
        """
        return [AudioProcessing.resample_audio(audio, orig_sample_rate, target_sample_rate) \
                for audio, orig_sample_rate in zip(audios, orig_sample_rates)]