import numpy as np
import librosa
import pandas as pd
import torch
import torchaudio
import parselmouth
from pipeline.data_processing import DatasetManager 

## It seems to provide all the features necessary (f0, pitch, jitter, shimmer etc.)
## The documentation also provides very good explanation of the algorithms used to calculate the specific measurements.
## This can be helpful when writing the actual methods.

SAMPLE_RATE = 16000
DEVICE = 'mps'

class FeatureDatasetManager():
    def __init__(self):
        # self._sample_rate = 16000
        # self._device = 'mps'
        pass
    '''
    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    def set_device(self, device):
        self._device = device
    '''

    @staticmethod
    def create_dataset(vowel_dataset, phrase_dataset, padsize_mfcc, padsize_wav2vec2, n_mfcc=13, n_harmonics_f0=5):
        """
        Creates two new DataFrames containing the features extracted from the audio data.

        Returns:
        - vowel_feature_dataset: The vowel featureset.
        - phrase_feature_dataset: The phrase featureset.
        """
        vowel_feature_dataset = FeatureDatasetManager._create_vowel_feature_dataset(vowel_dataset, n_mfcc=n_mfcc, padsize_mfcc=padsize_mfcc, n_harmonics_f0=n_harmonics_f0)
        phrase_feature_dataset = FeatureDatasetManager._create_phrase_feature_dataset(phrase_dataset, n_mfcc=n_mfcc, padsize_mfcc=padsize_mfcc, padsize_wav2vec2=padsize_wav2vec2)

        return vowel_feature_dataset, phrase_feature_dataset

    @staticmethod
    def _create_vowel_feature_dataset(vowel_dataset, n_mfcc, padsize_mfcc, n_harmonics_f0):
        """
        Creates a new DataFrame containing MFCC, f0, jitter, shimmer and sex features.
        This feature dataset comprises of features to be interpreted and trained on by a CNN-like algorithm.
        It does not contain temporal features, as the MFCCs will be averaged (suggested by literature).

        Parameters:
        - vowel_dataset (DataFrame): The dataset containing the audio data and sex and diagnosis information.
        - n_mfcc (int): The number of MFCCs to extract.
        - padsize_mfcc (int): The size to pad the MFCC features.
        - n_harmonics_f0 (int): The number of harmonics to consider in the HPS calculation.

        Returns:
        - vowel_feature_dataset: The vowel featureset.
        """
        
        # Apply extract_mfcc() for each audio data row in the passed vowel_dataset.
        # TODO: Padsize should be the maximum feature lenght
        ## Idea: Calculate size of features by window of calculation and size of the audios.
        vowel_feature_dataset = pd.DataFrame(columns=['mfcc', 'f0', 'jitter', 'shimmer', 'sex', 'diagnosis'])
        audio_datas = vowel_dataset['audio_data']

        vowel_feature_dataset['mfcc'] = audio_datas.apply(lambda audio_data: AudioFeatureExtraction.extract_mfcc(audio_data, padsize=padsize_mfcc, n_mfcc=n_mfcc))
        # AVERAGE - vowel_feature_dataset['mfcc'] = audio_data.apply(lambda audio_data: np.average(AudioFeatureExtraction.extract_mfcc(audio_data, padsize=padsize_mfcc, n_mfcc=n_mfcc), axis=1))
        audio_features = audio_datas.apply(
            lambda audio_data: AudioFeatureExtraction.extract_praat_measurements(audio_data)
        ).apply(pd.Series)
        vowel_feature_dataset['f0'] = audio_features['f0_mean']
        vowel_feature_dataset['jitter'] = audio_features['local_jitter']
        # vowel_feature_dataset['jitter'] = audio_features['local_abs_jitter']
        vowel_feature_dataset['shimmer'] = audio_features['local_shimmer']
        # vowel_feature_dataset['shimmer'] = audio_features['local_db_shimmer']
        vowel_feature_dataset['sex'] = vowel_dataset['sex'].apply(lambda sex: AudioFeatureExtraction.label_sex(sex))
        vowel_feature_dataset['diagnosis'] = vowel_dataset['diagnosis'].apply(lambda diagnosis: AudioFeatureExtraction.label_diagnosis(diagnosis))

        return vowel_feature_dataset

    @staticmethod
    def _create_phrase_feature_dataset(phrase_dataset, n_mfcc, padsize_mfcc, padsize_wav2vec2):
        """
        Creates a new DataFrame containing MFCC and wav2vec 2.0 features.

        Parameters:
        - phrase_dataset (DataFrame): The dataset containing the audio data and diagnosis information.
        - n_mfcc (int): The number of MFCCs to extract.
        - padsize_mfcc (int): The size to pad the MFCC features.
        - padsize_wav2vec2 (int): The size to pad the wav2vec2 features.

        Returns:
        - phrase_feature_dataset: The vowel featureset.
        """
        phrase_feature_dataset = pd.DataFrame(columns=['mfcc', 'wav2vec2', 'diagnosis'])
        audio_data = phrase_dataset['audio_data']

        phrase_feature_dataset['mfcc'] = audio_data.apply(lambda audio_data: AudioFeatureExtraction.extract_mfcc(audio_data, padsize=padsize_mfcc, n_mfcc=n_mfcc))
        phrase_feature_dataset['wav2vec2'] = audio_data.apply(lambda audio_data: AudioFeatureExtraction.extract_wav2vec2(audio_data, padsize=padsize_wav2vec2))
        phrase_feature_dataset['diagnosis'] = phrase_dataset['diagnosis'].apply(lambda diagnosis: AudioFeatureExtraction.label_diagnosis(diagnosis))

        return phrase_feature_dataset

class AudioFeatureExtraction:
    ## QUESTION --------------
    # Data is just used to train the algorithm, without any relation to the speaker_id. 
    # This suggests that it might be of value to extend the dataset to use both SVD and VOICED. 
    # This question would have to be discussed later on (if deadline is July). For now, only data augmentation is used.
    wav2vec2_model = None

    def __init__(self):
        pass
    
    def extract_mfcc(audio, padsize, n_mfcc=13):
        '''
        Extracts the Mel-frequency Cepstral Coefficients (MFCCs) from the audio sample.

        Parameters:
        - audio (ndarray): The audio signal.
        - padsize (int): The size to pad the MFCC features.
        - n_mfcc (optional) (int): The number of MFCCs to extract.

        Returns:
        - padded_mfcc: The padded MFCC features.
        '''
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=n_mfcc)
        padded_mfcc = librosa.util.pad_center(mfcc, size=padsize, axis=1)

        return padded_mfcc
    
    def extract_wav2vec2(audio, padsize):
        '''
        Extracts the Wav2Vec 2.0 embeddings from the audio sample.

        Parameters:
        - audio (ndarray): The audio signal.
        - padsize (int): The size to pad the wav2vec2 features.

        Returns:
        - padded_wav2vec2_features: The padded wav2vec features.
        '''
        if AudioFeatureExtraction.wav2vec2_model is None:
            AudioFeatureExtraction.wav2vec2_model = torchaudio.pipelines.WAV2VEC2_BASE.get_model().to(DEVICE)

        audio_tensor = torch.tensor(np.float32(audio).reshape(1, -1)).to(DEVICE)
        with torch.inference_mode():
            wav2vec2_features, _ = AudioFeatureExtraction.wav2vec2_model.extract_features(audio_tensor)
        padded_wav2vec2_features = np.array([
            librosa.util.pad_center(tensor.numpy(force=True), size=padsize, axis=1) 
            for tensor in wav2vec2_features
        ])
        padded_wav2vec2_features = np.reshape(padded_wav2vec2_features, (12, padsize, 768))

        return padded_wav2vec2_features
    
    def extract_praat_measurements(audio, f0_min=75, f0_max=300):
        '''
        Extracts all the vowel features using parselmouth (Praat). The algorithms are tested and evaluated, thus providing a solid computation.

        Parameters:
        - audio (ndarray): The audio signal (numpy array).
        - f0_min (int, optional): The minimum fundamental frequency to consider for pitch extraction. Default is 75 Hz.
        - f0_max (int, optional): The maximum fundamental frequency to consider for pitch extraction. Default is 300 Hz.

        Returns:
        - dict: A dictionary containing the following keys and their corresponding extracted values:
            - 'f0_mean': Mean fundamental frequency (f0) in Hertz.
            - 'local_jitter': Local jitter in the audio signal.
            - 'local_abs_jitter': Local absolute jitter in the audio signal.
            - 'local_shimmer': Local shimmer in the audio signal.
            - 'local_db_shimmer': Local shimmer in decibels in the audio signal.
        '''
        sound = parselmouth.Sound(audio)
        pitch = parselmouth.praat.call(sound, "To Pitch", 0.0, f0_min, f0_max) # Create a praat pitch object
        f0_mean = parselmouth.praat.call(pitch, "Get mean", 0, 0, 'Hertz') # Get mean pitch
        point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
        local_jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        local_abs_jitter = parselmouth.praat.call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        local_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        local_db_shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return {
            'f0_mean': f0_mean,
            'local_jitter': local_jitter,
            'local_abs_jitter': local_abs_jitter,
            'local_shimmer': local_shimmer,
            'local_db_shimmer': local_db_shimmer
        }

    def label_sex(sex):
        '''
        Encode sex into binary label.

        Parameters:
        - sex (str): The sex label ('male' or 'female').

        Returns:
        - label: 0 for male, 1 for female.
        '''
        if sex == 'male':
            return 0 
        elif sex == 'female':
            return 1
        else:
            raise ValueError(f"{sex} is not supported. Supported sexes are: 'male' or 'female'.")

    def label_diagnosis(diagnosis):
        '''
        Encode diagnosis into binary label.

        Parameters:
        - diagnosis (str): The diagnosis label ('health' or 'path').

        Returns:
        - label: 0 for healthy, 1 for pathological.
        '''
        if diagnosis == 'health':
            return 0
        elif diagnosis == 'path':
            return 1
        else:
            raise ValueError(f"{diagnosis} is not supported. Supported diagnoses are: 'health' or 'path'.")