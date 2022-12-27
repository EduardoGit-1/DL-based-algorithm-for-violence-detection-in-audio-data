import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import random

matplotlib.use('Agg')

class Loader:

    def __init__(self, sample_rate, mono):  # offset being time_start and duration is the window_size(seconds)
        self.sample_rate = sample_rate
        self.mono = mono

    def load(self, file_path, offset, time_end):
        duration = time_end - offset
        signal, sr = librosa.load(file_path, offset=offset, duration=duration, mono=self.mono, res_type="kaiser_fast")
        return signal, sr
    
    def get_sample(self, signal, offset, time_end, original_sample_rate):
        offset_samples = offset * original_sample_rate
        duration_samples = (time_end - offset) * original_sample_rate
        return signal[offset_samples:offset_samples + duration_samples]
    
    def resample(self, signal, original_sr):
        if self.sample_rate != original_sr:
            signal = librosa.resample(signal, original_sr, self.sample_rate, res_type="kaiser_best")
        return signal

class Padder:  #
    def __init__(self, num_expected_samples, mode = "constant"):
        self.num_expected_samples = num_expected_samples
        self.mode = mode

    def is_padding_needed(self, len_arr):
        return True if self.num_expected_samples > len_arr else False

    def pad(self, array):  # padding on the end of the original array
        if self.is_padding_needed(len(array)):
            num_missing_samples = self.num_expected_samples - len(array)
            array = np.pad(array, (0, num_missing_samples), mode=self.mode)
        return array


class MelSpecExtractor:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def extract(self, signal):
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate)[:-1]
        spectogram = np.abs(mel_signal)
        log_spec = librosa.amplitude_to_db(spectogram, ref = np.max)
        return log_spec

class MFCCExtractor:

    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def extract(self, signal):
        mfccs_features = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=40)
        #in order to find out scaled feature we do mean of transpose of value
        return mfccs_features

class Saver:

    def __init__(self, feature_save_dir):
        self.feature_save_dir = feature_save_dir

    def save_feature(self, feature, file_path, offset, time, label,type):
        save_path = self._generate_save_path(file_path, label, offset, time, type)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        librosa.display.specshow(feature, x_axis="time", y_axis="mel")
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
        plt.clf()
        plt.close("all")

    def _generate_save_path(self, file_path, label, offset, time, type):
        ending_str = "_{}_{}".format(offset, time)
        file_name = os.path.split(file_path)[1][:-4]
        #save_path = self.feature_save_dir+ type+ label +"/" + file_name + ending_str + "_augmented.png"
        save_path = file_name + ending_str + "_augmented.png"
        return save_path



class DataAugmentation:

    def __init__(self, sr):
        self.sr = sr
        

    def add_white_noise(self, signal, noise_percentage_factor = 0.1):
        noise = np.random.normal(0, signal.std(), signal.size)
        augmented_signal = signal + noise * noise_percentage_factor
        return augmented_signal
    
    def random_gain(self, signal, min_factor=0.1, max_factor=0.12):
        gain_rate = random.uniform(min_factor, max_factor)
        augmented_signal = signal * gain_rate
        return augmented_signal
    
    def time_strecth(self, signal, strech_rate = 0.4):
        return librosa.effects.time_stretch(signal, strech_rate)
    
    def pitch_scale(self, signal, num_semitones = 2):
        return librosa.effects.pitch_shift(signal, self.sr, num_semitones)

class MinMaxNormaliser:

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        a = (array - array.min())
        b = (array.max() - array.min())
        norm_array = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

class PreProcessingPipeline:

    def __init__(self, loader, padder, feature_extractor, saver, normaliser, data_augmentation):
        self.loader = loader
        self.padder = padder
        self.feature_extractor = feature_extractor
        self.normaliser = normaliser
        self.saver = saver
        self.data_augmentation = data_augmentation
        self.current_file = None
        self.current_signal = None
        self.current_sr = None

    def _process_file(self, file_path, offset, time_end,label, type):
        signal,sr = self.loader.load(file_path, offset, time_end)
        signal = self.loader.resample(signal, sr)
        signal = self.padder.pad(signal)
        augmentation = random.randint(0,1)
        feature_orig = self.feature_extractor.extract(signal)
        feature_orig = self.normaliser.normalise(feature_orig)
        feature_pitch = self.feature_extractor.extract(self.data_augmentation.pitch_scale(signal))
        feature_pitch = self.normaliser.normalise(feature_pitch)
        feature_time_streched = self.feature_extractor.extract(self.data_augmentation.time_strecth(signal))
        feature_time_streched = self.normaliser.normalise(feature_time_streched)
        return feature_orig, feature_pitch, feature_time_streched
        # if augmentation == 0:
        #     signal = self.data_augmentation.pitch_scale(signal)
        # else:
        #     signal = self.data_augmentation.add_white_noise(signal)
    
        # feature = self.feature_extractor.extract(signal)
        # feature = self.normaliser.normalise(feature)
        #feature = feature[..., np.newaxis]
        #self.saver.save_feature(feature, file_path, offset, time_end, label, type)

    def process(self, dataframe, type):
        for row in dataframe.itertuples():
            try:
                self._process_file(row.File_path, row.Time_start, row.Time_end, row.Label, type)
            except:
                 print(row.File_path, row.Time_start, row.Time_end)
                 continue


if __name__ == "__main__":
    DURATION = 3
    SAMPLE_RATE = 22050
    NUM_EXPECTED_SAMPLES = DURATION * SAMPLE_RATE
    MONO = True

    FEATURE_SAVE_DIR = "mel_augment/"
    RANDOM_FILE = r"D:\Projects\DL_Violence_Pytorch\fs\datasets\av\dataset_main\Hanau02\i3\Hanau02_i3_026_center_top_wav_audio_ros.wav"

    
    loader = Loader(SAMPLE_RATE, MONO)
    padder = Padder(NUM_EXPECTED_SAMPLES)
    mel_extractor = MelSpecExtractor(SAMPLE_RATE)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    mfcc_extractor = MFCCExtractor(SAMPLE_RATE)
    saver = Saver(FEATURE_SAVE_DIR)
    data_augmentation = DataAugmentation(SAMPLE_RATE)
    pipeline = PreProcessingPipeline(loader, padder, mel_extractor, saver, min_max_normaliser, data_augmentation)
    #feature = pipeline._process_file(RANDOM_FILE, 52, "violent", "training/")

