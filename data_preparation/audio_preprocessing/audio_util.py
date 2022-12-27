import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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
            signal = librosa.resample(signal, original_sr, self.sample_rate, res_type="kaiser_fast")
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

    def __init__(self, sample_rate, type_feature_name = "mel", type_yaxis = "mel"):
        self.sample_rate = sample_rate
        self.type_feature_name = type_feature_name
        self.type_yaxis = type_yaxis

    def extract(self, signal):
        mel_signal = librosa.feature.melspectrogram(y=signal, sr=self.sample_rate)[:-1]
        spectogram = np.abs(mel_signal)
        log_spec = librosa.amplitude_to_db(spectogram)
        return log_spec, self.type_feature_name, self.type_yaxis

class STFT_Extractor:

    def __init__(self, sample_rate, type_feature_name = "stft", type_yaxis = "log"):
        self.sample_rate = sample_rate
        self.type_feature_name = type_feature_name
        self.type_yaxis = type_yaxis

    def extract(self, signal):
        stft = librosa.stft(y=signal)
        stft = np.abs(stft)
        log_spec = librosa.amplitude_to_db(stft)
        return log_spec, self.type_feature_name, self.type_yaxis


class MFCCExtractor:

    def __init__(self, sample_rate, type_feature_name = "mfcc", type_yaxis = "mel"):
        self.sample_rate = sample_rate
        self.type_feature_name = type_feature_name
        self.type_yaxis = type_yaxis

    def extract(self, signal):
        mfccs_features = librosa.feature.mfcc(y=signal, sr=self.sample_rate, n_mfcc=40)
        return mfccs_features, self.type_feature_name, self.type_yaxis

class CQT_Extractor:
    
    def __init__(self, sample_rate, type_feature_name = "cqt", type_yaxis = "cqt_note"):
        self.sample_rate = sample_rate
        self.type_feature_name = type_feature_name
        self.type_yaxis = type_yaxis

    def extract(self, signal):
        cqt = np.abs(librosa.cqt(signal, sr=self.sample_rate))
        cqt = librosa.amplitude_to_db(cqt)
        return cqt, self.type_feature_name, self.type_yaxis

class Chroma_Extractor:
    
    def __init__(self, sample_rate, type_feature_name = "chroma", type_yaxis = "chroma"):
        self.sample_rate = sample_rate
        self.type_feature_name = type_feature_name
        self.type_yaxis = type_yaxis

    def extract(self, signal):
        chroma = np.abs(librosa.feature.chroma_stft(signal, sr=self.sample_rate))
        chroma = librosa.amplitude_to_db(chroma)
        return chroma, self.type_feature_name, self.type_yaxis

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


class Saver:

    def __init__(self, base_feature_save_dir, duration, step_size):
        self.base_feature_save_dir = base_feature_save_dir
        self.duration = duration
        self.step_size = step_size

    def save_feature(self, feature, type_feature, file_path, offset, time_end, label, type_df, format_type, y_axis="mel"):
        feature_save_dir = "{}/{}_{}_{}win_{}step/{}/{}/".format(self.base_feature_save_dir, type_feature, format_type, self.duration, self.step_size, type_df, label)
        save_path = self._generate_save_path(feature_save_dir, file_path, offset, time_end, format_type)
        if format_type == "img": 
            self.save_img(feature, save_path, y_axis) 
        else : 
            self.save_npy(feature, save_path)

    def save_img(self, feature, save_path, y_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        librosa.display.specshow(feature, x_axis = "time", y_axis=y_axis)
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0)
        plt.clf()
        plt.close("all")

    def save_npy(self, feature, save_path):
        feature = feature[..., np.newaxis]
        np.save(save_path, feature)

    def _generate_save_path(self, feature_save_dir, file_path, offset, time_end, format_type):
        format = "png" if format_type == "img" else "npy"
        ending_str = "_{}_{}.{}".format(offset, time_end, format)
        file_name = os.path.split(file_path)[1][:-4] + ending_str
        save_path = feature_save_dir + file_name
        return save_path


class PreProcessingPipeline:

    def __init__(self, loader, padder, feature_extractors, saver, normaliser, format_type):
        self.loader = loader
        self.padder = padder
        self.feature_extractors = feature_extractors
        self.normaliser = normaliser
        self.saver = saver
        self.format_type = format_type

    def _extract_feature(self, feature_extractor, signal, file_path, offset, time_end, label, type_df):
        feature, type_feature_name, type_yaxis = feature_extractor.extract(signal)
        feature = self.normaliser.normalise(feature)
        self.saver.save_feature(feature, type_feature_name, file_path, offset, time_end, label, type_df, self.format_type, type_yaxis)

    def _process_file(self, file_path, offset, time_end, label, type_df):
        signal, sr = self.loader.load(file_path, offset, time_end)
        signal = self.loader.resample(signal, sr)
        signal = self.padder.pad(signal)
        for feature_extactor in self.feature_extractors:
            self._extract_feature(feature_extactor, signal, file_path, offset, time_end, label, type_df)

    def get_train_val_df(self, df_input, stratify_colname='LabelID', frac_train = 0.8, frac_val = 0.1, frac_test = 0.1, random_state = None):
        
        X = df_input
        y = df_input[[stratify_colname]]


        df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                            y,
                                                            stratify=y,
                                                            test_size=(1.0 - frac_train),
                                                            random_state=random_state)

        relative_frac_test = frac_test / (frac_val + frac_test)
        
        df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                            y_temp,
                                                            stratify = y_temp,
                                                            test_size = relative_frac_test,
                                                            random_state = random_state)
        return df_train, df_val, df_test
     

    def process(self, dataframe, type):
        for row in dataframe.itertuples():
            try:
                self._process_file(row.File_path, row.Time_start, row.Time_end, row.Label, type)
            except:
                print(row.File_path, row.Time_start, row.Time_end)
                continue


if __name__ == "__main__":
    DURATION = 3
    STEP_SIZE = 3
    SAMPLE_RATE = 22050
    NUM_EXPECTED_SAMPLES = DURATION * SAMPLE_RATE
    MONO = True
    FORMAT_TYPE = "img"
    DATASET_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data_preparation/metadata_preprocessing/results/dataset_san_{}win_{}step_mnt.csv".format(DURATION, STEP_SIZE)
    BASE_FEATURE_SAVE_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data"
    TRAIN_DATASET_OUTPUT_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data_preparation/audio_preprocessing/datasets/processing_datasets/"
    TRAIN_DATASET_NAME = "train_dataset_san_{}win_{}step_mnt.csv".format(DURATION, STEP_SIZE)

    df = pd.read_csv(DATASET_DIR, index_col=0)
    loader = Loader(SAMPLE_RATE, MONO)
    padder = Padder(NUM_EXPECTED_SAMPLES)
    feature_extractors = [MelSpecExtractor(SAMPLE_RATE), MFCCExtractor(SAMPLE_RATE), CQT_Extractor(SAMPLE_RATE), STFT_Extractor(SAMPLE_RATE), Chroma_Extractor(SAMPLE_RATE)]
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(BASE_FEATURE_SAVE_DIR, DURATION ,STEP_SIZE)
    pipeline = PreProcessingPipeline(loader, padder, feature_extractors, saver, min_max_normaliser, FORMAT_TYPE)
    df_train, df_val, df_test = pipeline.get_train_val_df(df, "LabelID", frac_train = 0.8, frac_val=0.1, frac_test = 0.1,random_state=691)
    df_train.to_csv(TRAIN_DATASET_OUTPUT_DIR + TRAIN_DATASET_NAME)
    
    print(DATASET_DIR)
    print(BASE_FEATURE_SAVE_DIR)

    pipeline.process(df_train, "training")
    pipeline.process(df_val, "validation")
    pipeline.process(df_test, "testing")



