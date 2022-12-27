import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import librosa
import numpy as np

import os
from sklearn import preprocessing


def prepare_dataset(data, output_dir):
    df_structure = {
        "File_path" : [],
        "Sample_Rate" : [],
    }
    df = pd.DataFrame(df_structure)
    for row in data.itertuples():
        try:
            sr = librosa.get_samplerate(row.WAV_path)
        except:
            sr = "corrupted"
        f_df = pd.DataFrame({
            "File_path" : [row.WAV_path],
            "Sample_Rate": [sr]
        })
        df = pd.concat([df, f_df])
    df.to_csv(output_dir)

                    

if __name__ == "__main__":
    DATASET_PATH = "/home/goe2brg/DL_Violence_Detection_v3/data_understanding/audio_files/sample_rate_info/Hanau02_03_test.csv" #has to be changed 
    OUTPUT_DIR = "/home/goe2brg/DL_Violence_Detection_v3/data_understanding/audio_files/sample_rate_info/results/sr_dataset.csv"
    df = pd.read_csv(DATASET_PATH)
    scene_noises_df = df.loc[df["Type"] == "scene_activities_noises"]
    scene_noises_df = scene_noises_df.drop_duplicates(subset=['WAV_path'])
    
    prepare_dataset(scene_noises_df, OUTPUT_DIR)


