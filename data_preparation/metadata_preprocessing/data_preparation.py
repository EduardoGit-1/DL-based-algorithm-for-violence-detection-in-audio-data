import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import math
import numpy as np
import json
import os
from sklearn import preprocessing



VIOLENT_LABELS = ["anomaly_arguing", "anomaly_conversation", "anomaly_fighting", "anomaly_interaction", "anomaly_talking", "anomaly_violence"]
NON_VIOLENT_LABELS = ["normal_arguing", "normal_conversation", "normal_interaction", "normal_nointeraction", "normal_talking", 
                        "normal_radio_music","radio_music", "talking"]


def get_label(label):
    if label in VIOLENT_LABELS:
        return "violent"
    elif label in NON_VIOLENT_LABELS:
        return "non_violent"
    else:
        return None

def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        if "scene_activities_noises" in data['audio_metadata']:
            return data["audio_metadata"]["scene_activities_noises"]
        else:
            return []

def get_data(json_path, wav_path):
    scene_data = load_json(json_path)

    data = {
        "File_path" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],

    }
    for scene_noise in scene_data:
        if "event" in scene_noise:
            label = get_label(scene_noise["event"])
            if label == None:
                continue
            data["Label"].append(label)
            data["Time_start"].append(float(scene_noise["time_start"]))
            data["Time_end"].append(float(scene_noise["time_end"]))
            data["Duration"].append(round(float(scene_noise["time_end"]) - float(scene_noise["time_start"]), 2))
            data["File_path"].append(wav_path)
        else:
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def prepare_dataset(dataset_path, window_size, step_size, output_dir):
    df_structure = {
        "File_path" : [],
        "Time_start" : [],
        "Time_end" : [],
        "Duration" :  [],
        "Label" : []
    }
    df = pd.DataFrame(df_structure)
    for root, dirnames, filenames in os.walk(dataset_path):
            for f in filenames:
                if f.endswith('json'):
                    json_path = os.path.join(root, f)
                    wav_file = f.replace(".json", "_center_top_wav_audio_ros.wav")
                    wav_file_path = os.path.join(root, wav_file).replace(os.sep, '/')
                    data = get_data(json_path, wav_file_path)
                    for row in data.itertuples():
                        start_time = row.Time_start
                        end_time = row.Time_end
                        while start_time < end_time:
                            if start_time + step_size < end_time:  
                                window_time_end = start_time + step_size
                                duration = window_size
                            else:
                                window_time_end = end_time
                                duration = end_time - start_time
                            duration = round(duration, 2)
                            data_df = pd.DataFrame({'File_path': [row.File_path], 'Time_start': [start_time],
                                        'Time_end': [window_time_end],'Duration': [duration],
                                        'Label': [row.Label]})

                            df = pd.concat([df,data_df],axis=0)
                            start_time += step_size
                    
    label_encoder = preprocessing.LabelEncoder()
    df['LabelID'] = label_encoder.fit_transform(df["Label"])
    df.to_csv(output_dir, index= False)
    print(df[["Label", "LabelID"]].value_counts()) #9822
    print(df["Duration"].mean())
    print(df["Duration"].min())
    print(df["Duration"].max())



if __name__ == "__main__":
    RANDOM_JSON_FILE = "fs/datasets/av/dataset_main/Hanau02/i3/Hanau02_i3_027.json"
    DATASET_PATH = "fs/datasets/av/dataset_main/" #has to be changed 
    OUTPUT_DIR = "data_preparation/metadata_preprocessing/results/dataset_san_3win_3step.csv"
    STEP_SIZE = 3
    WIN_SIZE = 3
    prepare_dataset(DATASET_PATH,WIN_SIZE, STEP_SIZE, OUTPUT_DIR)


