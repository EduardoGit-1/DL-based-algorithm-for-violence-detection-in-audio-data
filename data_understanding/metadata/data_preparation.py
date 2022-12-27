import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None
import numpy as np
import json
import os


def load_json(file_path,type):
    with open(file_path, "r") as f:
        data = json.load(f)
        try:
            return data["audio_metadata"][type]
        except:
            return []


def get_speaker_data(file_path):
    speaker_data = load_json(file_path, "speaker")
    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for speaker in speaker_data:
        if "aggressiveness" in speaker:
            data["Label"].append(speaker["aggressiveness"])
            data["Type"].append("speaker_aggressiveness")
            data["Time_start"].append(speaker["time_start"])
            data["Time_end"].append(speaker["time_end"])
            data["Duration"].append(round(float(speaker["time_end"]) - float(speaker["time_start"]), 2))
            data["File_path"].append(file_path)
        else:
            try:
                for event in speaker["event"]:
                    data["Label"].append(event["description"])
                    data["Type"].append("speaker_event")
                    data["Time_start"].append(float(event["time_start"]))
                    data["Time_end"].append(float(event["time_end"]))
                    data["Duration"].append(round(float(speaker["time_end"]) - float(speaker["time_start"]), 2))
                    data["File_path"].append(file_path)
            except:
                print(file_path)
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def get_scene_data(json_path):
    scene_data = load_json(json_path, "scene_activities_noises")

    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for scene_noise in scene_data:
        if "event" in scene_noise:
            data["Label"].append(scene_noise["event"])
            data["Type"].append("scene_activities_noises")
            data["Time_start"].append(float(scene_noise["time_start"]))
            data["Time_end"].append(float(scene_noise["time_end"]))
            data["Duration"].append(round(float(scene_noise["time_end"]) - float(scene_noise["time_start"]), 2))
            data["File_path"].append(json_path)
        else:
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def get_backgroundnoise_data(json_path):
    background_noises = load_json(json_path, "background_noise")

    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for background_noise in background_noises:
        if "event" in background_noise:
            data["Label"].append(background_noise["event"])
            data["Type"].append("background_noise")
            data["Time_start"].append(float(background_noise["time_start"]))
            data["Time_end"].append(float(background_noise["time_end"]))
            data["Duration"].append(round(float(background_noise["time_end"]) - float(background_noise["time_start"]), 2))
            data["File_path"].append(json_path)
        else:
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def get_physical_violence_noises_data(json_path):
    phys_noises = load_json(json_path, "physical_violence_noises")

    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for phys_noise in phys_noises:
        if "event" in phys_noise:
            data["Label"].append(phys_noise["event"])
            data["Type"].append("physical_violence_noises")
            data["Time_start"].append(float(phys_noise["time_start"]))
            data["Time_end"].append(float(phys_noise["time_end"]))
            data["Duration"].append(round(float(phys_noise["time_end"]) - float(phys_noise["time_start"]), 2))
            data["File_path"].append(json_path)
        else:
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def get_silence_data(json_path):
    silence_data = load_json(json_path, "silence")
    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for silence in silence_data:
        data["Label"].append("silence")
        data["Type"].append("silence")
        data["Time_start"].append(float(silence["time_start"]))
        data["Time_end"].append(float(silence["time_end"]))
        data["Duration"].append(round(float(silence["time_end"]) - float(silence["time_start"]), 2))
        data["File_path"].append(json_path)

    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df

def get_scene_emotion_data(json_path):
    emotion_data = load_json(json_path, "scene_classification")

    data = {
        "File_path" : [],
        "Type" : [],
        "Time_start" : [],
        "Time_end": [],
        "Duration": [],
        "Label" : [],
    }
    for emotion in emotion_data:
        if "events" in emotion:
            data["Label"].append(emotion["events"])
            data["Type"].append("emotion")
            data["Time_start"].append(float(emotion["time_start"]))
            data["Time_end"].append(float(emotion["time_end"]))
            data["Duration"].append(round(float(emotion["time_end"]) - float(emotion["time_start"]), 2))
            data["File_path"].append(json_path)
        else:
            continue
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Time_start', 'Time_end'], ascending=[True, True])
    return df
def process_file(json_path):
    df_speaker = get_speaker_data(json_path)
    df_scene_data = get_scene_data(json_path)
    df_background_noise = get_backgroundnoise_data(json_path)
    df_phys_violence = get_physical_violence_noises_data(json_path)
    df_silence = get_silence_data(json_path)
    df_emotion = get_scene_emotion_data(json_path)

    return df_speaker, df_scene_data, df_background_noise, df_phys_violence, df_silence, df_emotion

def prepare_dataset(dataset_path, files_path, output_dir):
    df_structure = {
        "File_path" : [],
        "WAV_path" : [],
        "Time_start" : [],
        "Time_end" : [],
        "Duration" :  [],
        "Label" : []
    }
    results_df = pd.DataFrame(df_structure)
    for root, dirnames, filenames in os.walk(dataset_path):
            for f in filenames:
                if f.endswith('json'):
                    json_path = os.path.join(root, f)
                    df_data = process_file(json_path)
                    wav_file = f.replace(".json", "_center_top_wav_audio_ros.wav")
                    base_file_path = root.replace(os.sep, "/").split("/")
                    file_subfolder = base_file_path[-2] + "/" + base_file_path[-1]
                    current_file_path = files_path + file_subfolder
                    wav_file_path = os.path.join(current_file_path, wav_file).replace(os.sep, '/')
                    for df in df_data:
                        if len(df) > 0:
                            df["WAV_path"] = [wav_file_path] * len(df)
                            results_df = pd.concat([results_df, df], ignore_index=True)
    
    results_df.to_csv(output_dir)
    print(results_df[["Type"]].value_counts())
    print(results_df["Duration"].mean())
    print(results_df["Duration"].min())
    print(results_df["Duration"].max())

if __name__ == "__main__":
    DATASET_PATH = "fs/datasets/av/dataset_main/" #has to be changed 
    WAV_FILES_PATH = "/mnt/datasets_old/av/dataset_main/"
    OUTPUT_DIR = "data_understanding/metadata/dataset_results/Hanau02_03_test.csv"
    prepare_dataset(DATASET_PATH, WAV_FILES_PATH, OUTPUT_DIR)


