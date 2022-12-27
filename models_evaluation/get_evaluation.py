import pandas as pd
import numpy as np

CLASS_WEIGHTS = True
CLASS_WEIGHTS_NAME_EXTENSION = "classweights_true" if CLASS_WEIGHTS else "classweights_false"

BASE_RESNET_FOLDER = "model_logs/evaluation_logs/ResNet/"
BASE_MOBILENET_FOLDER = "model_logs/evaluation_logs/MobileNet/"
BASE_VGG_FOLDER = "model_logs/evaluation_logs/VGG/"

CURRENT_SCENARIO = "mfcc"
BASE_SCEARIO_PATH = "models_evaluation/scenario_{}/".format(CURRENT_SCENARIO)
RUN = 1

def get_eval(metric):
    vgg_model = BASE_VGG_FOLDER + "VGG_{}_img_V1_Run_{}_{}_{}.xlsx".format(CURRENT_SCENARIO, RUN, CLASS_WEIGHTS_NAME_EXTENSION, metric)
    resnet_model = BASE_RESNET_FOLDER + "ResNet18_{}_img_V1_Run_{}_{}_{}.xlsx".format(CURRENT_SCENARIO, RUN, CLASS_WEIGHTS_NAME_EXTENSION, metric)
    mobilenet_model = BASE_MOBILENET_FOLDER + "MobileNet_{}_img_V1_Run_{}_{}_{}.xlsx".format(CURRENT_SCENARIO, RUN, CLASS_WEIGHTS_NAME_EXTENSION, metric)
    
    df_dict = {
        "VGG-16" : pd.read_excel(vgg_model, engine='openpyxl',),
        "MobileNet" : pd.read_excel(mobilenet_model, engine='openpyxl',),
        "ResNet-16" : pd.read_excel(resnet_model, engine='openpyxl',)
    }

    results_dict = {
        "Model" : [],
        "Accuracy" : [], 
        "Precision" : [], 
        "Recall" : [], 
        "F1_Score" : [], 
        "AUC" : [],
        "Loss" : [],
    }

    for key, df in df_dict.items():
        results_dict["Model"].append(key)
        results_dict["Accuracy"].append(round(df["Accurracy"].values[0], 4))
        results_dict["Precision"].append(round(df["Precision"].values[0], 4))
        results_dict["Recall"].append(round(df["Recall"].values[0],4))
        f1_score = 2 *((df["Precision"].values[0] * df["Recall"].values[0]) / (df["Precision"].values[0] + df["Recall"].values[0]))
        results_dict["F1_Score"].append(round(f1_score, 4))
        results_dict["AUC"].append(round(df["AUC"].values[0], 4))
        results_dict["Loss"].append(round(df["Loss"].values[0], 4))

    pd.DataFrame.from_dict(results_dict).to_excel(BASE_SCEARIO_PATH + metric + ".xlsx")

get_eval("loss")
get_eval("f1_score")