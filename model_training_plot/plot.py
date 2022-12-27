import matplotlib.pyplot as plt
import pandas as pd

def plot_hist(df, title, out_dir):
    plt.plot(df['acc'])
    plt.plot(df['val_acc'])
    plt.plot(df['loss'])
    plt.plot(df['val_loss'])
    plt.title(title)
    plt.xlabel('Epoch')
    plt.legend(['train accuracy', 'val accuracy','train loss', 'val loss'], loc='upper left')
    plt.savefig(out_dir, dpi=1200)


AUDIO_FEATURE_NAME = "Constant-Q Transformation"
OUTPUT_DIR = "model_training_plot/models/VGG/" + AUDIO_FEATURE_NAME + ".png"
TRAIN_DF_PATH = r"D:\Projects\shared_repo\model_logs\training_logs\VGG\VGG_cqt_img_V1_Run_1_classweights_true_training.csv"
df = pd.read_csv(TRAIN_DF_PATH)
plot_hist(df, AUDIO_FEATURE_NAME, OUTPUT_DIR)