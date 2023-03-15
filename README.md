# Deep Learning-based algorithm for violence detection in audio data

This repository contains all the code used for my master thesis based on a Deep Learning approach for Violence Detection using auditory data and its visual representations.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) all the necessary python modules:

```bash
pip install -r requirements.txt
```

## Usage

Unfortunately the metadata and WAV files used on this project, could not be uploaded to the public so there's a need to make multiple changes in order to apply it to a similar use case. 

### Metadata preparation
Since metadata can differ a lot from dataset to dataset, its preparation will not be covered in this README. The only requirement in this phase is a CSV file with the following structure:
| File_Path         | Time_start      | Time_end        | Duration              | Label           |LabelID |
| ------------------|:---------------:|:---------------:|:---------------------:|:---------------:|:------:|
| /path/to/wav_file | n_float seconds | n_float seconds | time_end - time_start | str label name  | 0 or 1 |

*Note:* Each of the labels should be limited to a fixed duration (e.g 3 seconds). In case its duration is higher than the specified value it should be divided into 3 second windows, if its lower however, it will be later padded with silence on the audio preparation phase.

 ### Audio preparation
The ```data_preparation/audio_preprocessing/audio_util.py``` contains all the required code to extract the chosen audio features for this master's dissertation. Some changes can be done in this file in order to adapt it to your problem.
```python
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
```
In the table bellow, all the required changes are presented:

 | Variable         | Required change |
| :------------------:|---------------|
| ```DURATION``` | Should be changed to the window size selected on the metadata preparation phase.
| ```STEP_SIZE``` | Should be changed to the step size selected on the metadata preparation phase.
| ```SAMPLE_RATE``` | Number of samples in one second of audio. This ensures that all data has the same dimensions.
| ```MONO``` | ```True``` to transform the audio clip to mono (one channel). ```False``` if the original ammount of channels is required.
| ```DATASET_DIR``` | Should be changed to the metadata CSV file path.
| ```BASE_FEATURE_SAVE_DIR``` | Should be changed to the directory path where you wish to store the audio features.

Inside your ```BASE_FEATURE_SAVE_DIR``` create five subfolders: ```cqt```, ```mfcc```, ```stft```, ```mel``` and ```chroma```. These folders represent the location where each audio feature data will be stored. Inside of each of these create a folder for each set: ```training```, ```validation``` and ```testing``` which in turn should have two subfolders containing the two possible labels for your use case.

*Note:* This process could be automated by using the ```os``` module. Feel free to implement it in the ```save_feature()``` function present on the ```audio_util.py``` file.
``` 
project
├───data
│   └───mfcc
│       ├───testing
│       │   ├───non_violent      
│       │   └───violent
│       ├───training
│       │   ├───non_violent      
│       │   └───violent
│       └───validation
│           ├───non_violent
│           └───violent
├───data_preparation
│   ...
```
In order to run the audio transformations pipeline, use the following bash script:

```bash
cd data_preparation/audio_preprocessing
python audio_util.py
```
 ### Modelling
The ```models/{ARCHITECTURE_NAME}/main.py``` directory contains the implementation of each of the tested architectures as well as its training. Some variables can be changed here as well and are documented on the following table.
 | Variable         | Required changes |
| :------------------:|---------------|
| ```DURATION``` | Specifies the current run number, alter accordingly.
| ```WINDOW_SIZE``` | Should be changed to the window size selected on the metadata preparation phase.
| ```STEP_SIZE``` | Should be changed to the step size selected on the metadata preparation phase.
| ```INPUT_TYPE``` | This variable should be changed to the subfolder name of the audio feature that will be used as input (e.g "mfcc").
| ```TRAINING_DIR``` |  Alter the base path accordingly.
| ```VALIDATION_DIR``` | Alter the base path accordingly.
| ```TESTING_DIR``` |  Alter the base path accordingly.
| ```MODEL_PATH_LOSS``` | Change this variable to the directory path where the optmized model weights for the loss function should be stored.
| ```MODEL_PATH_F1SCORE``` | Change this variable to the directory path where the optmized model weights for F1-score should be saved.
| ```TRAINING_LOG_DIR``` | Change this variable to the directory path where the training CSV log file should be saved.
| ```evaluation_log_dir``` | Change this variable to the directory where evaluated model log file (on the respective test set) should be saved.

In order to run the model training, use the following bash script. Please alter the ```{ARCHITECTURE_NAME}``` to the desired architecture.

```bash
cd models/{ARCHITECTURE_NAME}
python main.py
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

