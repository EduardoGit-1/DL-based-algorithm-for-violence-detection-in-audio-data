from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from time import time

def depth_block(x, strides):   
    x = DepthwiseConv2D(3,strides=strides,padding='same',  use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def single_conv_block(x,filters):    
    x = Conv2D(filters, 1,use_bias=False)(x)
    x= BatchNormalization()(x)
    x = ReLU()(x)
    return x

def combo_layer(x,filters, strides):
    x = depth_block(x,strides)
    x = single_conv_block(x, filters)
    return x

def MobileNet(input_shape=(224,224,3),n_classes = 1, activation="sigmoid"):    
    input = Input ( input_shape)

    x = Conv2D(32,3,strides=(2,2),padding = 'same', use_bias=False) (input)
    x =  BatchNormalization()(x)
    x = ReLU()(x)

    x = combo_layer(x,64, strides=(1,1))    
    
    x = combo_layer(x,128,strides=(2,2))
    x = combo_layer(x,128,strides=(1,1))    
    
    x = combo_layer(x,256,strides=(2,2))
    x = combo_layer(x,256,strides=(1,1))   
    
    x = combo_layer(x,512,strides=(2,2))
    for _ in range(5):
        x = combo_layer(x,512,strides=(1,1))     
    
    x = combo_layer(x,1024,strides=(2,2))
    x = combo_layer(x,1024,strides=(1,1))     
    
    # x = GlobalAveragePooling2D()(x)     
    
    x = Flatten()(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)

    output = Dense(n_classes,activation=activation)(x)     
    model = Model(input, output)
    return model

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

    
if __name__ == "__main__":
    RUN_NUMBER = 2
    DURATION = 3
    STEP_SIZE = 3
    FORMAT_TYPE = "img"
    INPUT_TYPE = "mfcc"
    USING_CLASS_WEIGHTS = False
    CLASS_WEIGHT_NAME_EXTENSION = "classweights_true" if USING_CLASS_WEIGHTS else "classweights_false"
    INPUT_SHAPE = (254, 254, 3)
    TRAINING_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/training/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    VALIDATION_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/validation/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    TESTING_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/testing/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    NAME = "MobileNet_{}_{}_V1_Run_{}_{}".format(INPUT_TYPE, FORMAT_TYPE, RUN_NUMBER, CLASS_WEIGHT_NAME_EXTENSION)
    
    MODEL_PATH_LOSS = "/home/goe2brg/DL_Violence_Detection_v7/models/MobileNet_Img_v1/model_weights/{}_{}.h5".format(NAME, "loss")
    MODEL_PATH_F1SCORE = "/home/goe2brg/DL_Violence_Detection_v7/models/MobileNet_Img_v1/model_weights/{}_{}.h5".format(NAME, "f1_score")
    MODELS_PATH = {"loss" : MODEL_PATH_LOSS, 
                    "f1_score": MODEL_PATH_F1SCORE}
    TRAINING_LOG_DIR = "/home/goe2brg/DL_Violence_Detection_v7/model_logs/training_logs/MobileNet/{}_training.csv".format(NAME) #CHANGE DIR ON CLUSTER
    METRICS = [
        metrics.BinaryAccuracy(name="acc", threshold=0.5),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        f1_score,
        metrics.AUC(name='auc'),
        metrics.TrueNegatives(name="tn"),
        metrics.TruePositives(name="tp"),
        metrics.FalseNegatives(name="fn"),
        metrics.FalsePositives(name="fp"),
    ]
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 5e-3
    DECAY_RATE = LEARNING_RATE / EPOCHS
    LOSS = BinaryCrossentropy()

    model_checkpoint_callback_loss = ModelCheckpoint(
    filepath=MODEL_PATH_LOSS,
    save_best_only = True,
    save_weights_only = True,
    monitor='val_loss',
    verbose = 1,
    mode='min',
    )
    model_checkpoint_callback_f1_score = ModelCheckpoint(
    filepath=MODEL_PATH_F1SCORE,
    save_best_only = True,
    save_weights_only = True,
    monitor='val_f1_score',
    verbose = 1,
    mode='max',
    )

    CALLBACKS = [model_checkpoint_callback_loss, model_checkpoint_callback_f1_score]
    image_gen = ImageDataGenerator(rescale=1./255)
    train_gen = image_gen.flow_from_directory(TRAINING_DIR,
                                            target_size = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'binary',
                                            shuffle = True
                                            )

    val_gen = image_gen.flow_from_directory(VALIDATION_DIR,
                                            target_size = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'binary',
                                            shuffle = True
                                            )                                
    test_gen = image_gen.flow_from_directory(TESTING_DIR,
                                            target_size = (INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                            batch_size = BATCH_SIZE,
                                            class_mode = 'binary',
                                            #shuffle = True
                                            )


    class_weights = class_weight.compute_class_weight(
                class_weight = 'balanced',
                classes = np.unique(train_gen.classes), 
                y = train_gen.classes)
    
    CLASS_WEIGHTS = {
        0: class_weights[0],
        1: class_weights[1],
    }
    print(CLASS_WEIGHTS)
    

    print("Non_violent train data:", np.count_nonzero(train_gen.labels == 0),
            "Violent train data:",  np.count_nonzero(train_gen.labels == 1))

    print("Non_violent val data:", np.count_nonzero(val_gen.labels == 0),
            "Violent val data:",  np.count_nonzero(val_gen.labels == 1))
    
    print("Non_violent testing data:", np.count_nonzero(test_gen.labels == 0),
            "Violent testing data:",  np.count_nonzero(test_gen.labels == 1))
    
    print(TRAINING_DIR, VALIDATION_DIR, TESTING_DIR)    


    model = MobileNet(INPUT_SHAPE, 1, activation="sigmoid")

    #optimizer = Adam()
    optimizer = SGD(learning_rate=LEARNING_RATE, decay=LEARNING_RATE/EPOCHS)

    model.compile(optimizer=optimizer,
                loss = "binary_crossentropy",
                metrics = METRICS)
    
    history = model.fit(train_gen,
            validation_data = val_gen,
            epochs = EPOCHS,
            # class_weight = CLASS_WEIGHTS,
            callbacks = CALLBACKS,
            verbose = 2,
            )

    pd.DataFrame.from_dict(history.history).to_csv(TRAINING_LOG_DIR, index=False)

    for key, weights in MODELS_PATH.items():
        model.load_weights(weights)
        evaluation_log_dir = "/home/goe2brg/DL_Violence_Detection_v7/model_logs/evaluation_logs/MobileNet/{}_{}.xlsx".format(NAME, key)
        print("Evaluation using", key, "optimized weights")
        results = model.evaluate(test_gen, verbose = 2)
        evaluation_dict = {
            "Loss" : [results[0]],
            "Accurracy" : [results[1]],
            "Precision" : [results[2]],
            "Recall" : [results[3]],
            "F1_Score" : [results[4]],
            "AUC" : [results[5]],
            "TN" : [results[6]],
            "TP" : [results[7]],
            "FN" : [results[8]],
            "FP" : [results[9]],
        }
        pd.DataFrame(evaluation_dict).to_excel(evaluation_log_dir, index=False)