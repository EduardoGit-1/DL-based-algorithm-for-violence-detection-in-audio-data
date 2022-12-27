from tensorflow.keras.layers import Dense, Conv2D,  MaxPool2D, Flatten, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, Dropout
from sklearn.utils import class_weight
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy 
import tensorflow as tf
import pandas as pd 
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", )
        # self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", )
        # self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            # self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        # x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        # x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            #res = self.res_bn(res)

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, last_activation="sigmoid", **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(64, (7, 7), strides=2,
                             padding="same", kernel_initializer="he_normal")
        # self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(pool_size=(2, 2), strides=2, padding="same")
        self.res_1_1 = ResnetBlock(64)
        self.res_1_2 = ResnetBlock(64)
        self.res_2_1 = ResnetBlock(128, down_sample=True)
        self.res_2_2 = ResnetBlock(128)
        self.res_3_1 = ResnetBlock(256, down_sample=True)
        self.res_3_2 = ResnetBlock(256)
        self.res_4_1 = ResnetBlock(512, down_sample=True)
        self.res_4_2 = ResnetBlock(512)
        #self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        self.fc_1 = Dense(128, activation='relu')
        self.drop_out_1 = Dropout(0.5)
        self.fc_2 = Dense(64, activation='relu')
        self.drop_out_2 = Dropout(0.5)
        self.fc_3 = Dense(32, activation='relu')
        self.fc_4 = Dense(num_classes, activation=last_activation)

    def call(self, inputs):
        out = self.conv_1(inputs)
        #out = self.init_bn(out)
        out = tf.nn.relu(out)
        out = self.pool_2(out)
        for res_block in [self.res_1_1, self.res_1_2, self.res_2_1, self.res_2_2, self.res_3_1, self.res_3_2, self.res_4_1, self.res_4_2]:
            out = res_block(out)
        #out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc_1(out)
        out = self.drop_out_1(out)
        out = self.fc_2(out)
        out = self.drop_out_2(out)
        out = self.fc_3(out)
        out = self.fc_4(out)
        return out

def f1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

    
if __name__ == "__main__":
    RUN_NUMBER = 1
    DURATION = 3
    STEP_SIZE = 3
    FORMAT_TYPE = "img"
    INPUT_TYPE = "cqt"
    USING_CLASS_WEIGHTS = True
    CLASS_WEIGHT_NAME_EXTENSION = "classweights_true" if USING_CLASS_WEIGHTS else "classweights_false"
    INPUT_SHAPE = (254, 254, 3)
    TRAINING_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/training/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    VALIDATION_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/validation/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    TESTING_DIR = "/home/goe2brg/DL_Violence_Detection_v7/data/{}_{}_{}win_{}step/testing/".format(INPUT_TYPE, FORMAT_TYPE, DURATION, STEP_SIZE)
    NAME = "ResNet18_{}_{}_V1_Run_{}_{}".format(INPUT_TYPE, FORMAT_TYPE, RUN_NUMBER, CLASS_WEIGHT_NAME_EXTENSION)
    MODEL_PATH_LOSS = "/home/goe2brg/DL_Violence_Detection_v7/models/ResNet18_Img_v1/model_weights/{}_{}.h5".format(NAME, "loss")
    MODEL_PATH_F1SCORE = "/home/goe2brg/DL_Violence_Detection_v7/models/ResNet18_Img_v1/model_weights/{}_{}.h5".format(NAME, "f1_score")
    MODELS_PATH = {"loss" : MODEL_PATH_LOSS, 
                    "f1_score": MODEL_PATH_F1SCORE}
    TRAINING_LOG_DIR = "/home/goe2brg/DL_Violence_Detection_v7/model_logs/training_logs/ResNet/{}_training.csv".format(NAME) #CHANGE DIR ON CLUSTER
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


    optimizer = SGD(learning_rate=LEARNING_RATE, decay=LEARNING_RATE/EPOCHS)
    model = ResNet18(1, last_activation="sigmoid")    
    model.build(input_shape = (None, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    optimizer = SGD(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
    # optimizer = Adam(lr=LEARNING_RATE)

    model.compile(optimizer=optimizer,
                loss = LOSS,
                metrics = METRICS)

    model.summary()

    history = model.fit(train_gen,
            validation_data = val_gen,
            epochs = EPOCHS,
            class_weight = CLASS_WEIGHTS,
            callbacks=CALLBACKS,
            verbose = 2,
            )
    
    pd.DataFrame.from_dict(history.history).to_csv(TRAINING_LOG_DIR, index=False)

    for key, weights in MODELS_PATH.items():
        model.load_weights(weights)
        evaluation_log_dir = "/home/goe2brg/DL_Violence_Detection_v7/model_logs/evaluation_logs/ResNet/{}_{}.xlsx".format(NAME, key)
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