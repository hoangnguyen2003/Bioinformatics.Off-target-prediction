import dataset_utils

import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from keras.models import Model

import numpy as np
np.random.seed(42)

class OffTargetPrediction:
    def __init__(self,
                 dataset_dir,
                 batch_size,
                 lr
                 ):
        self.dataset_dir = dataset_dir
        self.lr = lr
        self.batch_size = batch_size

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            inputs = Input(shape=(1, 23, 4), name='main_input')
            conv_1 = Conv2D(10, (4, 1), padding='same', activation='relu')(inputs)
            conv_2 = Conv2D(10, (4, 2), padding='same', activation='relu')(inputs)
            conv_3 = Conv2D(10, (4, 3), padding='same', activation='relu')(inputs)
            conv_4 = Conv2D(10, (4, 5), padding='same', activation='relu')(inputs)

            conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

            bn_output = BatchNormalization()(conv_output)

            pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

            flatten_output = Flatten()(pooling_output)

            x = Dense(100, activation='softmax')(flatten_output)
            x = Dense(23, activation='softmax')(x)
            x = keras.layers.Dropout(rate=0.15)(x)

            prediction = Dense(2, activation='softmax', name='main_output')(x)

            self.model = Model(inputs, prediction)

            adam_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

            self.model.compile(loss='binary_crossentropy', optimizer = adam_opt)
        self.model.summary()

    def get_data(self):
        ds = dataset_utils.Dataset(self.dataset_dir).get_final_ds()
        self.X_train, self.y_train, self.X_test, self.y_test = ds

    def train(self, epochs):
        self.get_data()
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size, epochs=epochs,
                       shuffle=True
                       )
    
    def validate(self):
        pass