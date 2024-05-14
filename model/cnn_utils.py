import dataset_utils, model_utils

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
        self.lr = lr
        self.batch_size = batch_size

        inputs = Input(shape=(1, 23, 4), name='main_input')
        conv_1 = Conv2D(10, (4, 1), padding='same', activation='relu')(inputs)
        conv_2 = Conv2D(10, (4, 2), padding='same', activation='relu')(inputs)
        conv_3 = Conv2D(10, (4, 3), padding='same', activation='relu')(inputs)
        conv_4 = Conv2D(10, (4, 5), padding='same', activation='relu')(inputs)

        conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5), strides=None, padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='relu')(flatten_output)
        x = Dense(23, activation='relu')(x)
        x = keras.layers.Dropout(rate=0.15)(x)

        prediction = Dense(2, name='main_output')(x)

        self.model = Model(inputs, prediction)

        adam_opt = keras.optimizers.adam(lr=self.lr)

        self.model.compile(loss='binary_crossentropy', optimizer = adam_opt)
        print(self.model.summary())

    def get_data(self):
        X_train = 
        y_train =

    def train(self, epochs):
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size, epochs=epochs,
                       shuffle=True
                       )
    
    def predict(guide_seq, off_seq):
        code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'C': [0, 0, 0, 1]}
        gRNA_list = list(guide_seq)
        off_list = list(off_seq)
        pair_code = []

        for i in range(len(gRNA_list)):
            if gRNA_list[i] == 'N':
                gRNA_list[i] = off_list[i]
            gRNA_base_code = code_dict[gRNA_list[i]]
            DNA_based_code = code_dict[off_list[i]]
            pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
        input_code = np.array(pair_code).reshape(1, 1, 23, 4)
        y_pred = loaded_model.predict(input_code).flatten()
        print(y_pred)
        return y_pred