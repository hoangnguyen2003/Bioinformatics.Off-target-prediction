import dataset_utils

import numpy as np
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class OffTargetPrediction:
    def __init__(self,
                 dataset_dir,
                 model_name,
                 epochs,
                 batch_size,
                 lr,
                 retrain,
                 ):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

        if not retrain:
            self.model = load_model('SaveModel/' + self.model_name + '.h5')
            return

        eary_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                         min_delta=0.0001,
                                                         patience=5,
                                                         verbose=0,
                                                         mode='auto'
                                                         )
        self.callbacks = [eary_stopping]

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            inputs = Input(shape=(1, 23, 4), name='main_input')
            conv_1 = Conv2D(10, (1, 1), padding='same', activation='relu')(inputs)
            conv_2 = Conv2D(10, (1, 2), padding='same', activation='relu')(inputs)
            conv_3 = Conv2D(10, (1, 3), padding='same', activation='relu')(inputs)
            conv_4 = Conv2D(10, (1, 5), padding='same', activation='relu')(inputs)

            conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

            bn_output = BatchNormalization()(conv_output)

            pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5),
                                                    strides=None, padding='valid')(bn_output)

            flatten_output = Flatten()(pooling_output)

            x = Dense(100, activation='relu')(flatten_output)
            x = Dense(23, activation='relu')(x)
            x = keras.layers.Dropout(rate=0.15)(x)

            prediction = Dense(2, name='main_output')(x)

            self.model = Model(inputs, prediction)

            adam_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

            self.model.compile(loss='binary_crossentropy', optimizer = adam_opt)
        self.model.summary()

    def get_data(self):
        ds = dataset_utils.Dataset(self.dataset_dir).get_final_ds(num_classes=2)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = ds

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size, epochs=self.epochs,
                       shuffle=True,
                    #    validation_data=(X_val, y_val),
                    #    callbacks=self.callbacks,
                       )
        self.model.save('SaveModel/' + self.model_name + '.h5')
    
    def validate(self, X, y):
        a = np.array(self.X_train[:4]).reshape(4, 1, 23, 4)
        print(a)
        print(self.y_train[:4])
        print(self.model.predict(a, batch_size=4))
        y_score = self.model.predict(X)
        y_pred = np.argmax(y_score, axis=1)
        y_score = y_score[:, 1]

        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]

        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(y, y_pred), 4)
            else:
                score = np.round(function(y, y_score), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
    
    def do_all(self):
        self.get_data()
        self.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.validate(self.X_test, self.y_test)