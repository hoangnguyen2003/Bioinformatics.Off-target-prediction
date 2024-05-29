import dataset_utils

import os
import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, auc, roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N

def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP / P

def roc_auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)

class OffTargetPrediction:
    def __init__(self,
                 dataset_dir,
                 model_name,
                 roc_image_name,
                 epochs,
                 batch_size,
                 lr,
                 retrain,
                 is_sampling,
                 is_loso
                 ):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.roc_image_name = roc_image_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.retrain = retrain
        self.num_classes = 2
        self.is_sampling = is_sampling
        self.is_loso = is_loso

        os.environ['PYTHONHASHSEED'] = str(seed)

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

        inputs = Input(shape=(1, 23, 4), name='main_input')
        conv_1 = Conv2D(10, (4, 1), padding='same', activation='relu')(inputs)
        conv_2 = Conv2D(10, (4, 2), padding='same', activation='relu')(inputs)
        conv_3 = Conv2D(10, (4, 3), padding='same', activation='relu')(inputs)
        conv_4 = Conv2D(10, (4, 5), padding='same', activation='relu')(inputs)

        conv_output = keras.layers.concatenate([conv_1, conv_2, conv_3, conv_4])

        bn_output = BatchNormalization()(conv_output)

        pooling_output = keras.layers.MaxPool2D(pool_size=(1, 5),
                                                strides=None, padding='same')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='softmax')(flatten_output)
        x = Dense(23, activation='softmax')(x)
        x = keras.layers.Dropout(rate=0.15)(x)

        prediction = Dense(self.num_classes, activation='softmax', name='main_output')(x)

        self.model = Model(inputs, prediction)

        adam_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

        if self.is_sampling or self.is_loso:
            # self.model.compile(loss='binary_crossentropy', optimizer = adam_opt, metrics=['acc'])
            self.model.compile(loss='binary_crossentropy', optimizer = adam_opt, metrics=['acc', roc_auc])
        else:
            self.model.compile(loss='binary_crossentropy', optimizer = adam_opt)
        self.model.summary()

    def get_data(self):
        if self.is_sampling and self.is_loso:
            self.data, self.sgRNA_list, self.dict_address, self.X_test, self.y_test = dataset_utils.Dataset(self.dataset_dir).get_final_ds3(num_classes=self.num_classes)
            positoin_address = []
            for i in self.sgRNA_list:
                print(i)
                address_index = [x for x in range(len(self.sgRNA_list)) if self.sgRNA_list[x] == i]
                positoin_address.append([i, address_index])
            self.dict_address = dict(positoin_address)
            print(2)

        elif self.is_sampling:
            ds = dataset_utils.Dataset(self.dataset_dir).get_final_ds2(num_classes=self.num_classes)
            self.train_negative, self.train_positive, self.val_negative, self.val_positive, self.test_negative, self.test_positive = ds
            self.num_batch = int(len(self.train_negative) / self.batch_size)
        else:
            ds = dataset_utils.Dataset(self.dataset_dir).get_final_ds(num_classes=self.num_classes)
            self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = ds

    def train_flow(self, Train_Negative, Train_Positive, batchsize):
        train_Negative = Train_Negative
        train_Positive = Train_Positive

        Num_Positive = len(train_Positive)
        Num_Negative = len(train_Negative)
        Index_Negative = [i for i in range(Num_Negative)]
        Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
        random.shuffle(Index_Negative)
        Total_num_batch = int(Num_Negative / batchsize)
        num_counter = 0
        X_input = []
        Y_input = []
        while True:
            for i in range(Total_num_batch):
                for j in range(batchsize):
                    X_input.append(train_Negative[Index_Negative[j + i*batchsize]])
                    Y_input.append(0)
                    X_input.append(train_Positive[Index_Positive[j]])
                    Y_input.append(1)
                    num_counter += 1
                    if num_counter == batchsize:
                        Y_input = to_categorical(Y_input, num_classes=2)
                        yield (np.array(X_input), np.array(Y_input))
                        X_input = []
                        Y_input = []
                        Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                        num_counter = 0

    def valid_flow(self, Test_Negative, Test_Positive, batchsize):
        valid_Negative = Test_Negative
        valid_Positive = Test_Positive

        Num_Positive = len(valid_Positive)
        Num_Negative = len(valid_Negative)
        Index_Negative = [i for i in range(Num_Negative)]
        Index_Positive = np.random.randint(0, Num_Positive, batchsize,dtype='int32')
        random.shuffle(Index_Negative)
        num_counter = 0
        X_input = []
        Y_input = []
        while True:
            for j in range(batchsize):
                X_input.append(valid_Negative[Index_Negative[j]])
                Y_input.append(0)
                X_input.append(valid_Positive[Index_Positive[j]])
                Y_input.append(1)
                num_counter += 1
                if num_counter == batchsize:
                    Y_input = to_categorical(Y_input, num_classes=2)
                    yield (np.array(X_input), np.array(Y_input))
                    X_input = []
                    Y_input = []
                    Index_Positive = np.random.randint(0, Num_Positive, batchsize, dtype='int32')
                    num_counter = 0

    def train(self, X_train=None, y_train=None, X_val=None, y_val=None):
        if self.is_sampling and self.is_loso:
            print(3)
            keys = self.dict_address.keys()

            ROC_Mean = [[0 for i in range(3)] for j in range(len(keys))]
            PRC_Mean = [[0 for i in range(3)] for j in range(len(keys))]
            sgRNA_num = 0

            for key in keys:
                sgRNA_num = sgRNA_num + 1
                print("Training for the %sth time"%sgRNA_num)
                print("Leave-one-sgRNA-out:", key)
                test_index = self.dict_address[key]
                val_negative = []
                val_positive = []
                train_negative = []
                train_positive = []
                for i in range(len(self.sgRNA_list)):
                    if i in test_index:
                        if np.float(self.data[i]['labels']) > 0.0:
                            val_positive.append(self.data[i])
                        else:
                            val_negative.append(self.data[i])
                    else:
                        if np.float(self.data[i]['labels']) > 0.0:
                            train_positive.append(self.data[i])
                        else:
                            train_negative.append(self.data[i])
                train_negative = np.array(train_negative)
                train_positive = np.array(train_positive)
                val_positive = np.array(val_positive)
                val_negative = np.array(val_negative)
                Xtest = np.array(Xtest)
                NUM_BATCH = int(len(train_negative) / self.batch_size)

                History = self.model.fit_generator(self.train_flow(train_negative, train_positive, self.batch_size),
                                     shuffle=True,
                                     validation_data=self.valid_flow(val_negative, val_positive, 45),
                                     validation_steps=1,
                                     steps_per_epoch=NUM_BATCH,
                                     epochs=self.epochs,
                                     callbacks=self.callbacks)

                y_score_ = self.model.predict(self.X_test)

                y_pred = np.argmax(y_score_, axis=1)
                y_score = y_score_[:, 1]

                eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
                eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
                eval_fun_types = [True, True, True, True, False, False]
                for index_f, function in enumerate(eval_funs):
                    if eval_fun_types[index_f]:
                        score = np.round(function(self.y_test, y_pred), 4)
                    else:
                        score = np.round(function(self.y_test, y_score), 4)
                    print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))
                    if eval_fun_names[index_f] == "ROC AUC":
                        ROC_Mean[sgRNA_num-1] = score
                    elif eval_fun_names[index_f] == "PR AUC":
                        PRC_Mean[sgRNA_num-1] = score
            ROC_Mean = np.array(ROC_Mean)
            PRC_Mean = np.array(PRC_Mean)
            print("The result of cross validation under Leave_one_sgRNA_out：")
            print("ROC_Mean=%0.3f" % (np.mean(ROC_Mean)))
            print("PRC_Mean=%0.3f",np.mean(PRC_Mean))
            
        elif self.is_sampling:
            History = self.model.fit(self.train_flow(self.train_negative, self.train_positive, self.batch_size),
                                     shuffle=True,
                                     validation_data=self.valid_flow(self.val_negative, self.val_positive, 45),
                                     validation_steps=1,
                                     steps_per_epoch=self.num_batch,
                                     epochs=self.epochs,
                                     callbacks=self.callbacks)
            print(History.history.keys())
            plt.plot(History.history['acc'])
            plt.plot(History.history['val_acc'])
            plt.title("model accuracy")
            plt.xlabel("epoch")
            plt.ylabel("Accuracy")
            plt.legend(['train','test'],loc='upper left')

            save_path = os.path.join("images", "")
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, self.roc_image_name + "_11.png"))

            plt.show()
            plt.plot(History.history['loss'])
            plt.plot(History.history['val_loss'])
            plt.title("model loss")
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.legend(['train','test'],loc='upper left')

            plt.savefig(os.path.join(save_path, self.roc_image_name + "_22.png"))

            plt.show()
            plt.plot(History.history['roc_auc'])
            plt.plot(History.history['val_roc_auc'])
            plt.title("model auc")
            plt.xlabel("epoch")
            plt.ylabel("Loss")
            plt.legend(['train','test'],loc='upper left')

            plt.savefig(os.path.join(save_path, self.roc_image_name + "_33.png"))

            plt.show()
            plt.plot(History.history['lr'])

            plt.savefig(os.path.join(save_path, self.roc_image_name + "_44.png"))

            plt.show()
        else:
            self.model.fit(X_train, y_train,
                        batch_size=self.batch_size, epochs=self.epochs,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=self.callbacks,
                        )
        self.model.save('SaveModel/' + self.model_name + '.h5')
    
    def plotOvoRoc(self, y, y_score_):
        label_binarizer = LabelBinarizer().fit(self.y_train)
        fpr_grid = np.linspace(0.0, 1.0, 1000)

        a_mask = y == 0
        b_mask = y == 1
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        idx_a = np.flatnonzero(label_binarizer.classes_ == 0)[0]
        idx_b = np.flatnonzero(label_binarizer.classes_ == 1)[0]

        fpr_a, tpr_a, _ = roc_curve(a_true, y_score_[ab_mask, idx_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, y_score_[ab_mask, idx_b])

        mean_tpr = np.zeros_like(fpr_grid)
        mean_tpr += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr /= 2
        mean_score = auc(fpr_grid, mean_tpr)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.plot(
            fpr_grid,
            mean_tpr,
            label=f"Mean (AUC = {mean_score :.2f})",
            linestyle=":",
            linewidth=4,
        )
        RocCurveDisplay.from_predictions(
            a_true,
            y_score_[ab_mask, idx_a],
            ax=ax,
            name="Off-target as negative class",
        )
        RocCurveDisplay.from_predictions(
            b_true,
            y_score_[ab_mask, idx_b],
            ax=ax,
            name="Off-target as positive class",
            plot_chance_level=True,
        )
        ax.set(
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title="Receiver operating characteristic curve",
        )

        save_path = os.path.join("images", "")
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, self.roc_image_name + ".png"))

    def validate(self, X, y):
        y_score_ = self.model.predict(X)
        y_pred = np.argmax(y_score_, axis=1)
        y_score = y_score_[:, 1]

        eval_funs = [accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score]
        eval_fun_names = ['Accuracy', 'F1 score', 'Precision', 'Recall', 'ROC AUC', 'PR AUC']
        eval_fun_types = [True, True, True, True, False, False]
        for index_f, function in enumerate(eval_funs):
            if eval_fun_types[index_f]:
                score = np.round(function(y, y_pred), 4)
            else:
                score = np.round(function(y, y_score), 4)
            print('{:<15}{:>15}'.format(eval_fun_names[index_f], score))

        self.plotOvoRoc(y, y_score_)
    
    def do_all(self):
        self.get_data()
        if self.retrain:
            if self.is_sampling:
                self.train()
            else:
                self.train(self.X_train, self.y_train, self.X_val, self.y_val)
        if not self.is_sampling or not self.is_loso:
            self.validate(self.X_test, self.y_test)