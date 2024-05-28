import dataset_utils

import os
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, auc, roc_curve, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

class OffTargetPrediction:
    def __init__(self,
                 dataset_dir,
                 model_name,
                 roc_image_name,
                 epochs,
                 batch_size,
                 lr,
                 retrain,
                 ):
        self.dataset_dir = dataset_dir
        self.model_name = model_name
        self.roc_image_name = roc_image_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.retrain = retrain
        self.num_classes = 2

        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
                                                strides=(1, 5), padding='valid')(bn_output)

        flatten_output = Flatten()(pooling_output)

        x = Dense(100, activation='softmax')(flatten_output)
        x = Dense(23, activation='softmax')(x)
        x = keras.layers.Dropout(rate=0.15)(x)

        prediction = Dense(self.num_classes, activation='softmax', name='main_output')(x)

        self.model = Model(inputs, prediction)

        adam_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.model.compile(loss='binary_crossentropy', optimizer = adam_opt)
        self.model.summary()

    def get_data(self):
        ds = dataset_utils.Dataset(self.dataset_dir).get_final_ds(num_classes=self.num_classes)
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = ds

    def train(self, X_train, y_train, X_val, y_val):
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
            self.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.validate(self.X_test, self.y_test)