import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

seed = 42

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.file_names = ['SITE-Seq_offTarget_wholeDataset']
        self.file_column_dict = {
            'SITE-Seq_offTarget_wholeDataset': ('on_seq', 'off_seq', 'reads', 'on_seq')
        }

        self.code_dict = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1],
        }

    def load_data(self, filename):
        columns = self.file_column_dict[filename]
        data = pd.read_csv(self.dataset_dir)

        sgRNAs = data[columns[0]]
        DNAs = data[columns[1]]
        labels = data[columns[2]]

        sgRNAs = sgRNAs.apply(lambda sgRNA: sgRNA.upper())
        DNAs = DNAs.apply(lambda DNA: DNA.upper())
        labels = labels.apply(lambda label: int(label != 0))

        sgRNAs_new = []
        for index, sgRNA in enumerate(sgRNAs):
            sgRNA = list(sgRNA)
            sgRNA[-3] = DNAs[index][-3]
            sgRNAs_new.append(''.join(sgRNA))

        sgRNAs = pd.Series(sgRNAs_new)
        data = pd.DataFrame.from_dict({'sgRNAs':sgRNAs, 'DNAs':DNAs, 'labels':labels})
        return data[data.apply(lambda row: 'N' not in list(row['DNAs']), axis = 1)]

    def preprocess_function(self, guide_seq, off_seq):
        gRNA_list = list(guide_seq)
        off_list = list(off_seq)
        pair_code = []
        for i in range(len(gRNA_list)):
            gRNA_base_code = self.code_dict[gRNA_list[i]]
            DNA_based_code = self.code_dict[off_list[i]]
            pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
        return np.array(pair_code).reshape(1, 23, 4)

    def get_final_ds(self, num_classes):
        dataset = self.load_data(os.path.splitext(
            os.path.basename(self.dataset_dir))[0])
        train, val_test = train_test_split(dataset, test_size=0.2, random_state=seed)
        val, test = train_test_split(val_test, test_size=0.5, random_state=seed)

        X_train_encodings = np.array(train.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        X_val_encodings = np.array(val.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        X_test_encodings = np.array(test.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        
        return X_train_encodings, to_categorical(
            train['labels'], num_classes=num_classes), X_val_encodings, to_categorical(
                val['labels'], num_classes=num_classes), X_test_encodings, to_categorical(
                    test['labels'], num_classes=num_classes)