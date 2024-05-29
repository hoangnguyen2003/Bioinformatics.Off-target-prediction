import numpy as np
import os
import pandas as pd
import xlrd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

seed = 42

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.file_names = ['SITE-Seq_offTarget_wholeDataset',
                           'CIRCLE_seq_10gRNA_wholeDataset',
                           'off_data_twoset',
                           'changeseq',
                           'ttiss',
                           ]
        self.file_column_dict = {
            'SITE-Seq_offTarget_wholeDataset': ('on_seq', 'off_seq', 'reads', 'on_seq'),
            'CIRCLE_seq_10gRNA_wholeDataset': ('sgRNA_seq', 'off_seq', 'label', 'sgRNA_type'),
            'changeseq': ('On', 'Off', 'Active'),
            'ttiss': ('On', 'Off', 'Active'),
        }

        self.code_dict = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1],
            '_': [0, 0, 0, 0],
            '-': [0, 0, 0, 0],
        }

    def load_data(self, filename):
        if filename != "off_data_twoset":
            columns = self.file_column_dict[filename]
            data = pd.read_csv(self.dataset_dir)

            sgRNAs = data[columns[0]]
            DNAs = data[columns[1]]
            labels = data[columns[2]]
        else:
            xlrd.xlsx.ensure_elementtree_imported(False, None)
            xlrd.xlsx.Element_has_iter = True
            InputFile = xlrd.open_workbook(self.dataset_dir)
            sheet_hek293t = InputFile.sheet_by_name('hek293t')
            hek_sgRNA_list = sheet_hek293t.col_values(1)
            hek_DNA_list = sheet_hek293t.col_values(2)
            hek_labels_list = sheet_hek293t.col_values(3)

            sheet_K562 = InputFile.sheet_by_name('K562')
            K562_sgRNA_list = sheet_K562.col_values(1)
            K562_DNA_list = sheet_K562.col_values(2)
            K562_labels_list = sheet_K562.col_values(3)
            
            sgRNAs = pd.Series(hek_sgRNA_list + K562_sgRNA_list)
            DNAs = pd.Series(hek_DNA_list + K562_DNA_list)
            labels = pd.Series(hek_labels_list + K562_labels_list)

        sgRNAs = sgRNAs.apply(lambda sgRNA: sgRNA.upper())
        DNAs = DNAs.apply(lambda DNA: DNA.upper())
        labels = labels.apply(lambda label: int(label != 0))

        sgRNAs_new = []
        for index, sgRNA in enumerate(sgRNAs):
            sgRNA = list(sgRNA)
            sgRNA[-3] = DNAs[index][-3]
            sgRNAs_new.append(''.join(sgRNA))

        sgRNAs = pd.Series(sgRNAs_new)
        data = pd.DataFrame.from_dict({'sgRNAs': sgRNAs, 'DNAs': DNAs, 'labels': labels})
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
                val['labels'], num_classes=num_classes), X_test_encodings, test['labels']
    
    def get_final_ds2(self, num_classes):
        dataset = self.load_data(os.path.splitext(
            os.path.basename(self.dataset_dir))[0])
        train, val_test = train_test_split(dataset, test_size=0.2, random_state=seed)
        val, test = train_test_split(val_test, test_size=0.5, random_state=seed)

        for idx, f in enumerate((train, val, test)):
            negative = []
            positive = []
            for i in f:
                label_item = i['labels']
                if label_item != 0:
                    positive.append(i)
                else:
                    negative.append(i)
            if idx == 0:
                train_negative = negative
                train_positive = positive
            elif idx == 1:
                val_negative = negative
                val_positive = positive
            else:
                test_negative = negative
                test_positive = positive

        train_negative = np.array(train_negative.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        train_positive = np.array(train_positive.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        val_negative = np.array(val_negative.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        val_positive = np.array(val_positive.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        test_negative = np.array(test_negative.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        test_positive = np.array(test_positive.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())

        return train_negative, train_positive, val_negative, val_positive, test_negative, test_positive
    
    def get_final_ds3(self, num_classes):
        dataset = self.load_data(os.path.splitext(
            os.path.basename(self.dataset_dir))[0])
        train, val_test = train_test_split(dataset, test_size=0.2, random_state=seed)
        val, test = train_test_split(val_test, test_size=0.5, random_state=seed)
        dataset = pd.concat([train, val])

        sgRNAList = dataset['sgRNAs']
        data_list = []
        sgRNA_list = []
        position_address = [[] for i in range(len(sgRNAList))]
        index = 0
        print(sgRNAList)

        for index, ll in dataset.iterrows():
            sgRNA_item = ll['sgRNAs']
            data_item = ll
            for i in range(len(sgRNAList)):
                if sgRNA_item == sgRNAList[i]:
                    position_address[i].append(index)
            data_list.append(data_item)
            sgRNA_list.append(sgRNA_item)
            index += 1
        position = []
        for i in range(len(sgRNAList)):
            position.append([sgRNAList[i], position_address[i]])
        dict_address = dict(position)

        data_list['sgRNAs'] = np.array(data_list.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())

        X_test_encodings = np.array(test.apply(
            lambda row: self.preprocess_function(
                row['sgRNAs'], row['DNAs']), axis = 1).to_list())

        return data_list['sgRNA'], sgRNA_list, dict_address, X_test_encodings, test['labels']


        # train, val_test = train_test_split(dataset, test_size=0.2, random_state=seed)
        # val, test = train_test_split(val_test, test_size=0.5, random_state=seed)

        # for idx, f in enumerate((train, val, test)):
        #     data_list = []
        #     sgRNA_list = []
        #     position_address = [[] for i in range(len(sgRNAList))]
        #     index = 0
        #     for line in f:
        #         sgRNA_item = line['sgRNAs']
        #         data_item = line
        #         for i in range(len(sgRNAList)):
        #             if sgRNA_item == sgRNAList[i]:
        #                 position_address[i].append(index)
        #         data_list.append(data_item)
        #         sgRNA_list.append(sgRNA_item)
        #         index += 1
        #     position = []
        #     for i in range(len(sgRNAList)):
        #         position.append([sgRNAList[i],position_address[i]])
        #     dict_address = dict(position)
            
        #     if idx == 0:
        #         data_list_train = 
        #         sgRNA_list_train = 
        #         dict_address_train = 
        #     elif idx == 1:
        #     else:

        # train_negative = np.array(train_negative.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        # train_positive = np.array(train_positive.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        # val_negative = np.array(val_negative.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        # val_positive = np.array(val_positive.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        # test_negative = np.array(test_negative.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())
        # test_positive = np.array(test_positive.apply(
        #     lambda row: self.preprocess_function(
        #         row['sgRNAs'], row['DNAs']), axis = 1).to_list())

        # return train_negative, train_positive, val_negative, val_positive, test_negative, test_positive