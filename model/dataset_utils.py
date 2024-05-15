import numpy as np
from numpy import loadtxt

class Dataset:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.code_dict = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1],
        }

    def preprocess_function(self, guide_seq, off_seq):
        gRNA_list = list(guide_seq)
        off_list = list(off_seq)
        pair_code = []
        for i in range(len(gRNA_list)):
            gRNA_base_code = self.code_dict[gRNA_list[i]]
            DNA_based_code = self.code_dict[off_list[i]]
            pair_code.append(list(np.bitwise_or(gRNA_base_code, DNA_based_code)))
        
        return np.array(pair_code).reshape(1, 1, 23, 4)

    def get_final_ds(self):
        dataset = loadtxt(self.dataset_dir, delimiter=',')
        # X = dataset[:,0:8]
        # y = dataset[:,8]
        # return dataloader
        return dataset