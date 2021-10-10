"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np
import joblib

from .base_dataset import BaseDataset

class MixedDataset(torch.utils.data.Dataset):
    def __init__(self, options, dataset=None, **kwargs):
        self.dataset_list = ['h36m', 'lsp-orig', 'mpii', 'coco', 'youtube', 'mpi-inf-3dhp']
        self.datasets = [BaseDataset(options, ds, **kwargs) for ds in self.dataset_list]
        
        total_length = sum([len(ds) for ds in self.datasets])
        for ds_name, ds in zip(self.dataset_list, self.datasets):
            print("{} Train Num: {}".format(ds_name, format(len(ds), ",")))
        print("Total Train Num: {}".format(format(total_length, ",")))
        length_itw = sum([len(ds) for ds in self.datasets[1:-1]])
        self.length = max([len(ds) for ds in self.datasets])

        # self.amass = joblib.load("data/vibe_db/amass_db.pt")
        # self.amass = self.amass['theta']
        """
        Data distribution inside each batch:
        30% H36M - 60% ITW - 10% MPI-INF
        """
        if dataset == "pw_3d":
            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                            .6*len(self.datasets[2])/length_itw,
                            .6*len(self.datasets[3])/length_itw, 
                            .6*len(self.datasets[4])/length_itw,
                            .6*len(self.datasets[5])/length_itw,
                            0.1]
        
        elif dataset == "h36m":
            self.partition = [1]

        elif dataset == "no_youtube":
            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                            .6*len(self.datasets[2])/length_itw,
                            .6*len(self.datasets[3])/length_itw, 
                            0.1]
        else:    
            self.partition = [.3, .6*len(self.datasets[1])/length_itw,
                            .6*len(self.datasets[2])/length_itw,
                            .6*len(self.datasets[3])/length_itw, 
                            .6*len(self.datasets[4])/length_itw,
                            0.1]
    
        self.partition = np.array(self.partition).cumsum()

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(len(self.partition)):
            if p <= self.partition[i]:
                return self.datasets[i][index % len(self.datasets[i])]

    def __len__(self):
        return self.length
