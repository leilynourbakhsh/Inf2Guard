import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
sys.path.append("..")
import pickle


class ACTIVITY(BaseDataset):

    def __init__(self, name='ACTIVITY', single_bit_binary=False, device='cpu', random_state=42):
        super(ACTIVITY, self).__init__(name=name, device=device, random_state=random_state)

        self.features = {

        }

        self.single_bit_binary = single_bit_binary
        self.label = 'activity'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        train_x = pd.read_csv('textdata/Activity/train/X_train.csv', delimiter=',', engine='python')
        test_x = pd.read_csv('textdata/Activity/test/X_test.csv', delimiter=',', engine='python')
        train_y = pd.read_csv('textdata/Activity/train/y_train.csv', delimiter=',', engine='python')
        test_y = pd.read_csv('textdata/Activity/test/y_test.txt', delimiter=',', engine='python')

        train_x = train_x.to_numpy()
        test_x = test_x.squeeze().to_numpy()
        train_y = train_y.to_numpy()
        test_y = test_y.squeeze().to_numpy()

        self.num_features = train_x.shape[1]

        # transfer to torch
        self.Xtrain, self.Xtest = torch.tensor(train_x, dtype=torch.float32).to(self.device), torch.tensor(test_x, dtype=torch.float32).to(self.device)
        self.ytrain, self.ytest = torch.tensor(train_y, dtype=torch.long).squeeze().to(self.device), torch.tensor(test_y, dtype=torch.long).squeeze().to(self.device)

        # set to train mode as base
        self.train()
        self.ytrain = self.ytrain-1
        self.ytest = self.ytest-1
        self.Xtrain = self.Xtrain * 10
        self.Xtest = self.Xtest * 10
        # calculate the standardization statistics
        self._calculate_mean_std()
        del(train_x, train_y, test_x, test_y)
        # calculate the histograms and feature bounds
        #self._calculate_categorical_feature_distributions_and_continuous_bounds()

