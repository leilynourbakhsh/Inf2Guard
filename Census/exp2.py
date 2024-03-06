import pandas as pd
import numpy as np
import requests
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random

# loading data
BASE_DATA_DIR = "C:\\Users\\leily\\OneDrive\\Desktop\\property\\census_income_victim"

SUPPORTED_PROPERTIES = ["sex", "race", "none"]
PROPERTY_FOCUS = {"sex": "Female", "race": "White"}
SUPPORTED_RATIOS = ["0.0", "0.1", "0.2", "0.3",
                    "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]


# US Income dataset
class CensusIncome:
    def __init__(self, path=BASE_DATA_DIR, task='primary'):
        self.urls = [
            "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
        ]
        self.columns = [
            "age", "workClass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship",
            "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "income"
        ]
        self.dropped_cols = ["education", "native-country"]
        self.path = path
        self.download_dataset()
        # self.load_data(test_ratio=0.4)
        self.task = task
        self.load_data(test_ratio=0.5)

    # Download dataset, if not present
    def download_dataset(self):
        if not os.path.exists(self.path):
            print("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            print("==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    # Process, handle one-hot conversion of data etc
    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = x.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(x[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship']
        # Drop columns that do not help with task
        df = df.drop(columns=self.dropped_cols, axis=1)
        # Club categories not directly relevant for property inference
        df["race"] = df["race"].replace(
            ['Asian-Pac-Islander', 'Amer-Indian-Eskimo'], 'Other')
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        # Drop features pruned via feature engineering
        prune_feature = [
            "workClass:Never-worked",
            "workClass:Without-pay",
            "occupation:Priv-house-serv",
            "occupation:Armed-Forces"
        ]
        df = df.drop(columns=prune_feature, axis=1)
        return df

    # Create adv/victim splits, normalize data, etc
    def load_data(self, test_ratio, random_state=42, task='primary'):
        # Load train, test data
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'),
                                 names=self.columns, sep=' *, *',
                                 na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'),
                                names=self.columns, sep=' *, *', skiprows=1,
                                na_values='?', engine='python')

        # Add field to identify train/test, process together
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        # Take note of columns to scale with Z-score
        z_scale_cols = ["fnlwgt", "capital-gain", "capital-loss"]
        for c in z_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].mean()) / df[c].std()

        # Take note of columns to scale with min-max normalization
        minmax_scale_cols = ["age", "hours-per-week", "education-num"]
        for c in minmax_scale_cols:
            # z-score normalization
            df[c] = (df[c] - df[c].min()) / df[c].max()

        # Split back to train/test data
        self.train_df, self.test_df = df[df['is_train']
                                         == 1], df[df['is_train'] == 0]

        # Drop 'train/test' columns
        self.train_df = self.train_df.drop(columns=['is_train'], axis=1)
        self.test_df = self.test_df.drop(columns=['is_train'], axis=1)
        # Print the shape and columns
        # print("Training Data Shape:", self.train_df.shape)
        # print("Training Data Columns:", self.train_df.columns)

        # print("Test Data Shape:", self.test_df.shape)
        # print("Test Data Columns:", self.test_df.columns)

        # Convert DataFrames to PyTorch Datasets
        self.train_data = CensusDataset(self.train_df)
        self.test_data = CensusDataset(self.test_df)

        if self.task == 'primary':
            self.train_data = CensusDataset(self.train_df)
            self.test_data = CensusDataset(self.test_df)
        elif self.task == 'adversarial':
            self.train_data = CensusDatasetAdv(self.train_df, target_samples_per_label=100)
            self.test_data = CensusDatasetAdv(self.test_df, target_samples_per_label=100)
        else:
            raise ValueError(("Invalid task type. Expected 'primary' or 'adversarial'."))

    def get_data_loaders(self, batch_size=4):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader


class CensusDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.features = torch.tensor(df.drop(columns=['income']).values, dtype=torch.float)
        self.labels = torch.tensor(df['income'].values, dtype=torch.float)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



class CensusDatasetAdv(Dataset):
    def __init__(self, df, primary_df, target=2000):
        self.df = df
        self.primary_df = primary_df
        self.data = []
        self.labels = []
        self.primary_labels = []  # Add a list to hold primary task labels for each subset
        self.target = target
        self.ratios = [i / 10 for i in range(2, 6)]
        self.generate_samples()

    def generate_samples(self):
        count = {ratio: 0 for ratio in self.ratios}

        while min(count.values()) < self.target:
            n = random.randrange(2,50)
            subset = self.df.sample(n)
            ratio = round(subset['sex:Female'].mean(), 1)
            if ratio in self.ratios and count[ratio] < self.target:
                primary_subset = self.primary_df.loc[subset.index]  
                subset = subset.drop(columns=['income'])  
                #subset['placeholder'] = 0 
                subset_tensor = torch.tensor(subset.values, dtype=torch.float)  
                ratio_tensor = torch.tensor(ratio, dtype=torch.float)
                primary_labels_tensor = torch.tensor(primary_subset['income'].values,
                                                     dtype=torch.float)  # Convert primary task labels to tensor
                assert isinstance(subset_tensor, torch.Tensor)
                self.data.append(subset_tensor)
                self.labels.append(ratio_tensor)
                self.primary_labels.append(primary_labels_tensor)  # Append primary task labels to list
                count[ratio] += 1
                print(f'Added a subset of size {n} with label {ratio}. Current counts: {count}')

    def __getitem__(self, idx):
        return self.data[idx], (self.labels[idx], self.primary_labels[idx])
    
    def __len__(self):
        return len(self.data)

def custom_collate_fn(data):
    X, y = zip(*data)
    y1, y2 = zip(*y)
    lengths = [sample.shape[0] for sample in X]
    max_length = max(lengths)
    
    X_padded = torch.zeros(len(X), max_length, X[0].shape[1])
    for i in range(len(X)):
        X_padded[i, :lengths[i]] = X[i]
    
    y1 = torch.tensor(y1)  

    y2_padded = torch.full((len(X), max_length), fill_value=float('nan'))  # Padding with NaN or another appropriate value
    for i in range(len(X)):
       y2_padded[i, :len(y2[i])] = y2[i].clone().detach()

    return X_padded, (y1, y2_padded)
    
def get_dataloaders(force_generate=False):
    adv_train_file = 'census_income_adv_train.pt'
    adv_test_file = 'census_income_adv_test.pt'
    if not force_generate and os.path.exists(adv_train_file) and os.path.exists(adv_test_file):
        census_income_adv_train = torch.load(adv_train_file)
        census_income_adv_test = torch.load(adv_test_file)
    else:
        census_income_primary = CensusIncome(task='primary')
        census_income_primary.load_data(test_ratio=0.5)
        
        # Here we pass the census_income_primary.train_df as both df and primary_df
        census_income_adv_train = CensusDatasetAdv(census_income_primary.train_df, census_income_primary.train_df)
        
        # Do the same for test data
        census_income_adv_test = CensusDatasetAdv(census_income_primary.test_df, census_income_primary.test_df, target=500)

        # Save datasets
        torch.save(census_income_adv_train, adv_train_file)
        torch.save(census_income_adv_test, adv_test_file)

    # Create data loaders
    adv_trainloader = DataLoader(census_income_adv_train, batch_size=4, shuffle=True,collate_fn=custom_collate_fn)
    adv_testloader = DataLoader(census_income_adv_test, batch_size=4, shuffle=True,collate_fn=custom_collate_fn)

    return adv_trainloader, adv_testloader


def get_primary_dataloader(batch_size, train=True):
    # Instantiate the dataset class
    census_dataset = CensusIncome()

    if train:
        data = census_dataset.train_data
    else:
        data = census_dataset.test_data

    # Create a DataLoader instance
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    return dataloader