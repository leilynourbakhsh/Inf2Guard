import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
sys.path.append("..")
import pickle



def to_numeric(data: np.ndarray, features: dict, label: str = '', single_bit_binary: bool = False) -> np.ndarray:
    """
    Takes an array of categorical and continuous mixed type data and encodes it in numeric data. Categorical features of
    more than 2 categories are turned into a one-hot vector and continuous features are kept standing. The description
    of each feature has to be provided in the dictionary 'features'. The implementation assumes python 3.7 or higher as
    it requires the dictionary to be ordered.

    :param data: (np.ndarray) The mixed type input vector or matrix of more datapoints.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle to encode binary features in a single bit instead of a 2-component 1-hot.
    :return: (np.ndarray) The fully numeric data encoding.
    """
    num_columns = []
    n_features = 0
    for i, key in enumerate(list(features.keys())):
        if features[key] is None:
            num_columns.append(np.reshape(data[:, i], (-1, 1)))
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            num_columns.append(np.reshape(np.array([int(str(val) == str(features[key][-1])) for val in data[:, i]]), (-1, 1)))
        else:
            sub_matrix = np.zeros((data.shape[0], len(features[key])))
            col_one_place = [np.argwhere(np.array(features[key]) == str(val)) for val in data[:, i]]
            for row, one_place in zip(sub_matrix, col_one_place):
                row[one_place] = 1
            num_columns.append(sub_matrix)
        n_features += num_columns[-1].shape[-1]
    pointer = 0
    num_data = np.zeros((data.shape[0], n_features))
    for column in num_columns:
        end = pointer + column.shape[1]
        num_data[:, pointer:end] = column
        pointer += column.shape[1]
    return num_data.astype(np.float32)


def to_categorical(data: np.ndarray, features: dict, label: str = '', single_bit_binary=False, nearest_int=True) -> np.ndarray:
    """
    Takes an array of matrix of more datapoints in numerical form and turns it back into mixed type representation.
    The requirement for a successful reconstruction is that the numerical data was generated following the same feature
    ordering as provided here in the dictionary 'features'.

    :param data: (np.ndarray) The numerical data to be converted into mixed-type.
    :param features: (dict) A dictionary containing each feature of data's description. The dictionary's items have to
        be of the form: ('name of the feature', ['list', 'of', 'categories'] if the feature is categorical else None).
        The dictionary has to be ordered in the same manner as the features appear in the input array.
    :param label: (str) The name of the feature storing the label.
    :param single_bit_binary: (bool) Toggle if the binary features have been encoded in a single bit instead of a
        2-component 1-hot.
    :param nearest_int: (bool) Toggle to round to nearest integer.
    :return: (np.ndarray) The resulting mixed type data array.
    """
    cat_columns = []
    pointer = 0
    for key in list(features.keys()):
        if features[key] is None:
            if nearest_int:
                cat_columns.append(np.floor(data[:, pointer] + 0.5))
            else:
                cat_columns.append(data[:, pointer])
            pointer += 1
        elif len(features[key]) == 2 and (single_bit_binary or key == label):
            cat_columns.append([features[key][max(min(int(val + 0.5), 1), 0)] for val in data[:, pointer]])
            pointer += 1
        else:
            start = pointer
            end = pointer + len(features[key])
            hot_args = np.argmax(data[:, start:end], axis=1)
            cat_columns.append([features[key][arg] for arg in hot_args])
            pointer = end
    cat_array = None
    for cat_column in cat_columns:
        if cat_array is None:
            cat_array = np.reshape(np.array(cat_column), (data.shape[0], -1))
        else:
            cat_array = np.concatenate((cat_array, np.reshape(np.array(cat_column), (data.shape[0], -1))), axis=1)
    return cat_array


class ADULT(BaseDataset):

    def __init__(self, name='ADULT', single_bit_binary=False, device='cpu', random_state=42):
        super(ADULT, self).__init__(name=name, device=device, random_state=random_state)

        self.features = {
            'age': None,
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                          'Without-pay', 'Never-worked'],
            'fnlwgt': None,
            'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                          '9th', '7th-8th', '12th', 'Masters',
                          '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
            'education-num': None,
            'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                               'Married-spouse-absent', 'Married-AF-spouse'],
            'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                           'Prof-specialty', 'Handlers-cleaners',
                           'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                           'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
            'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
            'sex': ['Female', 'Male'],
            'capital-gain': None,
            'capital-loss': None,
            'hours-per-week': None,
            'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                               'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                               'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
                               'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                               'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                               'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                               'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                               'Holand-Netherlands'],
            'salary': ['>50K', '<=50K']
        }

        self.single_bit_binary = single_bit_binary
        self.label = 'salary'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        train_data_df = pd.read_csv('textdata/ADULT/adult.data', delimiter=', ', names=list(self.features.keys()), engine='python')
        test_data_df = pd.read_csv('textdata/ADULT/adult.test', delimiter=', ', names=list(self.features.keys()), skiprows=1, engine='python')

        train_data = train_data_df.to_numpy()
        test_data = test_data_df.to_numpy()

        # drop missing values
        # note that the category never worked always comes with a missing value for the occupation field, hence this
        # step effectively removes the never worked category from the dataset
        train_rows_to_keep = [not ('?' in row) for row in train_data]
        test_rows_to_keep = [not ('?' in row) for row in test_data]
        train_data = train_data[train_rows_to_keep]
        test_data = test_data[test_rows_to_keep]

        # remove the annoying dot from the test labels
        for row in test_data:
            row[-1] = row[-1][:-1]

        # convert to numeric features
        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)

        # split features and labels
        Xtrain, Xtest = train_data_num[:, :-1].astype(np.float32), test_data_num[:, :-1].astype(np.float32)
        ytrain, ytest = train_data_num[:, -1].astype(np.float32), test_data_num[:, -1].astype(np.float32)
        self.num_features = Xtrain.shape[1]

        # transfer to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()

    def load_gmm(self, base_path):
        # TODO: this functionality has to be extended to all classes implementing Base_dataset
        with open(base_path + '/ADULT/fitted_gmms/all_cont_gmm.sav', 'rb') as f:
            gmm = pickle.load(f)
        self.gmm_parameters = {
            'all': (torch.as_tensor(gmm.weights_, device=self.device),
                    torch.as_tensor(gmm.means_, device=self.device),
                    torch.as_tensor(gmm.covariances_, device=self.device))
        }
        for feature_name, (feature_type, _) in self.train_feature_index_map.items():
            if feature_type == 'cont':
                with open(base_path + f'/ADULT/fitted_gmms/{feature_name}_gmm.sav', 'rb') as f:
                    gmm = pickle.load(f)
                self.gmm_parameters[feature_name] = (torch.as_tensor(gmm.weights_, device=self.device),
                                                     torch.as_tensor(gmm.means_, device=self.device),
                                                     torch.as_tensor(gmm.covariances_, device=self.device))
        self.gmm_parameters_loaded = True
