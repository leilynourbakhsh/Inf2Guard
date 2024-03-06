import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import random
BASE_DATA_DIR = "C:\\Users\\leily\\OneDrive\\Desktop\\property\\archive\\archive"
SUPPORTED_RATIOS = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]

def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class BoneDataset(Dataset):
    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.processed:
            X = self.features[self.df['path'].iloc[index]]
        else:
            # X = Image.open(self.df['path'][index])
            X = Image.open(self.df['path'][index]).convert('RGB')
            if self.transform:
                X = self.transform(X)

        y = torch.tensor(int(self.df['label'][index]))
        gender = torch.tensor(int(self.df['gender'][index]))

        return X, y, (gender)
class BoneWrapper:
    def __init__(self, df_train, df_val, features=None, augment=False):
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = 224
        test_transform_list = [
            transforms.Resize((self.input_size, self.input_size)),
        ]
        train_transform_list = test_transform_list[:]

        post_transform_list = [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])]
        # Add image augmentations if requested
        if augment:
            train_transform_list += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(
                    shear=0.01, translate=(0.15, 0.15), degrees=5)
            ]

        train_transform = transforms.Compose(
            train_transform_list + post_transform_list)
        test_transform = transforms.Compose(
            test_transform_list + post_transform_list)

        if features is None:
            self.ds_train = BoneDataset(self.df_train, train_transform)
            self.ds_val = BoneDataset(self.df_val, test_transform)
        else:
            self.ds_train = BoneDataset(
                self.df_train, features["train"], processed=True)
            self.ds_val = BoneDataset(
                self.df_val, features["val"], processed=True)

    def get_loaders(self, batch_size, shuffle=False):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=shuffle, num_workers=2)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        val_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=shuffle, num_workers=2)

        return train_loader, val_loader


def stratified_df_split(df, second_ratio):
    # Get new column for stratification purposes
    def fn(row): return str(row.gender) + str(row.label)
    col = df.apply(fn, axis=1)
    df = df.assign(stratify=col.values)

    stratify = df['stratify']
    df_1, df_2 = train_test_split(
        df, test_size=second_ratio,
        stratify=stratify)

    # Delete remporary stratification column
    df.drop(columns=['stratify'], inplace=True)
    df_1 = df_1.drop(columns=['stratify'])
    df_2 = df_2.drop(columns=['stratify'])

    return df_1.reset_index(), df_2.reset_index()
def process_data(path, split_second_ratio=0.5):
    csv_path = "C:/Users/leily/OneDrive/Desktop/property/archive/archive/boneage-training-dataset.csv"
    df = pd.read_csv(csv_path)
    
    training_image_path = 'C:\\Users\\leily\\OneDrive\\Desktop\\property\\archive\\archive\\boneage-training-dataset\\boneage-training-dataset'
    df['path'] = df['id'].map(lambda x: os.path.join(training_image_path, '{}.png'.format(x)))

    df['gender'] = df['male'].map(lambda x: 0 if x else 1)

    # Binarize into bone-age <=132 and >132 (roughly-balanced split)
    df['label'] = df['boneage'].map(lambda x: 1 * (x > 132))
    df.dropna(inplace=True)

    # Drop temporary columns
    df.drop(columns=['male', 'id'], inplace=True)

    # Return stratified split
    return stratified_df_split(df, split_second_ratio)
# Get DF file
def get_df(split):
    if split not in ["victim", "adv"]:
        raise ValueError("Invalid split specified!")

    df_train = pd.read_csv(os.path.join(BASE_DATA_DIR, "%s/train.csv" % split))
    df_val = pd.read_csv(os.path.join(BASE_DATA_DIR, "%s/val.csv" % split))

    return df_train, df_val


# Load features file
def get_features(split):
    if split not in ["victim", "adv"]:
        raise ValueError("Invalid split specified!")

    # Load features
    features = {}
    features["train"] = torch.load(os.path.join(
        BASE_DATA_DIR, "%s/features_train.pt" % split))
    features["val"] = torch.load(os.path.join(
        BASE_DATA_DIR, "%s/features_val.pt" % split))

    return features


def useful_stats(df):
    print("%d | %.2f | %.2f" % (
        len(df),
        df["label"].mean(),
        df["gender"].mean()))


def comnined_data():
    #base = os.path.abspath(os.path.join(BASE_DATA_DIR, os.pardir))
    base = BASE_DATA_DIR
    df_victim, df_adv = process_data(base, split_second_ratio=0.33)
    # Merge the victim and adversary dataframes
    df = pd.concat([df_victim, df_adv])
    # Save these splits
    def save_split(df, split):
        useful_stats(df)
        print()

        # Get train-val splits
        train_df, test_df = stratified_df_split(df, 0.2)
        print(train_df.iloc[0:10])

        # Ensure directory exists
        dir_prefix = os.path.join(BASE_DATA_DIR, "data", split)
        ensure_dir_exists(dir_prefix)

        # Save train-test splits
        train_df.to_csv(os.path.join(dir_prefix, "train.csv"))
        test_df.to_csv(os.path.join(dir_prefix, "t.csv"))

    save_split(df, "data_combined")

class BonAgeDataset(Dataset):
    def __init__(self, df, target=2000):
        self.df = df
        self.data = []
        self.labels = []
        self.primary_labels = []  # Add a list to hold primary task labels for each subset
        self.target = target
        self.ratios = [i / 10 for i in range(2, 9)]
        self.generate_samples()

    def generate_samples(self):
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        count = {ratio: 0 for ratio in self.ratios}

        while min(count.values()) < self.target:
            n = random.randrange(20,50)
            subset = self.df.sample(n)
            ratio = round(subset['gender'].mean(), 1)
            if ratio in self.ratios and count[ratio] < self.target:
                primary_subset = self.df.loc[subset.index]  
                images = [transform(Image.open(path)) for path in subset['path']]
                subset_tensor = torch.stack(images)
                ratio_tensor = torch.tensor(ratio, dtype=torch.float)
                primary_labels_tensor = torch.tensor(primary_subset['label'].values,
                                                    dtype=torch.float)
                self.data.append(subset_tensor)
                self.labels.append(ratio_tensor)
                self.primary_labels.append(primary_labels_tensor)
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

    X_padded = torch.zeros(len(X), max_length, X[0].shape[1], X[0].shape[2], X[0].shape[3])
    for i in range(len(X)):
        X_padded[i, :lengths[i]] = X[i]

    y1 = torch.tensor(y1)

    y2_lengths = [sample.shape[0] for sample in y2]
    max_y2_length = max(y2_lengths)
    y2_padded = torch.full((len(X), max_y2_length), fill_value=float('nan'))
    for i in range(len(X)):
        y2_padded[i, :y2_lengths[i]] = y2[i].clone().detach()

    return X_padded, (y1, y2_padded)


def get_dataloaders(train_df, test_df, force_generate=False):
    adv_train_file = 'boneage_adv_train.pt'
    adv_test_file = 'boneage_adv_test.pt'
    if not force_generate and os.path.exists(adv_train_file) and os.path.exists(adv_test_file):
        boneage_adv_train = torch.load(adv_train_file)
        boneage_adv_test = torch.load(adv_test_file)
    else:
        # Create instances of the dataset using train and test dataframes
        boneage_adv_train = BonAgeDataset(train_df, target=2000)
        boneage_adv_test = BonAgeDataset(test_df, target=500)

        # Save datasets
        torch.save(boneage_adv_train, adv_train_file)
        torch.save(boneage_adv_test, adv_test_file)

    # Create data loaders
    adv_trainloader = DataLoader(boneage_adv_train, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
    adv_testloader = DataLoader(boneage_adv_test, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    return adv_trainloader, adv_testloader

if __name__ == "__main__":
    # Specify the path to your data
    path = os.path.join(BASE_DATA_DIR, 'boneage-training-dataset')

    df_train, df_val = process_data(path, split_second_ratio=0.5)

    # Get DataLoaders
    train_loader, test_loader = get_dataloaders(df_train, df_val, force_generate=True)

