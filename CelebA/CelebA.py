from torchvision import transforms
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torch
import os
import pickle
BASE_DATA_DIR = "C:/Users/leily/OneDrive/Desktop/property/CelebA"
img_path="C:/Users/leily/OneDrive/Desktop/property/CelebA/archive/img_align_celeba/img_align_celeba"
import sys

# class CustomUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == '__main__':
#             module = 'CelebA'
#         return super().find_class(module, name)

PRESERVE_PROPERTIES = ['Smiling', 'Young', 'Male', 'Attractive']
SUPPORTED_PROPERTIES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
    'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
    'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
    'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
    'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
    'Wearing_Necktie', 'Young'
]
SUPPORTED_RATIOS = ["0.0", "0.1", "0.2", "0.3",
                    "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]

class CelebASubsetDataset(Dataset):
    def __init__(self, df, img_path, transform=None, target=2000, subset_size_range=(1, 10)):
        self.df = df
        self.df = self.df.reset_index()
        self.img_path = img_path
        self.transform = transform
        self.target = target
        self.subset_size_range = subset_size_range

        self.subsets = []
        self.generate_subsets()
    def generate_subsets(self):
        #print(self.df.head())
        idx_pool = list(self.df.index)

        # To keep track of the female ratio counts.
        ratio_counts = {i/10: 0 for i in range(0, 11)}

        # Modify the loop condition to check all ratios
        while not all(count >= self.target for count in ratio_counts.values()):
            # Randomly choose a subset size.
            subset_size = np.random.randint(*self.subset_size_range)

            # Randomly sample a subset with possible repetition.
            subset_indices = np.random.choice(idx_pool, size=subset_size, replace=False)

            # Fetching image names and labels for the subset.
            subset_images = [os.path.join(self.img_path, self.df.iloc[int(idx)]['image_id']) for idx in subset_indices]
            subset_smiles = [self.df.iloc[idx]['Smiling'] for idx in subset_indices]
            female_ratio = round(sum([self.df.iloc[idx]['Male'] == 0 for idx in subset_indices]) / subset_size, 1)

            # If we have reached the target for this ratio, skip to the next iteration
            if ratio_counts[female_ratio] >= self.target:
                continue

            ratio_counts[female_ratio] += 1
            self.subsets.append((subset_images, subset_smiles, female_ratio))

            # Remove indices corresponding to the reached ratio to prevent generating more of them
            #if ratio_counts[female_ratio] == self.target:
            #    idx_pool = [idx for idx in idx_pool if round(int(self.df.iloc[idx]['Male'] == 0)) == female_ratio]
            # Printing details
            print(f"Added a subset of size {subset_size} with female ratio {female_ratio}. Current counts: {ratio_counts}")
            for img, smile in zip(subset_images, subset_smiles):
                print(f"Image: {os.path.basename(img)}, Smiling: {smile}")

        print(f"Generated {sum(ratio_counts.values())} subsets.")
        
            
    def __len__(self):
        return len(self.subsets)

    def __getitem__(self, idx):
        subset_images, subset_smiles, female_ratio = self.subsets[idx]
        images = [self.transform(Image.open(img_path).convert("RGB")) if self.transform else Image.open(img_path).convert("RGB") for img_path in subset_images]
        
        # Convert the labels into a tensor
        subset_smiles_tensor = torch.tensor(subset_smiles)
        #print(f"Subset images shape: {len(images)}, Subset smiles tensor shape: {subset_smiles_tensor.shape}")
        
        return images, subset_smiles_tensor, female_ratio
    
def custom_collate_fn(data):
    images_list, subset_smiles_tensor, female_ratio_list = zip(*data)

    # Find the maximum length among all image lists in the batch
    max_length = max([len(subset) for subset in images_list])

    # Pad each image list to the max_length
    padded_images_list = []
    for subset in images_list:
        padding_length = max_length - len(subset)
        # Assuming the images are 3x128x128 tensors
        padding = [torch.zeros((3, 128, 128)) for _ in range(padding_length)]
        padded_subset = torch.stack(list(subset) + padding, dim=0)
        padded_images_list.append(padded_subset)

    # Convert list of tensors to a single tensor
    images = torch.stack(padded_images_list, dim=0)
    
    # Handle the labels
    female_ratios = torch.tensor(female_ratio_list)
    
    # Padding the smile labels with NaNs
    smiles_padded = torch.full((len(images_list), max_length), fill_value=float('nan'))
    for i, subset_labels in enumerate(subset_smiles_tensor):
        smiles_padded[i, :len(subset_labels)] = subset_labels

    return images, (smiles_padded, female_ratios)



def get_dataloaders(df, force_generate=False):
    adv_train_file = 'celeba_adv_train.pt'
    adv_test_file = 'celeba_adv_test.pt'
    
    # Splitting the dataframe using list_eval_partition.csv
    partition_df = pd.read_csv("C:/Users/leily/OneDrive/Desktop/property/CelebA/archive/list_eval_partition.csv", index_col=0)
    train_df_full = df[partition_df["partition"] == 0]
    test_df_full = df[partition_df["partition"] == 2]

    train_df = train_df_full.sample(min(10000, len(test_df_full)))
    test_df = test_df_full.sample(min(1500, len(test_df_full)))
    adv_train_file = 'celeba_adv_train.pt'
    adv_test_file = 'celeba_adv_test.pt'

    # Check if data files already exist
    if not force_generate and os.path.exists(adv_train_file) and os.path.exists(adv_test_file):
        celeba_adv_train = torch.load(adv_train_file)
        celeba_adv_test = torch.load(adv_test_file)
    else:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        
        # Create instances of the dataset using train and test dataframes
        img_path = 'C:/Users/leily/OneDrive/Desktop/property/CelebA/archive/img_align_celeba/img_align_celeba'
        celeba_adv_train = CelebASubsetDataset(train_df, img_path, transform=transform, target=2000,subset_size_range=(2, 10))
        celeba_adv_test = CelebASubsetDataset(test_df, img_path, transform=transform, target=500, subset_size_range=(2, 10))

        # Save datasets
        torch.save(celeba_adv_train, adv_train_file)
        torch.save(celeba_adv_test, adv_test_file)

    # Create data loaders
    adv_trainloader = DataLoader(celeba_adv_train, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)
    adv_testloader = DataLoader(celeba_adv_test, batch_size=16, shuffle=True, collate_fn=custom_collate_fn)

    return adv_trainloader, adv_testloader

# 1. Load the CelebA Dataset Attributes
def loading():
    attrs_df = pd.read_csv(os.path.join(BASE_DATA_DIR, "archive/list_attr_celeba.csv"), index_col=0)
    print(attrs_df.head())
    # Convert -1 values to 0 for binary classification tasks
    attrs_df = attrs_df.replace(-1, 0)

    # 2. Run the get_dataloaders function
    adv_trainloader, adv_testloader = get_dataloaders(attrs_df,force_generate=False)
        
    return adv_trainloader,adv_testloader

import matplotlib.pyplot as plt
def visualize_subset(adv_trainloader, subset_idx=0):
    # Load the first batch from the training loader
    subsets, (smile_labels_list, female_ratios) = next(iter(adv_trainloader))


    # Get the specified subset and its labels
    subset_images = subsets[subset_idx]
    female_ratio = female_ratios[subset_idx]
    smile_labels_for_subset = smile_labels_list[subset_idx]
    
    # Check if smile_labels_for_subset is iterable, if not raise a more informative error.
    if not isinstance(smile_labels_for_subset, (list, tuple, torch.Tensor)):
        raise ValueError(f"Expected smile_labels_for_subset to be a list, tuple, or tensor, but got {type(smile_labels_for_subset)}.")
    # Plot the images from the chosen subset
    fig, axs = plt.subplots(1, len(subset_images), figsize=(15, 5))

    # Handle the special case where there's only one image
    if len(subset_images) == 1:
        axs.imshow(subset_images[0].permute(1, 2, 0))
        axs.set_title(f"Smile: {'Yes' if smile_labels_for_subset[0] == 1 else 'No'}")
        axs.axis('off')
    else:
        for i, (image, smile_label) in enumerate(zip(subset_images, smile_labels_for_subset)):
            axs[i].imshow(image.permute(1, 2, 0))
            axs[i].set_title(f"Smile: {'Yes' if smile_label == 1 else 'No'}")
            axs[i].axis('off')

    plt.suptitle(f"Subset Size: {len(subset_images)}, Female Ratio: {female_ratio}")
    plt.tight_layout()
    plt.show()


def main():
    # Move all the top-level code into this function
    adv_trainloader, adv_testloader = loading()
    visualize_subset(adv_testloader)
    # Load the CelebA Dataset Attributes
    attrs_df = pd.read_csv(os.path.join(BASE_DATA_DIR, "archive/list_attr_celeba.csv"), index_col=0)

    # Splitting the dataframe using list_eval_partition.csv
    partition_df = pd.read_csv("C:/Users/leily/OneDrive/Desktop/property/CelebA/archive/list_eval_partition.csv", index_col=0)
    train_df_full = attrs_df[partition_df["partition"] == 0]
    test_df_full = attrs_df[partition_df["partition"] == 2]

    print(f"Original number of training instances: {len(train_df_full)}")
    print(f"Original number of testing instances: {len(test_df_full)}")

if __name__ == '__main__':
    main()


