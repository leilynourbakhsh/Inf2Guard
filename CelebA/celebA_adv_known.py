import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import os
import numpy as np
import torchmetrics
from sklearn.metrics import precision_score,recall_score
from CelebA  import  CelebASubsetDataset,loading
from celebA_models_known import ATK
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
total_epoch=100
lr=0.001

# getting data_loaders
adv_trainloader, adv_testloader=loading()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
atk_criterion = nn.CrossEntropyLoss().to(device)

def adjust_learning_rate(epoch, init_lr=0.001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr
def freeze(model):
    for param in model.parameters():
        param.requires_grad_(False) 
    model.eval()

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad_(True)
    model.train()

def get_precision_recall(pred, true, zero_division=0):
    _, pred = torch.max(pred, 1)  # Get predicted classes
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    precision = precision_score(true, pred, average='macro', zero_division=zero_division)
    recall = recall_score(true, pred, average='macro', zero_division=zero_division)
    return precision, recall

def map_labels_to_indices(labels_tensor):
    # Initialize a tensor with -1 to denote unmapped values
    mapped_labels = torch.full_like(labels_tensor, -1).long()
    
    # Mapping dictionary
    mapping = {0.0: 0, 0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5, 
               0.6: 6, 0.7: 7, 0.8: 8, 0.9: 9, 1.0: 10}

    # Apply the mapping using tensor operations
    for label_value, mapped_value in mapping.items():
        mapped_labels[labels_tensor == label_value] = mapped_value

    # Mask to retain only valid (non-NaN) values
    valid_mask = ~torch.isnan(labels_tensor)
    valid_mapped_labels = mapped_labels[valid_mask]

    return valid_mapped_labels


def train_FE_INF(FE, INF, data_train_loader, current_lr, device):
    INF.train()
    INF_optimizer = torch.optim.Adam(INF.parameters(), lr=current_lr, weight_decay=1e-4)
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    freeze(FE)  # Assuming this function is provided

    for ind, (X, (_, privlabels)) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            X, privlabels = X.float().cuda(), privlabels.cuda()


        batch_size, subsets, channels, height, width = X.shape
        all_agg_features = []


        for i in range(batch_size):
            # Filter out zero-padded images using the mask
            #valid_subset_images = X[i][non_zero_mask[i]]
            #if valid_subset_images.shape[0] == 0:  # If all images in subset are zero-padded
            #    continue
            subset_images = X[i, :, :, :, :]
            subset_features = FE(subset_images)
            #print(subset_features.shape)

            #agg_features = subset_features.mean(dim=0)
            all_agg_features.append(subset_features)

        all_agg_features_tensor = torch.stack(all_agg_features, dim=0)
        #flattened_features = all_agg_features_tensor.view(batch_size, -1)
        
        
        outputs = INF(all_agg_features_tensor)

        privlabels = privlabels.float()
        privlabels = map_labels_to_indices(privlabels)
        loss = atk_criterion(outputs, privlabels.view(-1))
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_predictions += privlabels.size(0)
        correct_predictions += (predicted == privlabels.view(-1)).sum().item()

        INF_optimizer.zero_grad()
        loss.backward()
        INF_optimizer.step()

        running_loss += loss.item()
        if ind % 100 == 0:
            current_accuracy = 100.0 * correct_predictions / total_predictions
            print(f'Batch {ind}/{len(data_train_loader)}, Loss: {loss.item()}, Accuracy: {current_accuracy:.2f}%')

    # Calculate overall epoch accuracy
    epoch_accuracy = 100.0 * correct_predictions / total_predictions
    print(f'Epoch Training Accuracy: {epoch_accuracy:.2f}%')

    return FE, INF


def eval_atk(FE, INF, atk_test_dl, atk_criterion, device):
    FE.eval()
    INF.eval()

    acc = torchmetrics.Accuracy().to(device)
    losses = []

    with torch.no_grad():
        for ind, (X, (_, privlabels)) in enumerate(atk_test_dl):
            if torch.cuda.is_available():
                X, privlabels = X.float().cuda(), privlabels.float().cuda()
            batch_size, subsets, channels, height, width = X.shape
            all_agg_features=[]

            for i in range(batch_size):
                #valid_subset_images = X[i][non_zero_mask[i]]
                subset_images = X[i, :, :, :, :]
                #if valid_subset_images.shape[0] == 0:  # If all images in subset are zero-padded
                #    continue
                subset_features = FE(subset_images)

                #agg_features = subset_features.mean(dim=0)
                all_agg_features.append(subset_features)

            all_agg_features_tensor = torch.stack(all_agg_features, dim=0)
            #flattened_features = all_agg_features_tensor.view(batch_size, -1)
            
            pred_private_labels_agg = INF(all_agg_features_tensor)
            privlabels = map_labels_to_indices(privlabels).long().to(device)
            loss = atk_criterion(pred_private_labels_agg, privlabels)
            losses.append(loss.item())

            acc(torch.argmax(pred_private_labels_agg, dim=1), privlabels)

    mean_loss = np.mean(losses)

    print(f'Attacker Loss: {mean_loss:.6f} | Attacker Accuracy: {acc.compute():.2f}%')
    
    return mean_loss, acc.compute()


def get_FE_INF():
    save_dir = os.path.join(os.getcwd(), 'CelebA-w/o')
    fe_model_file = os.path.join(save_dir, "FE.pth")
    inf_model_file = os.path.join(save_dir, "INF.pth")

    fe = torch.load(fe_model_file)
    INF = ATK()
    if torch.cuda.is_available():
        fe = fe.to(device)
        INF =INF.to(device)
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            fe, INF = train_FE_INF(fe, INF, adv_trainloader, current_lr, device)
            eval_atk(fe,INF,adv_testloader,atk_criterion,device)
    except KeyboardInterrupt:
        pass

    torch.save(fe, fe_model_file)
    torch.save(INF, inf_model_file)

    return fe, INF
def extract_features(fe, dataloader, device, num_batches=None):
    fe.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (inputs, (_, privlabels)) in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
            inputs = inputs.to(device)
            batch_size, subsets, _, _, _ = inputs.shape
            aggregated_features = []
            for j in range(batch_size):
                subset_features = fe(inputs[j])
                aggregated_features.append(subset_features.mean(dim=0))
            # Flatten the features
            flattened_features = torch.flatten(torch.stack(aggregated_features), start_dim=1)
            features_list.append(flattened_features)
            labels_list.append(privlabels)
    
    features_array = torch.cat(features_list, dim=0).cpu().numpy()
    labels_array = torch.cat(labels_list, dim=0).cpu().numpy()

    return features_array, labels_array


def apply_tsne(features):
    tsne = TSNE(n_components=3, random_state=0)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

def plot_tsne_3d(reduced_features, labels):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    point_size=3
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        ax.scatter(reduced_features[labels == label, 0], 
                   reduced_features[labels == label, 1], 
                   reduced_features[labels == label, 2], 
                   label=f'Class {label}',s=point_size)
    
    plt.show()

def main():
    #fe, INF = get_FE_INF()
    path_to_fe="C:\\Users\\leily\\OneDrive\\Desktop\\property\\CelebA-w\\o\\FE.pth"
    fe= torch.load(path_to_fe)
    num_batches = 50
    features, labels = extract_features(fe, adv_testloader, device, num_batches)
    noise_factor = 0.5
    features_noisy = features + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=features.shape)
    reduced_features = apply_tsne(features_noisy)
    plot_tsne_3d(reduced_features, labels)
if __name__ == "__main__":
    print(next(iter(adv_testloader)))
    main()
