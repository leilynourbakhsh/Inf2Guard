import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score
import os
import numpy as np
from exp1_2 import BonAgeDataset, custom_collate_fn
from exp2_models import Attacker
import torchmetrics
from exp1_2 import process_data, get_dataloaders
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

BASE_DATA_DIR = "C:\\Users\\leily\\OneDrive\\Desktop\\property\\archive\\archive"
path = os.path.join(BASE_DATA_DIR, 'boneage-training-dataset')

train_df, test_df = process_data(path, split_second_ratio=0.5)
train_loader, test_loader = get_dataloaders(train_df, test_df, force_generate=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atk_criterion = nn.CrossEntropyLoss().to(device)
test_atk_loss=[]
train_loss = []
test_loss = []
test_acc = []
total_epoch=100
lr=0.001

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

def map_labels_to_indices(y):
    mapping = {0.2: 0, 0.3: 1, 0.4: 2, 0.5: 3, 0.6: 4, 0.7: 5, 0.8: 6}
    return torch.tensor([mapping[round(float(label), 1)] for label in y])

def train_FE_INF(FE, INF, data_train_loader, current_lr, device):
    INF.train()
    INF_optimizer = optim.Adam(INF.parameters(), lr=current_lr, weight_decay=1e-4)
    running_loss = 0.0
    running_precision = 0.0
    running_recall = 0.0
    freeze(FE)

    for ind, (X, (privlabels, _)) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            X, privlabels = X.float().cuda(), privlabels.cuda()
        
        batch_size, subsets, channels, height, width = X.shape
        X = X.view(batch_size * subsets, channels, height, width)

        # Get features from the feature extractor
        pooled_features = FE(X)
        # Aggregate over spatial dimensions (Global Average Pooling in this case)
        x_agg = pooled_features.mean(dim=[2, 3])
        
        # Feed them to the INF model
        x_agg = x_agg.view(batch_size, subsets, -1)
        all_preds = []
        for i in range(subsets):
            subset_features = x_agg[:, i, :]
            pred_private_labels = INF(subset_features)
            all_preds.append(pred_private_labels)
        
        all_preds_tensor = torch.stack(all_preds, dim=1)

        # Map the labels to indices
        privlabels = map_labels_to_indices(privlabels).to(device)
        privlabels = privlabels.unsqueeze(1).expand(-1, all_preds_tensor.shape[1])

        # Compute the loss
        loss_INF = atk_criterion(all_preds_tensor.view(-1, 7), privlabels.reshape(-1))

        precision, recall = get_precision_recall(all_preds_tensor.view(-1, 7), privlabels.reshape(-1))

        INF_optimizer.zero_grad()
        loss_INF.backward()
        INF_optimizer.step()

        running_loss += loss_INF.detach().item()
        running_precision += precision
        running_recall += recall

        if ind % 100 == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss attacker: {:.6f}\t Precision attacker: {:.6f}\t recall attacker: {:.6f}\t'.format(
                    ind, ind * len(X), len(data_train_loader.dataset),
                       100. * ind / len(data_train_loader), running_loss / (ind + 1),
                       running_precision / (ind + 1),
                       running_recall / (ind + 1)
                ))

    unfreeze(FE)
    return FE, INF



def eval_atk(fe, INF, atk_test_dl, atk_criterion, device):
    fe.eval()
    INF.eval()
    acc = torchmetrics.Accuracy().to(device)
    loss = []
    with torch.no_grad():
        for ind, (X, (privlabels, _)) in enumerate(atk_test_dl):
            if torch.cuda.is_available():
                X, privlabels = X.float().cuda(), privlabels.float().cuda()

            batch_size, subsets, channels, height, width = X.shape
            X = X.view(batch_size * subsets, channels, height, width)

            pooled_features = fe(X)
            x_agg = pooled_features.mean(dim=[2, 3])
            raw_output = INF(x_agg)

            # Reshape and aggregate over subsets
            pred_private_labels = raw_output.view(batch_size, subsets, -1)
            pred_private_labels_agg = pred_private_labels.mean(dim=1)

            # Map the labels to indices
            privlabels = map_labels_to_indices(privlabels).long().to(device)
            loss.append(atk_criterion(pred_private_labels_agg, privlabels).item())
            acc(torch.argmax(pred_private_labels_agg, dim=1), privlabels)

    loss = np.asarray(loss).mean()
    print(f'Attacker Loss: {loss} | Attacker Accuracy: {acc.compute()}') 
def get_FE_INF():
    save_dir = os.path.join(os.getcwd(), 'BoneAge-w/o')
    fe_model_file = os.path.join(save_dir, "FE.pth")
    inf_model_file = os.path.join(save_dir, "INF.pth")

    FE = torch.load(fe_model_file)
    INF = Attacker()
    if torch.cuda.is_available():
        FE = FE.to(device)
        INF =INF.to(device)
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            FE, INF = train_FE_INF(FE, INF, train_loader, current_lr, device)
            eval_atk(FE,INF,test_loader,atk_criterion,device)
    except KeyboardInterrupt:
        pass

    torch.save(FE, fe_model_file)
    torch.save(INF, inf_model_file)

    return FE, INF
def extract_features(fe, dataloader, device, num_batches=None):
    fe.eval()
    features_list = []
    temp_labels_list = []  # Temporary list to store labels

    with torch.no_grad():
        for i, (inputs, (privlabels, _)) in enumerate(dataloader):
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
            temp_labels_list.extend(privlabels.squeeze().tolist())  # Extend the temporary labels list

    features_array = torch.cat(features_list, dim=0).cpu().numpy()
    # Match the size of the labels list with the size of the features array
    labels_array = np.array(temp_labels_list[:features_array.shape[0]])

    return features_array, labels_array


def apply_tsne(features):
    tsne = TSNE(n_components=3, random_state=0)
    reduced_features = tsne.fit_transform(features)
    return reduced_features

def plot_tsne_3d(reduced_features, labels):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    point_size=4
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        ax.scatter(reduced_features[labels == lbl, 0], 
                   reduced_features[labels == lbl, 1], 
                   reduced_features[labels == lbl, 2], 
                   label=f'Class {lbl}', alpha=0.5,s=point_size)
    plt.show()
def main():
    #FE, INF = get_FE_INF()
    path_to_fe="C:\\Users\\leily\\OneDrive\\Desktop\\property\\BoneAge-w\\o\\FE.pth"
    fe= torch.load(path_to_fe)
    num_batches = 1500
    features, labels = extract_features(fe, test_loader, device, num_batches)
    noise_factor = 0.5
    features_noisy = features + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=features.shape)
    
    # Apply t-SNE and visualize
    reduced_features = apply_tsne(features_noisy)
    plot_tsne_3d(reduced_features, labels)

if __name__ == "__main__":
    #print(next(iter(train_loader)))
    main()
