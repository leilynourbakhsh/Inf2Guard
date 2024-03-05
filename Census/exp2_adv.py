import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from exp1 import CensusIncome, CensusDataset,CensusDatasetAdv
from exp1_models import IncomeFE, IncomeClassifier,GenderAdv
from exp2 import get_dataloaders
adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
import torchmetrics
device = 'cuda' if torch.cuda.is_available() else 'cpu'
atk_criterion = nn.CrossEntropyLoss().to(device)
test_atk_loss=[]
train_loss = []
test_loss = []
test_acc = []
total_epoch=100
lr=0.0001

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
def add_gaussian_noise(tensor, mean=0., std_dev=1):
    return tensor + torch.randn(tensor.size()).cuda() * std_dev + mean

def get_precision_recall(pred, true, zero_division=0):
    _, pred = torch.max(pred, 1)  # Get predicted classes
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    precision = precision_score(true, pred, average='macro', zero_division=zero_division)
    recall = recall_score(true, pred, average='macro', zero_division=zero_division)
    return precision, recall

def map_labels_to_indices(y):
    mapping = {0.2: 0, 0.3: 1, 0.4: 2, 0.5: 3}
    return torch.tensor([mapping[round(float(label), 1)] for label in y])

def train_FE_INF(FE, INF, data_train_loader, current_lr, device):
    INF.train()
    INF_optimizer = optim.Adam(INF.parameters(), lr=current_lr, weight_decay=1e-4)

    loss_INF = 0
    running_precision = 0
    running_recall = 0
    freeze(FE)

    for ind, (X, (privlabels, labels)) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            X, privlabels, labels = X.float().cuda(), privlabels.cuda(), labels.float().cuda()
        # get features from the feature extractor
        lengths = [len(sample) for sample in X]
        X = X.permute(0, 2, 1)
        X=add_gaussian_noise(X)
        pooled_feaures, _ = FE(X, lengths)
       
        # feed them to the inf model
        pred_private_labels = INF(pooled_feaures)
        #print("before mapping:",privlabels)
        privlabels = map_labels_to_indices(privlabels).to(device)
        #print("after mapping:",privlabels)
        loss_INF = atk_criterion(pred_private_labels, privlabels)
        precision, recall = get_precision_recall(pred_private_labels, privlabels)

        INF_optimizer.zero_grad()
        loss_INF.backward()
        INF_optimizer.step()

        running_precision += precision
        running_recall += recall

        if ind % 100 == 0:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss attacker: {:.6f}\t Precision attacker: {:.6f}\t recall attacler: {:.6f}\t'.format(
                    ind, ind * len(X), len(data_train_loader.dataset),
                       100. * ind / len(data_train_loader), loss_INF.detach().item(),
                       running_precision / 100,
                       running_recall / 100
                ))
        running_recall = 0
        running_precision = 0
    unfreeze(FE)
    return FE, INF


def eval_atk(fe, INF, atk_test_dl, atk_criterion, device):
    fe.eval()
    INF.eval()
    acc=torchmetrics.Accuracy().to(device)
    loss = []
    with torch.no_grad():
        for ind, (X, (privlabels, labels)) in enumerate(atk_test_dl):
            if torch.cuda.is_available():
                X, privlabels, labels = X.float().cuda(), privlabels.float().cuda(), labels.float().cuda()

            lengths = [len(sample) for sample in X]
            X = X.permute(0, 2, 1)
            X=add_gaussian_noise(X)
            pooled_features, _ = fe(X,lengths)
            raw_output = INF(pooled_features)  # Raw output from INF model
            atk_y = torch.argmax(raw_output, dim=1).squeeze()
            privlabels = map_labels_to_indices(privlabels).long().to(device)
            loss.append(atk_criterion(raw_output, privlabels).item())
            acc(atk_y, privlabels)

    loss = np.asarray(loss).mean()
    print(f'Classifier Loss: {loss} | Classifier Accuracy: {acc.compute()}')



def get_FE_INF():
    save_dir = os.path.join(os.getcwd(), 'Census-w/o')
    fe_model_file = os.path.join(save_dir, "FE.pth")
    inf_model_file = os.path.join(save_dir, "INF.pth")

    FE = torch.load(fe_model_file)
    INF = GenderAdv()
    if torch.cuda.is_available():
        FE = FE.to(device)
        INF =INF.to(device)
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            FE, INF = train_FE_INF(FE, INF, adv_trainloader, current_lr, device)
            eval_atk(FE,INF,adv_testloader,atk_criterion,device)
    except KeyboardInterrupt:
        pass

    torch.save(FE, fe_model_file)
    torch.save(INF, inf_model_file)

    return FE, INF
def extract_features_and_labels(fe, dataloader, device):
    fe.eval()
    all_features = []
    privlabels_list = []

    with torch.no_grad():
        for (X, (privlabels, _)) in dataloader:  # Ignore the other labels
            X = X.to(device)
            lengths = [len(sample) for sample in X]
            X = X.permute(0, 2, 1)
            pooled_features, _ = fe(X, lengths)
            
            # Map privlabels to indices
            mapped_privlabels = map_labels_to_indices(privlabels).numpy()

            # Use .squeeze() to make sure there are no singleton dimensions
            all_features.append(pooled_features.cpu().squeeze().numpy())
            privlabels_list.append(mapped_privlabels)

    # Convert lists to arrays
    all_features_array = np.vstack(all_features)
    privlabels_array = np.hstack(privlabels_list)

    return all_features_array, privlabels_array


from sklearn.preprocessing import StandardScaler

def plot_tsne_3d(features, labels, figsize=(5, 5), point_size=5, perplexity=30, learning_rate=20):
    # Standardize the data
    features = StandardScaler().fit_transform(features)

    tsne = TSNE(n_components=3, random_state=0, perplexity=perplexity, learning_rate=learning_rate)
    reduced_features = tsne.fit_transform(features)
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        idx = np.where(labels == lbl)
        ax.scatter(reduced_features[idx, 0], reduced_features[idx, 1], reduced_features[idx, 2], label=f'Class {lbl}', alpha=0.5, s=point_size)
    
    plt.show()


def main():
    FE, INF = get_FE_INF()
    #path_to_fe="C:\\Users\\leily\\OneDrive\\Desktop\\property\\Census-w\\o\\FE.pth"
    #fe= torch.load(path_to_fe)
    #features, labels = extract_features_and_labels(fe, adv_testloader, device)
    #noise_factor = 0
    #features_noisy = features + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=features.shape)
    
    #plot_tsne_3d(features_noisy, labels)

if __name__ == "__main__":
    #print(next(iter(adv_trainloader)))
    main()
