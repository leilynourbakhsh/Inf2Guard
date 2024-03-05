import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score
import os
import numpy as np
from exp1 import CensusIncome, CensusDataset,CensusDatasetAdv
from exp1_models import IncomeFE, IncomeClassifier,GenderAdv
from exp3 import get_dataloaders
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


def main():
    FE, INF = get_FE_INF()

if __name__ == "__main__":
    #print(next(iter(adv_trainloader)))
    main()
