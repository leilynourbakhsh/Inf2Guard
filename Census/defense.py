import torch 
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import os
import numpy as np
from exp1_models import GenderAdvDefense
from exp4_training import adjust_learning_rate
from exp4 import get_dataloaders
from exp4_adv import map_labels_to_indices


adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
save_dir = os.path.join(os.getcwd(), 'Census-w/o')
import torchmetrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss().cuda()
loss_fn = nn.CrossEntropyLoss().cuda()

total_epoch=100
lr=1e-5
lr_atk=1e-4
tradoff=0.5


def train(FE_model,INF_model, CF_model, trainloader, current_lr,current_lr_atk,tradoff=0.5 ):
    FE_model.train()
    INF_model.train()
    CF_model.train()

    #initialization for primary task loss computation and accuracy
    loss_CF = 0
    counters = [0]*4  # Assuming batch size is always 4

    FE_optimizer = optim.Adam(FE_model.parameters(), lr=current_lr, weight_decay=1e-4)
    CF_optimizer = optim.Adam(CF_model.parameters(), lr=current_lr, weight_decay=1e-4)
    INF_optimizer = optim.Adam(INF_model.parameters(), lr=current_lr_atk, weight_decay=1e-4)
    
    for ind, (X, (privlabels, labels)) in enumerate(trainloader):
        if torch.cuda.is_available():
            X, privlabels, labels = X.float().cuda(), privlabels.cuda(), labels.float().cuda()

        # get features from the feature extractor
        lengths = [len(sample) for sample in X]
        X = X.permute(0, 2, 1)
        pooled_features, unpooled_features = FE_model(X, lengths)
        privlabels = map_labels_to_indices(privlabels).to(device)
        # feed pooled features to the inf model
        pred_private_labels = INF_model(pooled_features)
        loss_INF = loss_fn(pred_private_labels, privlabels)

        # feed unpooled features to the CF
        total_loss_CF=0
        mask = ~torch.isnan(labels)

        for i in range(4):
            while counters[i] < len(labels[i]) and not mask[i, counters[i]]:  # Skip over padding values
                counters[i] += 1
                if counters[i] >= len(labels[i])-1:  # If we've gone through all data, reset the counter
                    counters[i] = 0
                    break
            if counters[i] >= len(labels[i])-1:
                continue
            single_sample = unpooled_features[i, :, counters[i]:counters[i]+1].unsqueeze(0)# Use counter to select sample
            single_label = labels[i, counters[i]]
            # Feed the single sample to the CF
            output_CF = CF_model(single_sample)
            loss_CF = criterion(output_CF.view(-1), single_label.view(-1))
            total_loss_CF += loss_CF
            # Increment the counter and reset to 0 if it exceeds the number of samples in the subset
            counters[i] += 1
            if counters[i] >= len(labels[i]):
                counters[i] = 0
        if total_loss_CF == 0:  # If no loss computation was performed, skip this iteration
            continue
        final_CF_loss = total_loss_CF / 4.0
        # compute loss and backprop
        loss = -tradoff * loss_INF + (1. - tradoff) * final_CF_loss
        
        FE_optimizer.zero_grad()
        loss.backward()
        FE_optimizer.step()

        
         # get features from the feature extractor
        pooled_features, unpooled_features = FE_model(X, lengths)
        pooled_features = pooled_features.detach()
        unpooled_features = unpooled_features.detach()
        INF_optimizer.zero_grad()
        CF_optimizer.zero_grad()
        
        # feed the private labels,female ratio, to the INF model
        pred_private_labels = INF_model(pooled_features)
        loss_INF = loss_fn(pred_private_labels, privlabels)
        loss_INF.backward()
        INF_optimizer.step()

       
        # feed primary labels to the CF model
        total_loss_CF=0
        mask = ~torch.isnan(labels)

        for i in range(4):
            while counters[i] < len(labels[i]) and not mask[i, counters[i]]:  # Skip over padding values
                counters[i] += 1
                if counters[i] >= len(labels[i])-1:  # If we've gone through all data, reset the counter
                    counters[i] = 0
                    break
            if counters[i] >= len(labels[i])-1:
                continue
            single_sample = unpooled_features[i, :, counters[i]:counters[i]+1].unsqueeze(0)# Use counter to select sample
            single_label = labels[i, counters[i]]
            # Feed the single sample to the CF
            output_CF = CF_model(single_sample)
            loss_CF = criterion(output_CF.view(-1), single_label.view(-1))
            total_loss_CF += loss_CF
            # Increment the counter and reset to 0 if it exceeds the number of samples in the subset
            counters[i] += 1
            if counters[i] >= len(labels[i]):
                counters[i] = 0
        if total_loss_CF == 0:  # If no loss computation was performed, skip this iteration
            continue
        final_CF_loss = total_loss_CF / 4.0
        final_CF_loss.backward()
        CF_optimizer.step()

        if ind % 100 == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss Classifier: {:.6f}\tLoss Attacker: {:.6f}'.format(
                ind, ind * len(X), len(trainloader.dataset),
                100. * ind / len(trainloader), loss_CF.item(), loss_INF.item()))

    return FE_model, INF_model, CF_model       


def load_model(model_file):
    model = torch.load(model_file)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def save_model(model, model_file):
    if torch.cuda.device_count() > 1:
        torch.save(model.module, model_file)
    else:
        torch.save(model, model_file)


def train_model(FE, INF, CF, adv_trainloader, total_epoch, lr, lr_atk, tradoff):
    try:
        for epoch in range(total_epoch):
            print(f"Epoch: {epoch}")
            current_lr = adjust_learning_rate(epoch, lr)
            current_lr_atk = adjust_learning_rate(epoch, lr_atk)
            FE, INF, CF = train(FE, INF, CF, adv_trainloader, current_lr, current_lr_atk, tradoff)
    except KeyboardInterrupt:
        print("Training interrupted")
    return FE, INF, CF

def get_FE_defense():
    FE = load_model(os.path.join(save_dir, "FE.pth"))
    #INF = load_model(os.path.join(save_dir, "INF.pth"))
    INF=GenderAdvDefense().to(device)
    CF = load_model(os.path.join(save_dir, "CF.pth"))

    FE, INF, CF = train_model(FE, INF, CF, adv_trainloader, total_epoch, lr, lr_atk, tradoff)

    if torch.cuda.device_count() > 1:
        torch.save(FE.module, os.path.join(save_dir, "FE_defense.pth"))
        torch.save(INF.module, os.path.join(save_dir, "INF_defense.pth"))
        torch.save(CF.module, os.path.join(save_dir, "CF_defense.pth"))
    else:
        torch.save(FE, os.path.join(save_dir, "FE_defense.pth"))
        torch.save(INF, os.path.join(save_dir, "INF_defense.pth"))
        torch.save(CF, os.path.join(save_dir, "CF_defense.pth"))

    return FE


if __name__ == "__main__":
    get_FE_defense()