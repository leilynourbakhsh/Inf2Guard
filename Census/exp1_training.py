import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import os
import numpy as np
from exp1 import CensusIncome, CensusDataset,CensusDatasetAdv
from exp1_models import IncomeFE, IncomeClassifier
from exp1 import get_dataloaders
adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
import torchmetrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss().to(device)
test_clf_loss=[]  # list to store loss values
total_epoch=60
lr=0.001
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def extract_features(model, dataloader, device, num_batches=None):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
            inputs = inputs.to(device)
            print(inputs.shape)
            features = model(inputs)
            features_flat = features.view(features.size(0), -1)  # Flatten the features
            features_list.append(features_flat.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    features_array = np.vstack(features_list)
    labels_array = np.concatenate(labels_list)

    return features_array, labels_array
def apply_tsne(features):
    tsne = TSNE(n_components=3, random_state=0)
    reduced_features = tsne.fit_transform(features)
    return reduced_features
def plot_tsne_3d(reduced_features, labels):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    point_size = 2
    
    for i in range(2):  # Adjusted for binary classification
        ax.scatter(reduced_features[labels == i, 0], 
                   reduced_features[labels == i, 1], 
                   reduced_features[labels == i, 2], 
                   label=f'Class {i}', s=point_size)
    
    plt.show()

def adjust_learning_rate(epoch, init_lr=0.001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr

def train_FE_CF(FE, CF, data_train_loader, current_lr, vis=None):
    FE.train()
    CF.train()
    FE_optimizer = optim.Adam(FE.parameters(), lr=current_lr, weight_decay=1e-4)
    CF_optimizer = optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)

    loss_CF = 0
    acc = Accuracy().to(device)  # Add accuracy computation
    # Initialize counters
    counters = [0]*4  # Assuming batch size is always 4
    total_samples = 0
    correct_samples = 0
    for ind, (X, (privlabels, labels)) in enumerate(data_train_loader):
        if torch.cuda.is_available():
            X, privlabels, labels = X.float().cuda(), privlabels.cuda(), labels.float().cuda()

        # get features from the feature extractor
        lengths = [len(sample) for sample in X]
        X = X.permute(0, 2, 1)
        _ , unpooled_features = FE(X, lengths)
        #print(labels.shape)
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
            #print(single_label.shape)
            # Feed the single sample to the CF
            output_CF = CF(single_sample)
            loss_CF = criterion(output_CF.view(-1), single_label.view(-1))
            total_loss_CF += loss_CF

            # Update accuracy computation
            total_samples += 1
            output_binary = (torch.sigmoid(output_CF) > 0.5).float()
            correct_samples += (output_binary == single_label).sum().item()

            # Increment the counter and reset to 0 if it exceeds the number of samples in the subset
            counters[i] += 1
            if counters[i] >= len(labels[i]):
                counters[i] = 0
        
        if total_loss_CF == 0:  # If no loss computation was performed, skip this iteration
            continue
        # Compute loss and backprop
        loss = total_loss_CF / 4.0

        FE_optimizer.zero_grad()
        CF_optimizer.zero_grad()

        loss.backward()

        FE_optimizer.step()
        CF_optimizer.step()

        if ind % 100 == 0:
            current_accuracy = correct_samples / total_samples
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss Classifier: {:.6f}\tTraining Accuracy: {:.6f}'.format(
                ind, ind * len(X), len(data_train_loader.dataset),
                100. * ind / len(data_train_loader), loss.item(), current_accuracy))

    return FE, CF

def eval_clf(FE, CF, data_test_loader, criterion, device):
    FE.eval()
    CF.eval()
    total_loss = 0.0
    total_samples = 0
    correct_samples = 0
    counters = [0]*4  # Assuming batch size is always 4
    with torch.no_grad():
        for _, (X, (privlabels, labels)) in enumerate(data_test_loader):
            if torch.cuda.is_available():
                X, privlabels, labels = X.float().cuda(), privlabels.cuda(), labels.float().cuda()

            lengths = [len(sample) for sample in X]
            X = X.permute(0, 2, 1)
            _, unpooled_features = FE(X, lengths)

            mask = ~torch.isnan(labels)  # Add this line to initialize mask

            total_loss_in_batch = 0
            for i in range(4):
                while counters[i] < len(labels[i]) and not mask[i, counters[i]]:  # Skip over padding values
                    counters[i] += 1
                    if counters[i] >= len(labels[i])-1:  # If we've gone through all data, reset the counter
                        counters[i] = 0
                        break  # Break to skip this iteration

                if counters[i] >= len(labels[i]):
                    continue  # Skip this iteration if counter has reached the end
                
                single_sample = unpooled_features[i, :, counters[i]:counters[i]+1].unsqueeze(0)
                single_label = labels[i, counters[i]]
                output = CF(single_sample)
                loss = criterion(output.view(-1), single_label.view(-1))
                total_loss_in_batch += loss

                # Update accuracy computation
                total_samples += 1
                output_binary = (torch.sigmoid(output) > 0.5).float()
                correct_samples += (output_binary == single_label).sum().item()

                # Increment the counter and reset to 0 if it exceeds the number of samples in the subset
                counters[i] += 1
                if counters[i] >= len(labels[i]):
                    counters[i] = 0
                    
            total_loss += total_loss_in_batch.item() / 4.0
    avg_loss = total_loss / len(data_test_loader)
    avg_acc = correct_samples / total_samples
    print('Test Loss: {:.6f}\tTest Accuracy: {:.6f}'.format(avg_loss, avg_acc))



def get_FE_CF():
    FE = IncomeFE().to(device)
    CF = IncomeClassifier().to(device)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            FE = torch.nn.DataParallel(FE)
            CF = torch.nn.DataParallel(CF)

    for epoch in range(total_epoch):
        print("epoch %d" % epoch)
        current_lr = adjust_learning_rate(epoch, lr)
        FE, CF = train_FE_CF(FE, CF, adv_trainloader, current_lr)
        eval_clf(FE, CF, adv_testloader, criterion, device)

    save_dir = os.path.join(os.getcwd(), 'Census-w/o')
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        torch.save(FE.module, os.path.join(save_dir, "FE.pth"))
        torch.save(CF.module, os.path.join(save_dir, "CF.pth"))
    else:
        torch.save(FE, os.path.join(save_dir, "FE.pth"))
        torch.save(CF, os.path.join(save_dir, "CF.pth"))

    return FE, CF

def main():
    #batch_size = 64  # adjust to your needs
    #train_dataloader, test_dataloader = load_data(batch_size)
    #batch_size = 4

# Get the dataloaders
  
    #FE, CF = get_FE_CF()
    path_to_FE="C:\\Users\\leily\\OneDrive\\Desktop\\property\\Census-w\\o\\INF.pth"
    FE = torch.load(path_to_FE)
    num_batches = 20
    features, labels = extract_features(FE, adv_testloader, device, num_batches)
    reduced_features = apply_tsne(features)
    plot_tsne_3d(reduced_features, labels)

if __name__ == "__main__":
    
    main()