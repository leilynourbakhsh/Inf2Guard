import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import os
import numpy as np
import torchmetrics
from CelebA  import  CelebASubsetDataset,loading
from celebA_models import FE,clf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
total_epoch=100
lr=0.0001

# getting data_loaders
adv_trainloader, adv_testloader=loading()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss().to(device)
def add_gaussian_noise(tensor, mean=0., std_dev=2):
    return tensor + torch.randn(tensor.size()).cuda() * std_dev + mean

def adjust_learning_rate(epoch, init_lr=0.001):
    schedule = [12]
    cur_lr = init_lr
    for schedule_epoch in schedule:
        if epoch >= schedule_epoch:
            cur_lr *= 0.1
    return cur_lr
def train_FE_CF(FE, CF, train_loader, current_lr):
    # Set the feature extractor and classifier to training mode
    FE.train()
    CF.train()

    # Define the loss function for binary classification
    criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Define the optimizers
    optimizer_FE = torch.optim.Adam(FE.parameters(), lr=current_lr, weight_decay=1e-4)
    optimizer_CF = torch.optim.Adam(CF.parameters(), lr=current_lr, weight_decay=1e-4)

    # Initialize variables for tracking loss and accuracy
    total_loss = 0.0
    correct_predictions = 0
    total_valid_samples = 0

    # Iterate through the data loader
    for batch_idx, (X, (labels, _)) in enumerate(train_loader):
        #print(X.shape)
        #X = torch.cat(tuple(X), dim=0)
        # Reshape the batch_images to merge the batch size and number of samples per subset
        batch_images = X.view(-1, *X.shape[2:])
        for tensor in labels:
            if tensor.dim() == 0:
                print("Found a zero-dimensional tensor!")

        batch_labels = labels.view(-1).to(device)
        valid_mask = ~torch.isnan(batch_labels)
        batch_labels = batch_labels[valid_mask].unsqueeze(1)  # Exclude nan values and add extra dimension
        
        # Move to GPU if available
        if torch.cuda.is_available():
            batch_images = batch_images.cuda()

        # Forward pass through the feature extractor
        batch_images = add_gaussian_noise(batch_images)
        features = FE(batch_images)
        valid_features = features[valid_mask]

        # Forward pass through the classifier
        outputs = CF(valid_features)

        # Compute the loss
        loss = criterion(outputs, batch_labels)
        loss = loss.mean()  # Compute the mean of the valid losses
        total_loss += loss.item()

        # Compute the accuracy
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct_predictions += (predicted == batch_labels).sum().item()

        # Zero the parameter gradients
        optimizer_FE.zero_grad()
        optimizer_CF.zero_grad()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer_FE.step()
        optimizer_CF.step()

        # Print the loss after every few batches
        if batch_idx % 100 == 0:
            print('Batch: {}/{}, Loss: {:.6f}'.format(batch_idx, len(train_loader), loss.item()))
        total_valid_samples += batch_labels.size(0)

    # Calculate the overall loss and accuracy for the epoch
    num_samples_per_subset = X.shape[1]
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = correct_predictions / total_valid_samples

    print('Epoch Loss: {:.6f}, Epoch Accuracy: {:.2f}%'.format(epoch_loss, epoch_accuracy * 100))
    return FE, CF

def eval_clf(FE, CF, data_test_loader, criterion, device):
    FE.eval()
    CF.eval()
    total_loss = 0.0
    correct_samples = 0
    total_valid_samples = 0  # Initialize the total valid samples counter

    with torch.no_grad():
        for batch_idx, (X, (labels, _)) in enumerate(data_test_loader):
            # Reshape the batch_images to merge the batch size and number of samples per subset
            batch_images = X.view(-1, *X.shape[2:])

            # Reshape the labels in the same way
            batch_labels = labels.view(-1).to(device)
            valid_mask = ~torch.isnan(batch_labels)
            batch_labels = batch_labels[valid_mask].unsqueeze(1)  # Exclude nan values and add extra dimension

            if torch.cuda.is_available():
                batch_images = batch_images.cuda()

            # Forward pass through the feature extractor
           # print('input FE', batch_images.shape)
            batch_images = add_gaussian_noise(batch_images)
            features = FE(batch_images)
            valid_features = features[valid_mask]

            # Forward pass through the classifier
            #print('output FE', valid_features.shape)
            outputs = CF(valid_features)

            # Compute the loss
            loss = criterion(outputs, batch_labels).mean()  # Compute the mean of the valid losses
            total_loss += loss.item()

            # Compute the accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct_samples += (predicted == batch_labels).sum().item()

            # Increment total valid samples count
            total_valid_samples += batch_labels.size(0)

    # Adjust the accuracy calculation based on valid samples
    avg_loss = total_loss / len(data_test_loader)
    avg_acc = correct_samples / total_valid_samples

    print('Test Loss: {:.6f}\tTest Accuracy: {:.2f}%'.format(avg_loss, avg_acc * 100))

def get_FE_CF():
    fe = FE().to(device)
    CF = clf().to(device)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            fe = torch.nn.DataParallel(fe)
            CF = torch.nn.DataParallel(CF)

    for epoch in range(total_epoch):
        print("epoch %d" % epoch)
        current_lr = adjust_learning_rate(epoch, lr)
        fe, CF = train_FE_CF(fe, CF, adv_trainloader, current_lr)
        eval_clf(fe, CF, adv_testloader, criterion, device)

    save_dir = os.path.join(os.getcwd(), 'CelebA-w/o')
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.device_count() > 1:
        torch.save(fe.module, os.path.join(save_dir, "FE_DP.pth"))
        torch.save(CF.module, os.path.join(save_dir, "CF_DP.pth"))
    else:
        torch.save(fe, os.path.join(save_dir, "FE_DP.pth"))
        torch.save(CF, os.path.join(save_dir, "CF_DP.pth"))

    return fe, CF
def extract_features(model, dataloader, device, num_batches=None):
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for i, (X, (labels, _)) in enumerate(dataloader):
            if num_batches and i >= num_batches:
                break
            batch_images = X.view(-1, *X.shape[2:]).to(device)
            valid_mask = ~torch.isnan(labels.view(-1))
            valid_labels = labels.view(-1)[valid_mask].cpu().numpy()
            valid_images = batch_images[valid_mask]
            
            features = model(valid_images)
            features_flat = features.view(features.size(0), -1)  # Flatten the features
            features_list.append(features_flat.cpu().numpy())
            labels_list.append(valid_labels)
    
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
    point_size=2
    
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        ax.scatter(reduced_features[labels == lbl, 0], 
                   reduced_features[labels == lbl, 1], 
                   reduced_features[labels == lbl, 2], 
                   label=f'Class {lbl}', alpha=0.5,s=point_size)
    
    plt.show()
def main():
    print(next(iter(adv_trainloader)))
    fe, CF = get_FE_CF()
    #for visualization please comment the above code. which the Fe file has already been saved in the following address
    #path_to_fe="C:\\Users\\leily\\OneDrive\\Desktop\\property\\CelebA-w\\o\\FE.pth"
    #fe= torch.load(path_to_fe)
    #num_batches = 30
    #features, labels = extract_features(fe, adv_testloader, device, num_batches)
    #reduced_features = apply_tsne(features)
    #plot_tsne_3d(reduced_features, labels)
if __name__ == "__main__":
    main()