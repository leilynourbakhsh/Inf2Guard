import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from exp1_models import IncomeFE, IncomeClassifier
from exp4 import get_dataloaders
adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
import torchmetrics
def add_gaussian_noise(tensor, mean=0., std_dev=1000):
    return tensor + torch.randn(tensor.size()).cuda() * std_dev + mean
device = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = nn.BCEWithLogitsLoss().to(device)
test_clf_loss=[]  # list to store loss values
total_epoch=300
lr=0.001


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
        X=add_gaussian_noise(X)
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
            X=add_gaussian_noise(X)
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
                if isinstance(labels[i], torch.Tensor):
                    
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

            if isinstance(total_loss_in_batch, torch.Tensor):
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

def extract_features_and_labels(fe, dataloader, device):
    fe.eval()
    all_features = []
    labels_list = []

    max_rows = 0  # to store the max number of rows among all feature vectors
    max_cols = 0  # to store the max number of columns among all feature vectors

    # First, compute the max number of rows and columns
    with torch.no_grad():
        for (X, (_, labels)) in dataloader:
            X = X.to(device)
            lengths = [len(sample) for sample in X]
            X = X.permute(0, 2, 1)
            _, unpooled_features = fe(X, lengths)

            for i in range(X.size(0)):
                feature = unpooled_features[i].squeeze().cpu().numpy()
                max_rows = max(max_rows, feature.shape[0])
                max_cols = max(max_cols, feature.shape[1])

    # Now, extract and pad features to max_rows and max_cols
    with torch.no_grad():
        for (X, (_, labels)) in dataloader:
            X = X.to(device)
            lengths = [len(sample) for sample in X]
            X = X.permute(0, 2, 1)
            _, unpooled_features = fe(X, lengths)

            for i in range(X.size(0)):
                feature = unpooled_features[i].squeeze().cpu().numpy()
                padded_feature = np.zeros((max_rows, max_cols))
                padded_feature[:feature.shape[0], :feature.shape[1]] = feature  # pad with zeros
                all_features.append(padded_feature.flatten())  # flatten the padded feature
                labels_list.append(labels[i].cpu().numpy()[0]) 

    return np.array(all_features), np.array(labels_list)




def plot_tsne_3d(features, labels):
    # Get indices for both classes
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]
    
    # Determine the number of samples to take from each class
    num_samples = min(len(class0_indices), len(class1_indices))
    
    # Randomly select 'num_samples' from each class
    selected_class0_indices = np.random.choice(class0_indices, num_samples, replace=False)
    selected_class1_indices = np.random.choice(class1_indices, num_samples, replace=False)
    
    # Combine the selected indices and extract the corresponding features and labels
    combined_indices = np.concatenate((selected_class0_indices, selected_class1_indices))
    selected_features = features[combined_indices]
    selected_labels = labels[combined_indices]

    # Perform t-SNE
    tsne = TSNE(n_components=3, random_state=0)
    reduced_features = tsne.fit_transform(selected_features)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(reduced_features[selected_labels == 0, 0], reduced_features[selected_labels == 0, 1], reduced_features[selected_labels == 0, 2], label='Class 0', alpha=0.5, s=5)
    ax.scatter(reduced_features[selected_labels == 1, 0], reduced_features[selected_labels == 1, 1], reduced_features[selected_labels == 1, 2], label='Class 1', alpha=0.5, s=5)
    #ax.legend()
    #ax.set_title("3D t-SNE of FE's outputs")
    plt.show()



def main():
    #batch_size = 64  # adjust to your needs
    #train_dataloader, test_dataloader = load_data(batch_size)
    batch_size = 4

# Get the dataloaders
  
    FE, CF = get_FE_CF()
    #path_to_fe="C:\\Users\\leily\\OneDrive\\Desktop\\property\\Census-w\\o\\FE.pth"
    #fe= torch.load(path_to_fe)
    #features, labels = extract_features_and_labels(fe, adv_testloader, device)
    #noise_factor = 0.1
    #features_noisy = features + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=features.shape)
    
    #plot_tsne_3d(features_noisy, labels)

if __name__ == "__main__":
    
    main()