import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
class ResNetFE(nn.Module):
    def __init__(self):
        super(ResNetFE, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(
            # Add this layer to expand the single channel to 3 channels
            nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False),
            *list(resnet.children())[:-2] # Excluding the fully connected layer
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.features(x)
        return x # You return the unpooled features
class DenseNetFE(nn.Module):
    def __init__(self):
        super(DenseNetFE, self).__init__()
        densenet = models.densenet121(pretrained=True)
        
        # For a single channel input (like grayscale images)
        # The following modifies the first convolution layer to accept a single channel
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        self.features = densenet.features  # Note: The classifier part of densenet is not included.

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = self.features(x)
        return x
####### for DensNet
class AgeClassifier(nn.Module):
    def __init__(self):
        super(AgeClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)  # Change 512 to 1024 for DenseNet121
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# #binary classification for the age prediction
# class AgeClassifier(nn.Module):
#     def __init__(self):
#         super(AgeClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(256, 1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
    
# class GenderAdv(nn.Module):
#     def __init__(self):
#         super(GenderAdv, self).__init__()
#         self.features = nn.Sequential(
#             nn.Linear(512, 128), # Assuming using ResNet-18
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 7), # 7 classes for the private label
#         )

#     def forward(self, x, mask=None):
       
#         x = x.mean(dim=[3, 4])  # Aggregation over spatial dimensions, shape becomes (batch_size, subsets, 512)

#         # Average over subsets as well
#         x = x.mean(dim=1)  # Shape becomes (batch_size, 512)

#         if mask is not None:
#             x = x * mask

#         out = self.features(x)
#         return out
###################################################################### the new one
# class GenderAdv(nn.Module):
#     def __init__(self):
#         super(GenderAdv, self).__init__()
#         self.features = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128,64),
#             nn.ReLU(),
#             nn.Linear(64, 7)  # 7 classes for the private label
#         )

#     def forward(self, x, mask=None):
#         if mask is not None:
#             x = x * mask
#         out = self.features(x)
#         return out
#######################################################################

# class Attacker(nn.Module):
#     def __init__(self):
#         super(Attacker, self).__init__()
        
#         # Assuming the output features from the DenseNetFE are pooled to a shape of (batch_size, 1024)
#         # If not, adjust the input features of fc1 accordingly (512 for DenseNet121 before pooling)
#         self.fc1 = nn.Linear(1024, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 7)  # 7 classes as you mentioned

#     def forward(self, x):
#         x = x.view(x.size(0), -1)  # Flatten the feature tensors
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
class Attacker(nn.Module):
    def __init__(self):
        super(Attacker, self).__init__()
        
        self.fc1 = nn.Linear(1024, 512) # This assumes the output of FE has 1024 channels
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 7)

    def forward(self, x):
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
