import torch.nn as nn
from torchvision.models import resnet18

class FE(nn.Module):
    def __init__(self):
        super(FE, self).__init__()
        # Pretrained ResNet-18 model
        model = resnet18(pretrained=True)
        
        # Remove the last fully connected layer to get the feature extractor
        self.features = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x):
        return self.features(x)

class clf(nn.Module):
    def __init__(self, num_classes=1):  # Default to 1000 classes like in ImageNet
        super(clf, self).__init__()
        # Last fully connected layer from ResNet-18
        self.classifier = nn.Linear(512, num_classes)  # ResNet-18's last layer has 512 input features

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)


feature_extractor = FE()
classifier = clf(num_classes=1) 


class FCResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, use_activation=True):
        super(FCResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()
        
        self.use_activation = use_activation
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        if self.use_activation:
            out = self.activation(out)
        out = self.fc2(out)
        out += self.shortcut(x)
        if self.use_activation:
            out = self.activation(out)
        return out


class ATK(nn.Module):
    def __init__(self, num_classes=11):
        super(ATK, self).__init__()

        self.block1 = FCResidualBlock(512, 256)
        self.block2 = FCResidualBlock(256, 128)
        self.classifier = nn.Linear(128, num_classes)  # Final layer

    def forward(self, x):
        # Aggregate the features here
        x = x.mean(dim=1)  # Assuming x.shape = (batch_size, num_subsets, feature_size)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
