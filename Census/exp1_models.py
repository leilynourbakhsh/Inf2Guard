import torch.nn as nn
import torch.nn.functional as F
import torch


class IncomeFE(nn.Module):
    def __init__(self):
        super(IncomeFE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=42, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(in_features=128, out_features=128)
        #self.pool = nn.AdaptiveAvgPool1d(1)  # Add a pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x, lengths=None):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        # Create a mask based on lengths if provided
        if lengths is not None:
            max_length = out.size(2)
            mask = (torch.arange(max_length).expand(len(lengths), max_length) < torch.tensor(lengths).unsqueeze(1)).unsqueeze(1)
            mask = mask.expand(-1, 128, -1).to(out.device)  # number of filters in the last convolutional layer
            out = out * mask.float()

        unpooled_out = out.clone()  # Keep a copy of the unpooled features
        out = self.pool(out)  # Apply the pooling operation
        out = out.view(out.size(0), -1)  # Flatten the output
        out = self.fc(out)
        return out, unpooled_out  # Return both the pooled and unpooled features """



class IncomeClassifier(nn.Module):
    def __init__(self):
        super(IncomeClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # add a pooling layer
        self.fc = nn.Linear(32, 1)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out


    


class GenderAdv(nn.Module):
    def __init__(self):
        super(GenderAdv,self).__init__()
        self.features = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4),
        )
    def forward(self,x):
        out=self.features(x)
        return out
    
class GenderAdvDefense(nn.Module):
    def __init__(self):
        super(GenderAdvDefense,self).__init__()
        self.features = nn.Sequential(
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4),
        )
    def forward(self,x):
        out=self.features(x)
        return out


# Initialize the model
""" model = IncomeFE()

# Create a dummy input
x = torch.rand(1, 42, 1000)  # Adjust the shape according to your needs

# Pass the dummy input through all layers up to the pooling layer
out = F.relu(model.conv1(x))
out = F.relu(model.conv2(out))
out = model.pool(out)

# Print the output shape
print(out.shape)
 """