import math

import torch
from torch import nn


class Cifar_Encoder(nn.Module):
    def __init__(self,  num_step=0):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            #nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
        )
        self.layer44 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer55 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer7 = View(-1, 256 * 4 * 16)
        self.layer8 = nn.Sequential(
            nn.Linear(4096 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )
        self.layer9 = nn.Sequential(
            nn.Linear(1024, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer10 = nn.Sequential(
            nn.Linear(64, 10),
            #nn.BatchNorm1d(10),
            #nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.layers.append(self.layer44)
        self.layers.append(self.layer5)
        self.layers.append(self.layer55)
        self.layers.append(self.layer6)
        self.layers.append(self.layer7)
        self.layers.append(self.layer8)
        self.layers.append(self.layer9)
        self.layers.append(self.layer10)
        self.layers = self.layers[0:num_step+1]

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Mnist_Encoder(nn.Module):
    def __init__(self,  num_step=0):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            #nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            #nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(4),
            #nn.ReLU(),
        )
        self.layer3 = View(-1, 4*28*28)
        self.layer4 = nn.Sequential(
            nn.Linear(4*28*28, 1024),
            nn.BatchNorm1d(1024),
            #nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(1024, 10),
            nn.BatchNorm1d(10),
            #nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.layers.append(self.layer5)
        self.layers = self.layers[0:num_step + 1]

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.noise_size = args.noise_structure
        if args.dataset == 'cifar':
            self.a = 32
        else:
            self.a = 28
        self.k = self.noise_size[1]
        self.layer0 = nn.Sequential(
            nn.Conv2d(self.k, self.k*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.k*2),
            # nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.k*2, self.k*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.k*2),
            # nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.k*2, self.k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.k),
            # nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.k, self.k, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.k),
            # nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Image_MI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.noise_size = args.noise_structure
        if args.dataset in ['cifar', 'cifar100']:
            self.a = 3
            self.b = 36688
        else:
            self.a = 1
            self.b = 9152
        self.k = self.noise_size[1]
        self.convnet_1 = nn.Sequential(
            nn.Conv2d(self.a, 5, kernel_size=4),  # 3 color channels in the input image, 6 output channels
            nn.BatchNorm2d(5),  # normalize
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.convnet_2 = nn.Sequential(
            nn.Conv2d(self.k, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.fcnet = nn.Sequential(
            nn.Linear(self.b, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # change the dimention
            nn.Linear(64, 1)
        )
        # initialize weight
        self._initialize_weights()
    def forward(self, x, r):
        out1 = self.convnet_1(x)
        out1 = out1.view(out1.size(0), -1)
        out2 = r.view(r.size(0), -1)
        out = torch.cat((out1, out2), dim=1)
        out = self.fcnet(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Text_MI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.noise_size = args.noise_structure
        if args.dataset == 'activity':
            self.a = 561
        else:
            self.a = 105
        self.fc1 = nn.Sequential(
            nn.Linear(self.a, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.fcnet = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            # change the dimention
            nn.Linear(32, 1)
        )
        # initialize weight
        self._initialize_weights()
    def forward(self, x, r):
        out1 = self.fc1(x)
        out1 = out1.view(out1.size(0), -1)
        out2 = r.view(r.size(0), -1)
        out = torch.cat((out1, out2), dim=1)
        out = self.fcnet(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class Text_Denoiser(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.noise_size = args.noise_structure
        self.layer0 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class View(torch.nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

#Currently the mimic encoder is configured for layer 1 attack, may need to manually shift to fit other layers
class Cifar_tinyEncoder(nn.Module):
    def __init__(self,  feature_size):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.MaxPool2d(2, 2),
            #nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Mnist_tinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            #nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Income_Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(105, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Activity_Encoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(561, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Activity_tinyEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(561, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Income_tinyEncoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(105, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class ImageNet_Encoder(nn.Module):
    def __init__(self, num_step=0):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            #nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            #nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer9 = View(-1, 256 * 4 * 16)
        self.layer10 = nn.Sequential(
            nn.Linear(4096 * 4, 5739),
            nn.BatchNorm1d(5739),
            nn.ReLU(),
        )
        self.layers = []
        self.layers.append(self.layer0)
        self.layers.append(self.layer1)
        self.layers.append(self.layer2)
        self.layers.append(self.layer3)
        self.layers.append(self.layer4)
        self.layers.append(self.layer5)
        self.layers.append(self.layer6)
        self.layers.append(self.layer7)
        self.layers.append(self.layer8)
        self.layers.append(self.layer9)
        self.layers.append(self.layer10)
        self.layers = self.layers[0:num_step+1]

    def forward(self, x: torch.Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out