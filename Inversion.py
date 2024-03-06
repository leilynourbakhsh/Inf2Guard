import time
import math
import os
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
#from net import *
from inverseUtil import *


def inverse(DATASET = 'CIFAR10', network = 'CIFAR10CNN', NIters = 500, imageWidth = 32, inverseClass = None,
        imageHeight = 32, imageSize = 32*32, NChannels = 3, NClasses = 10, layer = 'conv22',
        BatchSize = 32, learningRate = 1e-3, NDecreaseLR = 20, eps = 1e-3, lambda_TV = 1e3, lambda_l2 = 1.0,
        AMSGrad = True, model = None,
        save_img_dir = "inverted/CIFAR10/MSE_TV/", saveIter = 10, gpu = True, validation=False):

    print ("DATASET: ", DATASET)
    print ("inverseClass: ", inverseClass)

    assert inverseClass < NClasses

    if DATASET == 'CIFAR10':

        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        Normalize = transforms.Normalize(mu.tolist(), sigma.tolist())
        Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())

        tsf = {
            'train': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ]),
            'test': transforms.Compose(
            [
            transforms.ToTensor(),
            Normalize
            ])
        }

        trainset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = True,
                                        download=True, transform = tsf['train'])
        testset = torchvision.datasets.CIFAR10(root='./data/CIFAR10', train = False,
                                       download=True, transform = tsf['test'])


    print ("len(trainset) ", len(trainset))
    print ("len(testset) ", len(testset))
    x_train, y_train = trainset.data, trainset.targets,
    x_test, y_test = testset.data, testset.targets,

    print ("x_train.shape ", x_train.shape)
    print ("x_test.shape ", x_test.shape)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1000,
                                      shuffle = False, num_workers = 1)
    inverseloader = torch.utils.data.DataLoader(testset, batch_size = 1,
                                      shuffle = False, num_workers = 1)
    trainIter = iter(trainloader)
    testIter = iter(testloader)
    inverseIter = iter(inverseloader)

    net = model

    net.eval()
    print ("Validate the model accuracy...")
    if validation:
        accTest = evalTest(testloader, net, gpu = gpu)

    targetImg, _ = getImgByClass(inverseIter, C = inverseClass)
    print ("targetImg.size()", targetImg.size())

    deprocessImg = deprocess(targetImg.clone())

    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    torchvision.utils.save_image(deprocessImg, save_img_dir + str(inverseClass) + '-ref.png')

    if gpu:
        targetImg = targetImg.to(args.device)
        softmaxLayer = nn.Softmax().to(args.device)

    if layer == 'prob':
        reflogits = net.forward(targetImg)
        refFeature = softmaxLayer(reflogits)
    elif layer == 'label':
        refFeature = torch.zeros(1,NClasses)
        refFeature[0, inverseClass] = 1
    else:
        targetLayer = net.layerDict[layer]
        refFeature = net.getLayerOutput(targetImg, targetLayer)

    print ("refFeature.size()", refFeature.size())

    if gpu:
        xGen = torch.zeros(targetImg.size(), requires_grad = True, device=args.device)
    else:
        xGen = torch.zeros(targetImg.size(), requires_grad = True)

    optimizer = optim.Adam(params = [xGen], lr = learningRate, eps = eps, amsgrad = AMSGrad)

    for i in range(NIters):

        optimizer.zero_grad()
        if layer == 'prob':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = ((xFeature - refFeature)**2).mean()
        elif layer == 'label':
            xlogits = net.forward(xGen)
            xFeature = softmaxLayer(xlogits)
            featureLoss = - torch.log(xFeature[0, inverseClass])
        else:
            xFeature = net.getLayerOutput(xGen, targetLayer)
            featureLoss = ((xFeature - refFeature)**2).mean()

        TVLoss = TV(xGen)
        normLoss = l2loss(xGen)

        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss #- 1.0 * conv1Loss

        totalLoss.backward(retain_graph=True)
        optimizer.step()

        print ("Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())

    # save the final result
    imgGen = xGen.clone()
    imgGen = deprocess(imgGen)
    torchvision.utils.save_image(imgGen, save_img_dir + str(inverseClass) + '-inv.png')

    print ("targetImg l1 Stat:")
    getL1Stat(net, targetImg)
    print ("xGen l1 Stat:")
    getL1Stat(net, xGen)
    print ("Done")