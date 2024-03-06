# @Author: zechenghe
# @Date:   2019-01-21T12:01:09-05:00
# @Last modified by:   zechenghe
# @Last modified time: 2019-02-01T14:50:41-05:00

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


def accuracy(predictions, labels):

    if not (predictions.shape == labels.shape):
        print ("predictions.shape ", predictions.shape, "labels.shape ", labels.shape)
        raise AssertionError

    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def pseudoInverse(W):
    return np.linalg.pinv(W)


def getImgByClass(Itr, C = None):

    if C == None:
        return Itr.next()

    while (True):
        img, label = Itr.next()
        if label == C:
            break
    return img, label


def clip(data):
    data[data > 1.0] = 1.0
    data[data < 0.0] = 0.0
    return data


def preprocess(data):

    size = data.shape
    NChannels = size[1]
    assert NChannels == 1 or NChannels == 3
    if NChannels == 1:
        mu = 0.5
        sigma = 0.5
        data = (data - mu) / sigma
    elif NChannels == 3:
        mu = [0.485, 0.456, 0.406]
        sigma = [0.229, 0.224, 0.225]
        for i in range(NChannels):
            data[:, i, :, :] = (data[:, i, :, :] - mu[i]) / sigma[i]
    assert data.shape == size
    return data


def deprocess(data):

    assert len(data.size()) == 4

    BatchSize = data.size()[0]
    assert BatchSize == 1

    NChannels = data.size()[1]
    if NChannels == 1:
        mu = torch.tensor([0.5], dtype=torch.float32)
        sigma = torch.tensor([0.5], dtype=torch.float32)
    elif NChannels == 3:
        mu = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        sigma = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    else:
        print ("Unsupported image in deprocess()")
        exit(1)

    Unnormalize = transforms.Normalize((-mu / sigma).tolist(), (1.0 / sigma).tolist())
    return clip(Unnormalize(data[0,:,:,:]).unsqueeze(0))


def evalTest(testloader, net, args=None, gpu=True):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        NBatch += 1
        batchX, batchY = data
        if gpu:
            batchX = batchX.to(args.device)
            batchY = batchY.to(args.device)
        logits = net.forward(batchX)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
    accTest = acc / NBatch
    print ("Test accuracy: ", accTest )#, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest


def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def l2loss(x):
    return (x**2).mean()


def l1loss(x):
    return (torch.abs(x)).mean()


def getL1Stat(net, x):
    for layer in net.layerDict:
        targetLayer = net.layerDict[layer]
        layerOutput = net.getLayerOutput(x, targetLayer)
        print ("Layer " + layer + ' l1 loss:', l1loss(layerOutput).cpu().detach().numpy())


def getModule(net, blob):
    modules = blob.split('.')
#    print "Target layer: ", modules
#    if len(modules) == 1:
#        return net._modules.get(blob)
#    else:

    curr_module = net
    print (curr_module)
    for m in modules:
        curr_module = curr_module._modules.get(m)
    return curr_module


def getLayerOutputHook(module, input, output):
    if not hasattr(module, 'activations'):
        module.activations = []
    module.activations.append(output)


def getHookActs(model, module, input):
    if hasattr(module, 'activations'):
        del module.activations[:]
    _ = model.forward(input)
    assert(len(module.activations) == 1)
    return module.activations[0]


def saveImage(img, filepath):
    torchvision.utils.save_image(img, filepath)


def apply_noise(input, noise_type, noise_level, mean=0.0, gpu=True, args=None):

    if noise_type == 'Gaussian':
        noise = torch.randn(input.size()) * noise_level + mean
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'Laplace':
        noise = np.random.laplace(
            loc= mean,
            scale = noise_level,
            size = input.size()
        )
        noise = torch.tensor(noise, dtype = torch.float)
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'dropout':
        mask = np.random.choice([0.0, 1.0], size=input.size(), replace=True, p=[noise_level, 1-noise_level])
        mask = torch.tensor(mask, dtype = torch.float)
        mask = mask.cuda() if gpu else mask
        output = input * mask

    elif noise_type == 'dropout-non-zero':
        input_list = input.detach().cpu().numpy().reshape([-1])
        output = input_list.copy()

        for i in range(len(input_list)):
            if input_list[i] > 0:
                if np.random.rand() < noise_level:
                    output[i] = -1.0
            else:
                output[i] = -np.random.rand() * 10.0
        output = torch.tensor(np.array(output).reshape(input.size()), dtype = torch.float)
        output = output.cuda() if gpu else output

    elif noise_type == 'redistribute':
        input_list = input.detach().cpu().numpy().reshape([-1])
        idx = np.argsort(input_list)
        map = np.linspace(start=0.0, stop=1.0, num=len(input_list))

        output = [0]*len(input_list)
        for i in range(len(idx)):
            if input_list[idx[i]] != 0 and np.random.rand() > noise_level:
                output[idx[i]] = 1.0
        output = torch.tensor(np.array(output).reshape(input.size()), dtype = torch.float)
        output = output.cuda() if gpu else output

        #print "input", input
        #print "output", output

    elif noise_type == 'impulse':
        noise = np.random.choice([0.0, 1.0], size=input.size(), replace=True, p=[1-noise_level, noise_level])
        noise = torch.tensor(noise, dtype = torch.float)
        noise = noise.cuda() if gpu else noise
        output = input + noise

    elif noise_type == 'noise_gen' or 'noise_gen_opt':
        noise_dir = 'noise' + ('_opt' if noise_type == 'noise_gen_opt' else "") + '/' + args.dataset + '/'
        noise_file_name = args.noise_sourceLayer + '-' + args.noise_targetLayer + '-' + str(round(noise_level, 2))

        noise = np.load(noise_dir + noise_file_name + '.npy')
        noise = torch.tensor(noise, dtype = torch.float)

        batch_size = input.size()[0]
        noise = torch.cat(batch_size * [noise])
        noise = noise.cuda() if gpu else noise
        output = input + noise

    else:
        print ("Unsupported Noise Type: ", noise_type)
        exit(1)

    return output


def evalTestSplitModel(testloader, netEdge, netCloud, layer, gpu, noise_type = None, noise_level = 0.0, args=None):
    testIter = iter(testloader)
    acc = 0.0
    NBatch = 0
    for i, data in enumerate(testIter, 0):
        batchX, batchY = data
        if gpu:
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        if hasattr(args, 'add_noise_to_input') and args.add_noise_to_input:
            batchX = apply_noise(batchX, noise_type, noise_level, gpu=gpu, args=args)

        try:
            edgeOutput = netEdge.getLayerOutput(batchX, netEdge.layerDict[layer]).clone()
        except Exception as e:
            #print "Except in evalTestSplitModel getLayerOutput, this is a Edge-only model"
            #print str(e)
            edgeOutput = netEdge.forward(batchX).clone()

        if noise_type != None and not (hasattr(args, 'add_noise_to_input') and args.add_noise_to_input):
            edgeOutput = apply_noise(edgeOutput, noise_type, noise_level, gpu=gpu, args=args)

        #cloudOuput = net.forward(batchX)
        logits = netCloud.forward_from(edgeOutput, layer)

        #softmax = nn.Softmax().cuda()
        #prob = softmax(logits)
        #print prob[:100,:].max(dim=1)

        if gpu:
            pred = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            groundTruth = batchY.cpu().detach().numpy()
        else:
            pred = np.argmax(logits.detach().numpy(), axis = 1)
            groundTruth = batchY.detach().numpy()
        acc += np.mean(pred == groundTruth)
        NBatch += 1

    accTest = acc / NBatch
    #print "Test accuracy: ", accTest #, "NBatch: ", NBatch, "pred == groundTruth.shape", (pred == groundTruth).shape
    return accTest


def get_PSNR(refimg, invimg, peak = 1.0):
    refimg = refimg.detach().cpu().numpy().squeeze()
    invimg = invimg.detach().cpu().numpy().squeeze()
    psnr = 10*np.log10(peak**2 / np.mean((refimg - invimg)**2))
    return psnr