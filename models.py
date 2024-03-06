from Utils import helpers
from Utils import classifiers

import torch as ch
from Utils import *
from Defense.Generators import Generators
from Utils.encoder import *


def generate_Models(args):
    if args.dataset == 'cifar':
        encoder = Cifar_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Cifar10_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'cifar100':
        encoder = Cifar_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Cifar100_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'mnist':
        encoder = Mnist_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Mnist_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'letters':
        encoder = Mnist_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Letters_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'fashion':
        encoder = Mnist_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Mnist_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'income':
        encoder = Income_Encoder()
        classsifier = classifiers.Income_Classifier()
    elif args.dataset == 'activity':
        encoder = Activity_Encoder()
        classsifier = classifiers.Activity_Classifier()
    elif args.dataset == 'imagenet':
        encoder = ImageNet_Encoder()
        classsifier = classifiers.ImageNet_Classifier()
    return encoder, classsifier


class enc_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder, self.classifier = generate_Models(args)
        if args.noise_type == 'phoni':
            self.generators = Generators(args)
        if args.attack_type == 'denoiser':
            if (args.dataset in ['cifar', 'mnist', 'fashion', 'cifar100']):
                self.denoiser = Denoiser(args)
            else:
                self.denoiser = Text_Denoiser(args)
            #self.noiser = Noiser(args)
        if args.noise_knowledge == 'pattern' and args.noise_type == 'phoni':
            self.atk_generators = Generators(args)
        if args.atk_model_knowledge == 'pattern':
            if args.dataset in ['cifar', 'cifar100']:
                self.decoder = Cifar_Encoder(num_step=args.attack_layer)
            elif args.dataset in ['mnist', 'fashion', 'letters']:
                self.decoder = Mnist_Encoder(num_step=args.attack_layer)
            elif args.dataset == 'activity':
                self.decoder = Activity_Encoder()
            elif args.dataset == 'income':
                self.decoder = Income_Encoder()
        elif args.atk_model_knowledge == 'none':
            if args.dataset in ['cifar', 'cifar100']:
                self.decoder = Cifar_tinyEncoder(args.noise_structure[1])
                self.MI_estimator = Image_MI(args)
            elif args.dataset in ['mnist', 'fashion', 'letters']:
                self.decoder = Mnist_tinyEncoder()
                self.MI_estimator = Image_MI(args)
            elif args.dataset == 'activity':
                self.decoder = Activity_tinyEncoder()
                self.MI_estimator = Text_MI(args)
            elif args.dataset == 'income':
                self.decoder = Income_tinyEncoder()
                self.MI_estimator = Text_MI(args)
        self.criterion = nn.CrossEntropyLoss().to(args.device)

    def forward(self, input, target):
        rep_out = self.encoder(input)
        #rep_out = self.noiser(rep_out)
        out = self.classifier(rep_out)
        loss = self.criterion(ch.sigmoid(out), target)
        acc = helpers.accuracy(out, target)[0]
        return out, loss, acc

    def rep_forward(self, rep_out, target):
        out = self.classifier(rep_out)
        sig = ch.sigmoid(out)
        loss = self.criterion(sig, target)
        acc = helpers.accuracy(out, target)[0]
        return loss, acc

    def get_rep(self, rep_out, target):
        out = self.encoder(rep_out)
        loss = self.criterion(ch.sigmoid(self.classifier(out)), target)
        acc = helpers.accuracy(self.classifier(out), target)[0]
        return out, loss, acc


class dec_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = Denoiser()
        self.criterion = nn.MSELoss().to(args.device)

    def forward(self, input, rep_out):
        est = self.decoder(rep_out)
        est = est.view(-1, 3, 32, 32)
        loss = self.criterion(ch.sigmoid(est), input)
        acc = helpers.accuracy(est, input)[0]
        return est, loss
