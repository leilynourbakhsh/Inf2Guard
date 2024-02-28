import torch
import torch as ch
from torch.optim import Adam

from Utils.helpers import AverageMeter
from inverseUtil import *
from tqdm import tqdm as tqdm


class Generators(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.generators = []
        for i in range(args.num_class):
            self.generators.append(Generator(args, i))

    def generate(self, cla, batch_size, target, args):
        adv, loss = self.generators[target].generate(cla, batch_size, args)
        return adv, loss

    def get_dummy(self, batch_size, args):
        dummy = []
        for i in range(batch_size):
            for j in range(args.num_class):
                adv = self.generators[j].generate(noise_size=1, args=args)
                dummy.append(adv)
        return dummy

    def gen_train(self, cla, args):
        for gen in self.generators:
            self.eval()
            gen.train()
            gen.train_gen(cla, args)


class Generator(nn.Module):
    def __init__(self, args, target):
        super().__init__()
        self.mean = ch.autograd.Variable(torch.zeros(args.noise_structure[1:]).to(args.device), requires_grad=True)
        self.var = ch.autograd.Variable(torch.zeros(args.noise_structure[1:]).to(args.device), requires_grad=True)
        self.target = torch.tensor(target)
        self.optimizer = optim.Adam([self.mean, self.var], lr=args.atk_lr, betas=(0.0, 0.0))

    def get_weights(self):
        return [self.mean, self.var]

    def forward(self, cla, noise_size, args):
        adv_std = F.softplus(self.var)
        rand_noise = torch.randn(noise_size, *adv_std.shape, device=args.device)
        adv = torch.tanh(self.mean + rand_noise * adv_std)
        negative_logp = (rand_noise ** 2) / 2. + (adv_std + 1e-8).log() + (1 - adv ** 2 + 1e-8).log()
        entropy = negative_logp.mean()  # entropy
        target = self.target * torch.ones(noise_size, device=args.device).to(ch.long)
        loss = - F.cross_entropy(cla(adv), target) + args.lam * entropy
        return adv, loss

    def generate(self, noise_size, args):
        adv_std = F.softplus(self.var)
        rand_noise = torch.randn(noise_size, *adv_std.shape, device=args.device)
        adv = torch.tanh(self.mean + rand_noise * adv_std)
        return adv

    def train_gen(self, cla, args):
        ep_list = list(range(0, args.phoni_epoch))
        iterator = tqdm(ep_list, total=args.phoni_epoch)
        loss_enc = AverageMeter()
        for i in iterator:
            adv, loss = self.forward(cla, args.phoni_size, args)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.mean.detach()
            self.var.detach()
            loss_enc.update(loss.item())
            desc = ('Generator:{0} | '
                    'Loss {Loss:.4f} | '
            .format(
                self.target,
                Loss=loss_enc.avg))
            iterator.set_description(desc)
            iterator.refresh()
        iterator.close()
