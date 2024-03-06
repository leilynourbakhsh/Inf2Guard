import torch as ch
from torch.optim import Adam
from inverseUtil import *


def get_Noise(models, args):
    if args.noise_type == 'gau':
        noise = gau_Noise(args)
    if args.noise_type == 'uni':
        noise = uni_Noise(args)
    if args.noise_type == 'lap':
        noise = lap_Noise(args)
    elif args.noise_type == 'phoni':
        noise = models.generators.get_dummy(args.phoni_size, args)
    return noise


def add_Noise(rep, dummy_data, args):
    for k in range(len(rep)):
        # print(rep[k].mean())
        rand_classes = np.random.choice(args.num_class, args.num_class, replace=False)
        rand_index = np.random.choice(int(len(dummy_data) / args.num_class), args.phoni_num)
        dat = rep[k]
        for i in range(args.phoni_num):
            rand_dummy = rand_classes[i] + (rand_index[i] * args.num_class)
            #noise = (dummy_data[rand_dummy] / dummy_data[rand_dummy].mean()) * rep[k].mean()
            dat = dat + (dummy_data[rand_dummy] * args.noise_scale)
            #print(dummy_data[rand_dummy].mean(), rep[k].mean())
        # print(dat.mean())
        # dat = dat / args.data_scale
        dat = (dat / dat.mean()) * rep[k].mean()
        rep[k] = dat
    return rep


def phoni_Noise(model, args):
    dummy_data = ch.randn(args.noise_structure).to(args.device).requires_grad_(True).to(ch.float)
    dummy_label = ch.arange(0.0, args.num_class)
    dummy_label = dummy_label.repeat(1, args.phoni_size).reshape(-1).to(args.device).requires_grad_(True).to(ch.long)
    optimizer = Adam([dummy_data], 0.05)
    criterion = nn.CrossEntropyLoss()

    for iters in range(args.phoni_epoch):
        optimizer.zero_grad()
        dummy_pred = model.forward(dummy_data).to(ch.float)
        dummy_ce = - args.alpha * criterion(dummy_pred, dummy_label) \
                   + (dummy_pred ** 2).mean()
        # + abs(abs(dummy_data).sum(axis=(1, 2, 3)).mean() - args.a.detach()) \
        # + abs(abs(dummy_data).sum(axis=(1, 2, 3)).var() - args.b.detach()) \
        # + abs(abs(dummy_data).var(axis=(1, 2, 3)).mean() - args.c.detach()) \
        # + abs(abs(dummy_data).var(axis=(1, 2, 3)).var() - args.d.detach())
        dummy_ce.backward()
        optimizer.step()
    # print(ch.softmax(model.forward(dummy_data).to(ch.float).cpu().detach(), dim=0)[0:5])
    return dummy_data


def gau_Noise(args):
    noise = torch.randn(args.noise_structure).to(args.device).requires_grad_(True).to(ch.float)
    return noise


def uni_Noise(args):
    noise = torch.FloatTensor(*args.noise_structure).uniform_(args.noise_a, args.noise_b).to(args.device).requires_grad_(True).to(ch.float)
    return noise


def lap_Noise(args):
    laplace_dist = ch.distributions.Laplace(args.noise_a, args.noise_b)
    noise = laplace_dist.sample(args.noise_structure).to(args.device).requires_grad_(True).to(ch.float)
    return noise
