import copy
import random

import numpy as np

import os
from Attacks.attacks import attack, attack_inversion
from Defense.Generators import Generators
from Defense.OurDefense import *
from torch.optim import Adam
from tqdm import tqdm as tqdm
from Utils.helpers import AverageMeter, get_mse, jsd_MI
from skimage.metrics import structural_similarity as compare_ssim
import pickle


def train_model(models, loaders, args, dataset=None):
    print('\n\nBegin Training\n')
    models.train()
    train_loader, val_loader = loaders
    opts = []
    enc_opt = Adam(models.encoder.parameters(), args.lr)
    cla_opt = Adam(models.classifier.parameters(), args.lr)
    opts.append(enc_opt)
    opts.append(cla_opt)
    if not args.pretrain:
        for i in range(0, args.epoch):
            train_loop(train_loader, models, opts, args, i)
        with open(os.path.join(r'./Output/', str(args.dataset) + '/encoder' + str(args.attack_layer) + '.pkl'), "wb") as f:
            pickle.dump(models.encoder, f)
        with open(os.path.join(r'./Output/', str(args.dataset) + '/cla' + str(args.attack_layer) + '.pkl'), "wb") as f:
            pickle.dump(models.classifier, f)
    else:
        print('Loading pre-trained model\n')
        with open(os.path.join(r'./Output/', str(args.dataset) + '/encoder' + str(args.attack_layer) + '.pkl'), "rb") as f:
            models.encoder = pickle.load(f)
        with open(os.path.join(r'./Output/', str(args.dataset) + '/cla' + str(args.attack_layer) + '.pkl'), "rb") as f:
            models.classifier = pickle.load(f)
    if args.noise_type == 'phoni':
        print('\nBegin Phoni Training: Phoni num ' + str(args.phoni_num) + ' size ' +
              str(args.phoni_size) + ' epoch ' + str(args.phoni_epoch) + '\n')
        train_phoni(models, args)
    if args.atk_model_knowledge != 'exact':
        dec_opt = Adam(models.decoder.parameters(), args.lr)
        MI_opt = Adam(models.MI_estimator.parameters(), args.lr)
        print('\nBegin defender encoder Training\n')
        if not args.pretrain:
            for i in range(0, int(args.epoch * 0.3)+1):
                train_decoder(train_loader, models, dec_opt, i, args)
                with open(os.path.join(r'./Output/', str(args.dataset) + '/decoder' + str(args.attack_layer) + '.pkl'),
                      "wb") as f:
                    pickle.dump(models.decoder, f)
        else:
            print('\nDefender encoder Loaded\n')
            with open(os.path.join(r'./Output/', str(args.dataset) + '/decoder' + str(args.attack_layer) + '.pkl'), "rb") as f:
                models.decoder = pickle.load(f)

        if args.MI != 'DP': #Flag not working sometimes, may need to manually comment
            print('\nBegin MI Training\n')
            for i in range(0, 5):
                train_MI(train_loader, models, dec_opt, MI_opt, args, i)
    if args.attack_type == 'denoiser' and args.noise_knowledge != 'none':
        print('\nBegin attacker denoiser Training\n')
        for j in range(0, args.atk_itr):
            opt = Adam(models.denoiser.parameters(), args.lr)
            denoise(train_loader, models, opt, args, j)
    print('\nBegin Eval\n')
    return eval_loop(val_loader, models, args, dataset)
    # train_decoder(train_loader, models, itr=500, lr=0.05)


def train_phoni(models, args):
    models.eval()
    models.generators.train()
    models.generators.gen_train(models.classifier, args)


def train_loop(train_loader, models, opts, args, epoch):
    models.to(args.device)
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    enc_opt, cla_opt = opts
    loss_enc = AverageMeter()
    acc_enc = AverageMeter()
    for i, (input, target) in iterator:
        input = input.to(args.device)
        target = target.to(args.device)
        if args.data_aug:
            input = preprocess(input)
        rep_out = models.encoder.forward(input)
        loss, acc = models.rep_forward(rep_out, target)
        enc_opt.zero_grad()
        cla_opt.zero_grad()
        loss.backward()
        enc_opt.step()
        cla_opt.step()
        _, loss, acc = models.forward(input, target)
        loss_enc.update(loss.item(), input.size(0))
        acc_enc.update(acc.item(), input.size(0))
        desc = ('Epoch:{0} | '
                'Loss {Loss:.4f} | '
                'prec {prec:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg,
            prec=acc_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
    iterator.close()


def train_decoder(train_loader, models, opt, epoch, args):
    models.eval()
    models.decoder.train()
    num_samples = int(len(train_loader) * args.atk_sample)
    iterator = tqdm(enumerate(train_loader), total=num_samples)
    loss_enc = AverageMeter()
    acc_enc = AverageMeter()
    for i, (input, target) in iterator:
        input = input.to(args.device)
        target = target.to(args.device)
        if args.data_aug:
            input = preprocess(input)
        rep_out = models.decoder.forward(input)
        loss, acc = models.rep_forward(rep_out, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        rep_out = models.decoder.forward(input)
        loss, acc = models.rep_forward(rep_out, target)
        loss_enc.update(loss.item(), input.size(0))
        acc_enc.update(acc.item(), input.size(0))
        desc = ('Epoch:{0} | '
                'Loss {Loss:.4f} | '
                'prec {prec:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg,
            prec=acc_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
        if i >= num_samples:
            break
    iterator.close()


def eval_loop(eval_loader, models, args, dataset=None):
    models.eval()
    iterator = tqdm(enumerate(eval_loader), total=len(eval_loader))
    loss_enc = AverageMeter()
    acc_enc = AverageMeter()
    target_images = []
    for i, (input, target) in iterator:
        input = input.to(args.device)
        target = target.to(args.device)
        if args.data_aug:
            input = preprocess(input)
        if args.atk_model_knowledge == 'exact':
            rep_out = models.encoder.forward(input)
        else:
            rep_out = models.decoder.forward(input)
        # if args.noise_type != 'none':
        #     rep_out = add_Noise(rep_out, dummy_data, args)
        # rep_out = models.noiser.forward(rep_out)
        if args.noise_type != 'none':
            rep_out = add_Noise(rep_out, get_Noise(models, args), args)
        loss, acc = models.rep_forward(rep_out, target)
        loss_enc.update(loss.item(), input.size(0))
        acc_enc.update(acc.item(), input.size(0))
        desc = ('Eval    | '
                'Loss {Loss:.4f} | '
                'prec {prec:.4f} | '
        .format(
            Loss=loss_enc.avg,
            prec=acc_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
        target_images.append(ch.unsqueeze(input[0], 0))
        target_images.append(ch.unsqueeze(input[1], 0))
    iterator.close()

    if args.attack_type != 'none':
        atk_images, MSEs, SSIMs, PSNRs, losses = [], [], [], [], []
        dummy_data = None
        if args.noise_type == 'phoni' and args.noise_knowledge == 'pattern':
            dummy_data = models.atk_generators.get_dummy(args.phoni_size, args)
        elif args.noise_type != 'none':
            dummy_data = get_Noise(models, args)
        iterator = tqdm(enumerate(target_images), total=args.num_attacked)
        for j, target_image in iterator:
        #for j in range(args.num_attacked):
            if args.multi_target:
                atk_image, MSE, SSIM, PSNR, atk_loss = attack(target_images[j], models, args, dummy_data, j, dataset)
            else:
                atk_image, MSE, SSIM, PSNR, atk_loss = attack(target_images[0], models, args, dummy_data, j)
            atk_images.append(atk_image)
            MSEs.append(MSE)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
            losses.append(atk_loss)
            desc = ('Attack  | '
                    'MSE {mse:.4f} | '
                    'SSIM {ssim:.4f} | '
                    'PSNR {psnr:.4f} | '
                    'Loss {loss:.4f} | '
            .format(
                mse=np.round(np.mean(MSEs), 2),
                ssim=np.round(np.mean(SSIMs), 2),
                psnr=np.round(np.mean(PSNRs), 2),
                loss=np.round(np.mean(atk_loss), 2)))
            iterator.set_description(desc)
            iterator.refresh()
            if j >= args.num_attacked:
                break
        iterator.close()

        print('The mean MSE is : ', np.round(np.mean(MSEs), 2))
        print('The mean SSIM is : ', np.round(np.mean(SSIMs), 2))
        print('The mean PSNR is : ', np.round(np.mean(PSNRs), 2))
        print('The mean Attack Loss is : ', np.round(np.mean(atk_loss), 2))
        text = ''
        text = text + str(args)
        text = text + '\nThe mean MSE is : ' + str(np.round(np.mean(MSEs), 2)) \
               + '\nThe mean SSIM is : ' + str(np.round(np.mean(SSIMs), 2)) \
               + '\nThe mean PSNR is : ' + str(np.round(np.mean(PSNRs), 2)) \
               + '\nThe mean atta' \
                 'ck feature loss is : ' + str(np.round(np.mean(atk_loss), 2)) \
               + '\nThe test acc is : ' + str(np.round(acc_enc.avg, 2))
        output_dir = r'./Output/'
        output_dir = os.path.join(output_dir, args.dataset + '/' + args.image_names)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'logs.txt')
        if args.dataset in ['activity', 'income']:
            target_file = os.path.join(output_dir, 'xGen.csv')
            atk_arrays = [tensor.cpu().detach().numpy() for tensor in atk_images[:args.num_attacked]]
            atk_array = np.array(atk_arrays[:args.num_attacked])
            flat_images = atk_array.reshape(atk_array.shape[0], -1)
            np.savetxt(target_file, flat_images, delimiter=",", fmt='%s')
            ori_file = os.path.join(output_dir, 'xOri.csv')
            atk_arrays = [tensor.cpu().detach().numpy() for tensor in target_images[:args.num_attacked]]
            atk_array = np.array(atk_arrays[:args.num_attacked])
            flat_images = atk_array.reshape(atk_array.shape[0], -1)
            np.savetxt(ori_file, flat_images, delimiter=",", fmt='%s')
        # Open the file in write mode and save the text
        with open(output_file, 'w') as file:
            file.write(text)
        return np.round(np.mean(MSEs), 2), np.round(np.mean(SSIMs), 2), np.round(np.mean(PSNRs), 2), np.round(np.mean(atk_loss), 2), np.round(acc_enc.avg)


def denoise(train_loader, models, opt, args, epoch):
    models.eval()
    models.denoiser.train()
    num_samples = int(len(train_loader) * args.atk_sample)
    iterator = tqdm(enumerate(train_loader), total=num_samples)
    loss_enc = AverageMeter()
    # dummy_data = get_Noise(models.classifier, args)
    for i, (input, target) in iterator:
        input = input.to(args.device)
        target = target.to(args.device)
        if args.data_aug:
            input = preprocess(input)
        rep_out = models.encoder.forward(input).detach()
        # rep_out = add_Noise(rep_out, dummy_data, args)
        # rep_out = models.noiser.forward(rep_out)
        if args.noise_type == 'phoni' and args.noise_knowledge == 'pattern':
            dummy_data = models.atk_generators.get_dummy(args.phoni_size, args)
            rep_out = add_Noise(rep_out, dummy_data, args)
        elif args.noise_type != 'none':
            dummy_data = get_Noise(models, args)
            rep_out = add_Noise(rep_out, dummy_data, args)
        rep_gen = models.denoiser.forward(rep_out)
        ori_rep = models.encoder.forward(input).detach()
        decoder_loss = ((rep_gen - ori_rep) ** 2).mean()
        opt.zero_grad()
        decoder_loss.backward()
        opt.step()
        loss_enc.update(decoder_loss.item(), input.size(0))
        desc = ('Denoiser:{0} | '
                'Loss {Loss:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
        if i >= num_samples:
            break
    iterator.close()


def train_MI(train_loader, models, opt_decoder, opt_MI, args, epoch):
    models.eval()
    models.MI_estimator.train()
    models.decoder.train()
    num_samples = int(len(train_loader) * args.atk_sample)
    iterator = tqdm(enumerate(train_loader), total=num_samples)
    loss_enc = AverageMeter()
    loss_MI = AverageMeter()
    loss_total = AverageMeter()
    avg_acc = AverageMeter()
    for i, (input, target) in iterator:
        input = input.to(args.device)
        target = target.to(args.device)
        if args.data_aug:
            input = preprocess(input)
        aux1 = input[1:].clone()
        aux2 = input[0].clone().unsqueeze(0)
        x_prime = torch.cat((aux1, aux2), dim=0)
        ori_rep = models.decoder.forward(input)
        rep_out = ori_rep
        if args.noise_type == 'phoni' and args.noise_knowledge == 'pattern':
            dummy_data = models.atk_generators.get_dummy(args.phoni_size, args)
            rep_out = add_Noise(ori_rep.clone(), dummy_data, args)
        elif args.noise_type != 'none':
            dummy_data = get_Noise(models, args)
            rep_out = add_Noise(ori_rep.clone(), dummy_data, args)
        #ori_rep = models.decoder.forward(input)
        loss, acc = models.rep_forward(rep_out, target)
        #loss2, acc2 = models.rep_forward(ori_rep, target)
        #loss = args.lam*(loss + loss2)/2
        mi_value = -(1.-args.lam)*jsd_MI(models.MI_estimator, input, ori_rep, x_prime)
        total_loss = loss + mi_value
        opt_decoder.zero_grad()
        opt_MI.zero_grad()
        #loss.backward(retain_graph=True)
        #mi_value.backward(retain_graph=True)
        total_loss.backward()
        opt_MI.step()
        opt_decoder.step()
        loss_enc.update(loss.item(), input.size(0))
        loss_MI.update(mi_value.item(), input.size(0))
        loss_total.update(total_loss.item(), input.size(0))
        avg_acc.update(acc.item(), input.size(0))
        desc = ('MI Defense: {0} | '
                'Loss {Loss:.4f} | '
                'MI {MI:.4f} | '
                'Total Loss {Total:.4f} | '
                'prec {accuracy:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg,
            MI=loss_MI.avg,
            Total=loss_total.avg,
            accuracy=avg_acc.avg))
        iterator.set_description(desc)
        iterator.refresh()
        if i >= num_samples:
            break
    iterator.close()