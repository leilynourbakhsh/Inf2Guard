import csv

import torch as ch

import shutil
import dill
import os
import numpy as np
from subprocess import Popen, PIPE
from PIL import Image

from .tools import constants
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from textdata import ADULT, ACTIVITY, Lawschool, HealthHeritage, German


def has_attr(obj, k):
    """Checks both that obj.k exists and is not equal to None"""
    try:
        return (getattr(obj, k) is not None)
    except KeyError as e:
        return False
    except AttributeError as e:
        return False


def get_PSNR(refimg, invimg, peak=1.0):
    psnr = 10 * np.log10(peak ** 2 / np.mean((refimg - invimg) ** 2))
    return psnr


def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples // 2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1] * N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B * Q, -1).norm(dim=-1).view(B * Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2 * Q, *y_shape)).view(-1, *extender)
        grad = (l.view(2 * Q, B, *extender) * noise.view(2 * Q, B, *noise.shape[1:])).mean(dim=0)
    return grad


def calc_fadein_eps(epoch, fadein_length, eps):
    """
    Calculate an epsilon by fading in from zero.

    Args:
        epoch (int) : current epoch of training.
        fadein_length (int) : number of epochs to fade in for.
        eps (float) : the final epsilon

    Returns:
        The correct epsilon for the current epoch, based on eps=0 and epoch
        zero and eps=eps at epoch :samp:`fadein_length` 
    """
    if fadein_length and fadein_length > 0:
        eps = eps * min(float(epoch) / fadein_length, 1)
    return eps


def ckpt_at_epoch(num):
    return '%s_%s' % (num, constants.CKPT_NAME)


def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


def dummy_accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes)
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0]

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, False, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


class InputNormalize(ch.nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''

    def __init__(self, new_mean, new_std):
        super(InputNormalize, self).__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean) / self.new_std
        return x_normalized


class DataPrefetcher():
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        #self.stream = ch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        #with ch.cuda.stream(self.stream):
        self.next_input = self.next_input#.cuda(non_blocking=True)
        self.next_target = self.next_target#.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            #ch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break


def save_checkpoint(state, is_best, filename):
    ch.save(state, filename, pickle_module=dill)
    if is_best:
        shutil.copyfile(filename, filename + constants.BEST_APPEND)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ImageNet label mappings
def get_label_mapping(dataset_name, ranges):
    if dataset_name == 'imagenet':
        label_mapping = None
    elif dataset_name == 'restricted_imagenet':
        def label_mapping(classes, class_to_idx):
            return restricted_label_mapping(classes, class_to_idx, ranges=ranges)
    else:
        raise ValueError('No such dataset_name %s' % dataset_name)

    return label_mapping


def restricted_label_mapping(classes, class_to_idx, ranges):
    range_sets = [
        set(range(s, e + 1)) for s, e in ranges
    ]

    # add wildcard
    # range_sets.append(set(range(0, 1002)))
    mapping = {}
    for class_name, idx in class_to_idx.items():
        for new_idx, range_set in enumerate(range_sets):
            if idx in range_set:
                mapping[class_name] = new_idx
        # assert class_name in mapping
    filtered_classes = list(mapping.keys()).sort()
    return filtered_classes, mapping


def get_mse(actual, predicted):
    differences = np.subtract(actual, predicted)
    squared_differences = np.square(differences)
    return squared_differences.mean()


def income_loss(actual, predicted, args, dataset=None):
    x_ori = dataset.decode_batch(ch.tensor(actual, device=args.device), standardized=dataset.standardized)
    x_gen = dataset.decode_batch(ch.tensor(predicted, device=args.device), standardized=dataset.standardized)
    tolerance_map = dataset.create_tolerance_map(tol=args.tol)
    loss, _ = feature_wise_accuracy_score(x_ori, x_gen, tolerance_map, dataset.train_features)
    return loss


def feature_wise_accuracy_score(true_data, reconstructed_data, tolerance_map, train_features):
    """
    Calculates the categorical accuracy and in-tolerance-interval accuracy for continuous features per feature.

    :param true_data: (np.ndarray) The true/reference mixed-type feature vector.
    :param reconstructed_data: (np.ndarray) The reconstructed mixed-type feature vector.
    :param tolerance_map: (list or np.ndarray) A list with the same length as a single datapoint. Each entry in the list
        corresponding to a numerical feature in the data should contain a floating point value marking the
        reconstruction tolerance for the given feature. At each position corresponding to a categorical feature the list
        has to contain the entry 'cat'.
    :param train_features: (dict) A dictionary of the feature names per column.
    :return: (dict) A dictionary with the features and their corresponding error.
    """
    true_data = true_data.reshape(-1)
    reconstructed_data = reconstructed_data.reshape(-1)
    feature_errors = {}
    for feature_name, true_feature, reconstructed_feature, tol in zip(train_features.keys(), true_data, reconstructed_data, tolerance_map):
        if tol == 'cat':
            feature_errors[feature_name] = 0 if str(true_feature) == str(reconstructed_feature) else 1
        else:
            feature_errors[feature_name] = 0 if (float(true_feature) - tol <= float(reconstructed_feature) <= float(true_feature) + tol) else 1
    loss = sum(int(value) for value in feature_errors.values()) / len(feature_errors)
    return loss, feature_errors


def ADT_gen(rep, G, epsilon=8.0 / 255.0):
    mean, var = G
    rep_adv = ch.randn_like(rep)
    for k in range(len(rep)):
        adv_std = F.softplus(var)
        rand_noise = ch.randn_like(adv_std)
        adv = ch.tanh(mean + rand_noise * adv_std)
        # omit the constants in -logp
        x_in = rep[k]
        rep_adv[k] = x_in + 3 * adv
    return rep_adv


def jsd_MI(MI_estimator, x, z, x_prime):
    Ej = (-ch.nn.functional.softplus(-MI_estimator(x, z))).mean()
    Em = ch.nn.functional.softplus(MI_estimator(x_prime, z)).mean()
    return Ej - Em


class CustomDataset(Dataset):
    def __init__(self, csv_file, train=True, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        if train:
            self.data=self.data[:int(len(self.data) * 0.8)]
        else:
            self.data=self.data[int(len(self.data) * 0.8):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the entire row from the CSV data
        row = self.data.iloc[index].values.astype(float)

        # Separate features and label from the row
        features = row[:-1]
        label = row[-1]

        if self.transform:
            features = self.transform(features)

        return features, label


def get_other_data(args):
    textdata = {
        'income': ADULT,
        'activity': ACTIVITY,
        'German': German,
        'Lawschool': Lawschool,
        'HealthHeritage': HealthHeritage
    }
    data = textdata[args.dataset](device=args.device, random_state=args.random_seed)
    if args.dataset not in ['income', 'activity']:
        data.standardize()
    train_X, train_y = data.get_Xtrain(), data.get_ytrain()
    test_X, test_y = data.get_Xtest(), data.get_ytest()

    #X = data.decode_batch(train_X, standardized=data.standardized)

    train_features_tensor = train_X.clone().detach()
    train_labels_tensor = train_y.clone().detach()
    test_features_tensor = test_X.clone().detach()
    test_labels_tensor = test_y.clone().detach()

    # Create TensorDatasets
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_features_tensor, test_labels_tensor)

    # Define a batch size
    batch_size = 32

    # Create data loaders with transformations and device placement
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # Assuming 'device' is already defined
    # Move the data loaders to the CUDA device
    ##test_loader = [(features.to(device).float(), labels.to(device).long()) for features, labels in test_loader]
    return data, train_loader, test_loader


def write_batch(runs, MSEs, SSIMs, PSNRs, Losses, Accs, args):
    text = ''
    text = text + str(args)
    text = text + '\nTotal runs : ' + str(runs) \
           + '\nThe mean MSE is : ' + str(np.round(np.mean(MSEs), 2)) \
           + '\nThe mean SSIM is : ' + str(np.round(np.mean(SSIMs), 2)) \
           + '\nThe mean PSNR is : ' + str(np.round(np.mean(PSNRs), 2)) \
           + '\nThe mean atta' \
             'ck feature loss is : ' + str(np.round(np.mean(Losses), 2)) \
           + '\nThe test acc is : ' + str(np.round(np.mean(Accs), 2))
    output_dir = r'./Output/'
    output_dir = os.path.join(output_dir, args.dataset + '/' + args.image_names)
    dir_name = os.path.join(output_dir, 'run')
    os.makedirs(output_dir, exist_ok=True)
    suffix_counter = 0
    while True:
        # Create the directory name based on the base name and suffix
        directory_name = f"{dir_name}_{suffix_counter}"
        # Check if the directory already exists
        if not os.path.exists(directory_name):
            break
        # Increment the suffix counter
        suffix_counter += 1
    os.makedirs(directory_name, exist_ok=True)
    logs_file = os.path.join(directory_name, 'logs.txt')
    with open(logs_file, 'w') as file:
        file.write(text)
    mse_file = os.path.join(output_dir, 'mse.csv')
    with open(mse_file, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write data to the CSV file
        for item in MSEs:
            csv_writer.writerow([item])
        # Complete the writing process
        csv_file.close()
    ssim_file = os.path.join(output_dir, 'ssim.csv')
    with open(ssim_file, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write data to the CSV file
        for item in SSIMs:
            csv_writer.writerow([item])
        # Complete the writing process
        csv_file.close()
    psnr_file = os.path.join(output_dir, 'psnr.csv')
    with open(psnr_file, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write data to the CSV file
        for item in PSNRs:
            csv_writer.writerow([item])
        # Complete the writing process
        csv_file.close()
    acc_file = os.path.join(output_dir, 'acc.csv')
    with open(acc_file, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write data to the CSV file
        for item in Accs:
            csv_writer.writerow([item])
        # Complete the writing process
        csv_file.close()

