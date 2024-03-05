import torch
import torch.nn as nn
import os

from exp4_training import adjust_learning_rate
from exp4 import get_dataloaders
from defense import load_model
from exp4_adv import train_FE_INF,eval_atk

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atk_criterion = nn.CrossEntropyLoss().to(device)

adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
save_dir = os.path.join(os.getcwd(), 'Census-w/o')

total_epoch=50
lr=1e-4

def get_INF_defense():
    FE = load_model(os.path.join(save_dir, "FE_defense.pth"))
    INF = load_model(os.path.join(save_dir, "INF_defense.pth"))
    if torch.cuda.is_available():
        FE = FE.cuda()
        INF = INF.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch,lr)
            FE, INF = train_FE_INF(FE, INF, adv_trainloader, current_lr, device)
            #test_FE_INF(FE, INF, atk_test_dl,device)
            eval_atk(FE,INF,adv_testloader,atk_criterion,device)
    except KeyboardInterrupt:
        pass

    return FE,INF

if __name__ == "__main__":
    FE,INF=get_INF_defense()