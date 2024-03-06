import torch
import torch.nn as nn
import os

from exp4_training import adjust_learning_rate
from exp4 import get_dataloaders
from defense import load_model

from exp4_training import train_FE_CF,eval_clf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
atk_criterion = nn.CrossEntropyLoss().to(device)

adv_trainloader, adv_testloader = get_dataloaders(force_generate=False)
save_dir = os.path.join(os.getcwd(), 'Census-w/o')


clf_criterion = nn.BCEWithLogitsLoss().cuda()


total_epoch=50
lr=1e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def get_clf_defense():
    FE = load_model(os.path.join(save_dir, "FE_defense.pth"))
    CF = load_model(os.path.join(save_dir, "CF_defense.pth"))
    if torch.cuda.is_available():
        FE = FE.cuda()
        CF= CF.cuda()
    try:
        for epoch in range(total_epoch):
            print("epoch %d" % epoch)
            current_lr = adjust_learning_rate(epoch, lr)
            FE,CF= train_FE_CF(FE, CF, adv_trainloader, current_lr, device)
            eval_clf(FE, CF,adv_testloader, clf_criterion, device)
    except KeyboardInterrupt:
        pass

    return CF

if __name__ == "__main__":
    CF=get_clf_defense()