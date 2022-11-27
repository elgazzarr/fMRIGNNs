import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, Subset,  DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
import wandb
from networks.utils import LabelSmoothingCrossEntropy
from networks.static_models import *
from networks.dynamic_models import *
from copy import deepcopy
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, val_loader, config, device, a=None):

    model = model.to(device)

    '''model_hparams  = ["dropout", "dilation_exponential", "conv_channels", "layers", "adaptive", "readout",
    "jumping_knowledge",  "adj_norm",  "kernel_set", "bn", 'gcn', 'gcn_depth']

    model_params =  {key: config[key] for key in model_hparams}'''


    loss_function = LabelSmoothingCrossEntropy()
    optimizer = torch.optim.Adam(model.parameters(), config['lr'], weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 400, eta_min = 1e-5)

    # start a typical PyTorch training
    best_metric = -1
    best_metric_epoch = -1
    best_val_loss = 1000
    best_model = None
    epochs = 1000

    print('-'*30)
    print ('Training ... ')
    early_stop = 30
    es_counter = 0

    for epoch in range(epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_train_loss = 0

        for i , data in enumerate(tqdm(train_loader)):

            x = data.to(device)
            y = torch.tensor(data.y).type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            out = model(x)
            step_loss = loss_function(out, y, smoothing=config['ls'])
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()

        epoch_train_loss = epoch_train_loss/(i+1)
        lr_scheduler.step()
        val_loss, val_acc = validate_model(model, val_loader, device)
        print(f"epoch {epoch + 1} train loss: {epoch_train_loss:.4f}")

        #if val_acc > best_metric and epoch>15:
        if val_loss < best_val_loss:
            best_metric = val_acc
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            best_model = deepcopy(model)
            print("saved new best metric model")
            es_counter = 0
        else:
            es_counter += 1

        if es_counter > early_stop:
            print('No loss improvment.')
            break

        wandb_metric = {'train_loss':epoch_train_loss, 'val_loss':val_loss, 'val_acc': val_acc, 'best_val_loss': best_val_loss }
        wandb.log(wandb_metric)

        print(
            "current epoch: {} current val loss {:.4f} current accuracy: {:.4f}  best accuracy: {:.4f} at loss {:.4f} at epoch {}".format(
                epoch + 1, val_loss, val_acc, best_metric, best_val_loss, best_metric_epoch))

    print(f"train completed, best_val_loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}")

    return best_model


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = nn.CrossEntropyLoss()

    labels = []
    preds = []
    for i, data in enumerate(val_loader):
            data = data.to(device)
            label = torch.tensor(data.y).type(torch.LongTensor).to(device)
            out = model(data)
            step_loss = loss_func(out, label)
            val_loss += step_loss.detach().item()
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(preds, labels)
    loss = val_loss/(i+1)

    return loss, acc

def test_model(model, test_loader, device):
    model.eval()
    labels = []
    preds = []
    for i, data in enumerate(test_loader):
            data = data.to(device)
            label = torch.tensor(data.y).type(torch.LongTensor).to(device)
            out = model(data)
            preds.append(out.argmax(dim=1).detach().cpu().numpy())
            labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels =  np.concatenate(labels).ravel()
    acc = balanced_accuracy_score(labels, preds )
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    spec = tn / (tn+fp)
    sens = tp / (tp + fn)
    return acc, sens, spec
