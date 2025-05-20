from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from collections import Counter

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


# val setting
def val(loader, model):
        model.eval()
        model = model.cuda()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for _, input, target in loader:
                input, target = input.cuda(), target.cuda()
                output = model(input)
                _, pred = torch.max(output, 1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        return np.array(all_preds), np.array(all_targets)

def cal_total_acc(y_pred, y_true, ul_set=None):
    idxes = np.where(~np.isin(y_true, list(ul_set)))[0]  
    return np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

def cal_cls_acc(y_pred, y_true):
    cf = confusion_matrix(y_true, y_pred).astype(float)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    return cls_hit / (cls_cnt + 1e-8)


def cal_task_acc(y_pred, y_true, learned_cls_num, cls_num_per_task=10, ul_set=None):
    acc = {}
    # Task accuracy 
    for class_id in range(0, np.max(y_true), cls_num_per_task):
        idxes = np.where(
            np.logical_and(
                np.logical_and(y_true >= class_id, y_true < class_id + cls_num_per_task),
                ~np.isin(y_true, list(ul_set)) 
            )
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + cls_num_per_task - 1).rjust(2, "0")
        )
        acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy 
    idxes = np.where(
        np.logical_and(
            y_true < learned_cls_num,
            ~np.isin(y_true, list(ul_set))  
        )
    )[0]
    acc["old"] = (
        0 if len(idxes) == 0
        else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    # New accuracy 
    idxes = np.where(
        np.logical_and(
            y_true >= learned_cls_num,
            ~np.isin(y_true, list(ul_set))  
        )
    )[0]
    acc["new"] = (
        0 if len(idxes) == 0
        else np.around((y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)
    )

    return acc

def dataloader_analysis(dataloader, gen = False):
    label_counts = Counter()
    for batch in dataloader:
        if gen:
            labels = batch["rel_class_id"]
        else:
            labels = batch[-1]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()  
        label_counts.update(labels)
    print('len',len(dataloader),'bs',dataloader.batch_size,'||',dict(sorted(label_counts.items())))


