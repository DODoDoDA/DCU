import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import (
    val,
    cal_total_acc,
    cal_cls_acc,
    cal_task_acc,
    partition_data, 
    DatasetSplit,
    dataloader_analysis
)


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._known_classes_set = set()
        self._ul_classes_set = set()
        self._total_classes = 0
        self._network = None
        self._old_network = None

        self.args = args
        self.each_task = args["increment"]
        self.seed = args["seed"]
        self.tasks = args["tasks"]
        self.save_dir = args["save_dir"]
        self.dataset_name = args["dataset"]

        self.label_ratio = None
        # metrics
        self.tot_acc_dict = {i: [] for i in range(self.args["num_users"])}
        self.tot_acc_dict["global"] = []

        self.per_cls_acc_dict = {i: [] for i in range(self.args["num_users"])}
        self.per_cls_acc_dict["global"] = []
        
        self.per_task_acc_dict = {i: [] for i in range(self.args["num_users"])}
        self.per_task_acc_dict["global"] = []

        #self.forgetting_measure_dict = {i: [] for i in range(self.args["num_users"])}
        #self.forgetting_measure_dict["global"] = []
        
        self.local_acc_mean_std = []
        #self.local_fm_mean_std = []

    def after_task(self):
        pass

    def CL_train(self):
        pass

    def _fl_train(self):
        pass

    def eval_task(self):
        global_preds, global_targets = val(self.test_loader, self._network)
        assert len(global_preds) == len(global_targets), "Data length error."
        self.tot_acc_dict["global"].append(cal_total_acc(global_preds, global_targets,self._ul_classes_set))# total acc
        cls_acc = cal_cls_acc(global_preds, global_targets)
        formatted_cls_acc = [f"{acc * 100:.2f}" for acc in cls_acc]
        self.per_cls_acc_dict["global"].append(formatted_cls_acc)#cls acc
        self.per_task_acc_dict["global"].append(cal_task_acc(global_preds, global_targets, self._known_classes, self.each_task, self._ul_classes_set))# task acc

    def log_metrics(self, print_to_console=True, save_to_file=False):
        log_str  = "\nEVAL:\n"
        log_str += "global_test:\n"
        log_str += "\tTot ACC curve: {}\n".format([tot_acc for tot_acc in self.tot_acc_dict["global"]])
        log_str += "\tPer Task ACC: {}\n".format(self.per_task_acc_dict["global"][-1])
        log_str += "\tPer Cls ACC: {}\n".format(self.per_cls_acc_dict["global"][-1])
        if save_to_file:
            log_txt_pth = os.path.join(self.save_dir, "log.txt")
            os.makedirs(os.path.dirname(log_txt_pth), exist_ok=True)
            with open(log_txt_pth, "a") as file:
               file.write(log_str)
        if print_to_console:
            print(log_str)
    

    def init_data(self, data_manager):
        # train
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        user_groups, label_ratio = partition_data(self.train_dataset.labels, beta=self.args["beta"], n_parties=self.args["num_users"], return_ratio=True)
        self.label_ratio = label_ratio if self.label_ratio is None \
                else np.concatenate((self.label_ratio, label_ratio), axis=1)            
        self.local_train_dataset = []
        for idx in range(self.args["num_users"]):
            self.local_train_dataset.append(DatasetSplit(self.train_dataset, user_groups[idx]))
        self.cur_task_classes = np.unique(self.train_dataset.labels)
        # test    
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    def modify_data(self):
        pass
    def data_info(self):
        print("----data info----")
        print("Train data info:")
        for idx in range(self.args["num_users"]):
            dataloader_analysis(self.local_train_loaders[idx])
        print("Test data info:")
        dataloader_analysis(self.test_loader)
        print("-----------------")

