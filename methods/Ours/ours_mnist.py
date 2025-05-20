import os
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
from methods.base import BaseLearner
from einops import rearrange
from utils import (
    GenDataset_Mnist,
    CilModel, 
    average_weights, 
    setup_seed,
    val,
    cal_cls_acc,
    cal_total_acc,
    cal_task_acc,
    kd_loss,
    AuxDataset,
    loss_PreCE,
    DataIter,
    dataloader_analysis,
    compute_weight_matrix,
    load_topologies,
    weighted_average_weights
)
from mnist_diffusion_model.models import ContextUnet
from mnist_diffusion_model.diffusion_tools import generator_train,get_custom_context,MNIST_sample_ddpm,MNIST_sample_ddim
import time
class ours_mnist(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_init()
        self.generator_init()
        self.syn_imgs_dir = os.path.join(args['save_dir'], "syn_imgs")
        self.pre_ce_loss = loss_PreCE()
        self.topology_sets = load_topologies("./topology/topologies.txt")
 
    def model_init(self):
        tmp_network = CilModel(self.args['net'])
        self._networks_list = {}
        self._old_networks_list = {}
        for idx in range(self.args["num_users"]):
            self._networks_list[idx] = deepcopy(tmp_network)
            self._old_networks_list[idx] = None


########CL part
    def CL_train(self, data_manager, gen = False):
        setup_seed(self.seed)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        print("===========from cls {} to cls {}===========".format(self._known_classes, self._total_classes))
        for idx in range(self.args["num_users"]):
            self._networks_list[idx].update_fc(self._total_classes)
        self.init_data(data_manager)
        if gen:
            self.diffusion_train()
        print('known:',self._known_classes_set)
        self.CL_modify_data(data_manager.get_train_trsf())
        self._fl_train()

    def _fl_train(self):
        prog_bar = tqdm(range(self.args["com_round"]))
        for cl_ep in prog_bar:
            ep_topology = self.topology_sets[cl_ep]
            topo_weight = compute_weight_matrix(ep_topology)
            local_weights = []
            for idx in range(self.args["num_users"]):
                if self._cur_task == 0:
                    w = self._local_update(self._networks_list[idx], self.local_train_loader[idx])
                else:
                    w = self._local_finetune(self._networks_list[idx], self.local_train_loader[idx], self.pre_iters[idx], self._old_networks_list[idx])
                local_weights.append(deepcopy(w))
            cls_acc_list = []
            for cid in range(self.args["num_users"]):
                cid_topo_weight = topo_weight[cid]
                tmp_weights = weighted_average_weights(local_weights, cid_topo_weight)
                self._networks_list[cid].load_state_dict(tmp_weights)
                tmp_preds, tmp_targets = val(self.test_loader, self._networks_list[cid])
                cls_acc = cal_cls_acc(tmp_preds, tmp_targets)
                cls_acc_list.append(cls_acc)
            cls_acc_list = np.mean(cls_acc_list, 0)
            formatted_cls_acc = [f"{100 * acc:.2f}" for acc in cls_acc_list]
            info=("EP {}/{}| {}".format(
                cl_ep+1, self.args["com_round"], formatted_cls_acc,))
            prog_bar.set_description(info)

    def _local_update(self, model, train_data_loader):
        model.cuda()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                images, labels = images.cuda(), labels.cuda()
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()
    
    def _local_finetune(self, model, train_data_loader, pre_iter=None, teacher=None):
        teacher.eval()
        model.cuda()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for _ in range(self.args["local_ep"]):
            for batch_idx, (_, images, labels) in enumerate(train_data_loader):
                _, pre_imgs, pre_labels = pre_iter.next()
                images, labels, pre_imgs, pre_labels = images.cuda(), labels.cuda(), pre_imgs.cuda(), pre_labels.cuda()
                fake_targets = labels - self._known_classes
                logits = model(images)
                loss = F.cross_entropy(logits[:, self._known_classes:], fake_targets)
                pre_logits = model(pre_imgs)
                with torch.no_grad():
                    pre_logits_teacher = teacher(pre_imgs.detach())
                teacher_model_prob = F.softmax(pre_logits_teacher, 1) # [batch_size, num_classes]
                bool_label_mask = F.one_hot(pre_labels, num_classes=pre_logits_teacher.shape[-1]) > 0 # [batch_size, num_classes] bool
                adaptive_weight = teacher_model_prob[bool_label_mask] # [batch_size]
                loss_kd = kd_loss(
                    pre_logits[:, : self._known_classes],   
                    pre_logits_teacher.detach(),
                    2,
                )
                loss = loss + loss_kd 
                loss_ce_pre = self.pre_ce_loss(pre_logits[:, :self._known_classes], pre_labels, adaptive_weight)
                loss = loss + loss_ce_pre
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.state_dict()
    
    def CL_modify_data(self, train_trsf):
        trsf = train_trsf
        # cur
        self.local_train_loader = []
        for idx in range(self.args["num_users"]):
            path = os.path.join(self.syn_imgs_dir, 'topo_'+str(idx))
            cur_aux_dataset = ConcatDataset([AuxDataset(path, i, self.args['cur_size'], trsf) 
                for i in self.cur_task_classes])
            cur_loader = DataLoader(ConcatDataset([self.local_train_dataset[idx], cur_aux_dataset]), 
                           batch_size=self.args["local_bs"], shuffle=True)
            self.local_train_loader.append(cur_loader)
            #print(f"Client {idx} Cur Loader Info")
            #dataloader_analysis(cur_loader)
    
        # pre
        self.pre_iters = []
        if self._cur_task > 0:
            for idx in range(self.args["num_users"]):
                path = os.path.join(self.syn_imgs_dir, 'topo_'+str(idx))
                pre_dataset = ConcatDataset([AuxDataset(path, i, self.args['pre_size'], trsf) 
                    for i in self._known_classes_set])
                pre_loader = DataLoader(pre_dataset, batch_size=self.args["local_bs"], shuffle=True)
                self.pre_iters.append(DataIter(pre_loader))
                #print(f"Client {idx} Pre Loader Info")
                #dataloader_analysis(pre_loader)
        else:
            for idx in range(self.args["num_users"]):
                self.pre_iters.append(None)
            #print(f"Pre Loader Info: None")

    def CL_after_task(self):
        self._known_classes = self._total_classes
        self._known_classes_set.update(self.cur_task_classes.tolist())  
        for cid in range(self.args["num_users"]):
            self._old_networks_list[cid] = self._networks_list[cid].copy().freeze()
        #model_save_path = os.path.join(self.args['save_dir'], 'UL_model')
        #os.makedirs(model_save_path, exist_ok=True)
        #torch.save(self._networks_list[0], model_save_path + '/After_CL' + str(self._cur_task) + '_Cid' + str(0) + '.pth')

#######generator part
    def generator_init(self):
        self._generator = ContextUnet(in_channels=1, height=32, width=32, n_feat=64, n_cfeat=10, n_downs=4)
        checkpoint = torch.load('./mnist_diffusion_model/checkpoints/allmnist_checkpoint_99.pth', map_location='cpu')
        base_part = {k: v for k, v in checkpoint["model_state_dict"].items() if 'contextembs' not in k}
        self._generator.load_state_dict(base_part,strict=False)
        for name, param in self._generator.named_parameters():
            if 'contextembs' in name:
                param.requires_grad = True  
            else:
                param.requires_grad = False  

        self.generator_init_embedding =  deepcopy({name: param for name, param in self._generator.state_dict().items() if 'contextembs' in name})
        self._generator_emebedding_list = {}
        print('generator init embedding loaded')

    def debug_only_diffusion_(self, data_manager):
        setup_seed(self.seed)
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        print("===========from cls {} to cls {}===========".format(self._known_classes, self._total_classes))
        for idx in range(self.args["num_users"]):
            self._networks_list[idx].update_fc(self._total_classes)
        self.init_data(data_manager)
        self.diffusion_train()

    def diffusion_train(self):
        self.G_modify_data()
        self._class_inversion()
        self._synthesis_imgs(self._generator_emebedding_list)
        
    def G_modify_data(self):
        print("---------------gen stage data info--------------")
        self.min_class_id, self.max_class_id = np.min(self.cur_task_classes), np.max(self.cur_task_classes)
        self.gen_data_iters = []
        for idx in range(self.args["num_users"]):
            local_gen_dataset = deepcopy(GenDataset_Mnist(
                    np_imgs=self.local_train_dataset[idx].images,
                    np_labels=self.local_train_dataset[idx].labels,
                    num_classes=10, 
                    min_class_id=self.min_class_id
                ))
            local_gen_loader = DataLoader(local_gen_dataset, batch_size=self.args['g_local_bs'], 
                        num_workers=4, shuffle=True)
            self.gen_data_iters.append(DataIter(local_gen_loader))
            print(f"Client {idx} Diffusion Loader Info")
            dataloader_analysis(local_gen_loader)
    def _class_inversion(self):
        for idx in range(self.args["num_users"]):
            self._generator_emebedding_list[idx] = deepcopy(self.generator_init_embedding)
        self._generator.cuda()
        prog_bar = tqdm(range(self.args["com_round_gen"]), desc='G train')
        for cl_ep in prog_bar:
            ep_topology = self.topology_sets[cl_ep]
            topo_weight = compute_weight_matrix(ep_topology)
            local_weights = []
            for idx in range(self.args["num_users"]):
                self._generator.load_state_dict(self._generator_emebedding_list[idx],strict=False)
                w = generator_train(deepcopy(self._generator),self.gen_data_iters[idx],
                                    g_local_steps=self.args["g_local_steps"], lr=1e-3,timesteps=500, beta1=0.0001, beta2=0.02)
                local_weights.append(deepcopy(w))
            for cid in range(self.args["num_users"]):
                cid_topo_weight = topo_weight[cid]
                tmp_weights = weighted_average_weights(local_weights, cid_topo_weight)
                self._generator_emebedding_list[cid] = deepcopy(tmp_weights)
        
        if self.args["save_cls_embeds"]:
            for cid in range(self.args["num_users"]):
                cls_embeds_path = os.path.join(self.save_dir, 
                    'cls_embeds', 'client%d_%d-%d_embedding_manager.pt' % (cid, self._known_classes, self._total_classes))
                os.makedirs(os.path.dirname(cls_embeds_path), exist_ok=True)
                torch.save(self._generator_emebedding_list[cid], cls_embeds_path)

    def _synthesis_imgs(self, inv_text_embeds):
        timesteps = 500
        beta1=0.0001
        beta2=0.02
        n_samples = 500
        os.makedirs(self.syn_imgs_dir, exist_ok=True)
        for cid in range(self.args["num_users"]):
            self._generator.load_state_dict(inv_text_embeds[cid], strict=False)
            self._generator.cuda()
            with torch.no_grad():
                for tmp_cls in self.cur_task_classes:
                    outdir = os.path.join(self.syn_imgs_dir, 'topo_'+str(cid), str(tmp_cls))
                    os.makedirs(outdir, exist_ok=True)
                    tmp_cls_num = 0
                    for _ in range(self.args['n_iter']):
                        c = get_custom_context(n_samples, 10, target_class=tmp_cls - self.min_class_id)
                        x_samples_ddpm = MNIST_sample_ddpm(self._generator, n_samples, c, timesteps,beta1,beta2,)
                        for x_sample in x_samples_ddpm:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            x_sample = x_sample.squeeze(-1)
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(outdir,  f"{tmp_cls}-{tmp_cls_num}.jpg"))
                            tmp_cls_num += 1
            
            

#########ul part  
    def UL_train(self, data_manager, cur_ul_class):
        setup_seed(self.seed)
        print("===========ul cls ",cur_ul_class,"===========")
        self.cur_task_classes = np.array([])
        self.cur_ul_class = cur_ul_class
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
        self.UL_modify_data(data_manager.get_train_trsf())
        self._ul_fl_train()
        self._ul_classes_set.add(self.cur_ul_class) 
        


    def UL_modify_data(self, train_trsf):
        print("---------------UL stage data info--------------")
        trsf = train_trsf
        self.ul_iters = []
        for idx in range(self.args["num_users"]):
            path = os.path.join(self.syn_imgs_dir, 'topo_'+str(idx))
            #ul_dataset = ConcatDataset([AuxDataset(path, i, self.args['ul_size'], trsf) 
            #    for i in self._known_classes_set])
            ul_dataset = AuxDataset(path, self.cur_ul_class, self.args['ul_size'], trsf) 
            ul_loader = DataLoader(ul_dataset, batch_size=self.args["ul_local_bs"], shuffle=True)
            self.ul_iters.append(DataIter(ul_loader))

    def _ul_fl_train(self):
        start_time = time.time()
        for cid in range(self.args["num_users"]):
            ul_w = self._ul_local_update(deepcopy(self._networks_list[cid]), self.ul_iters[cid])
            ul_state_dict = deepcopy(self._networks_list[cid].classifier.state_dict())
            ul_sga_dict = deepcopy(self._networks_list[cid].classifier.state_dict())
            ul_state_dict['weight'] += 0.8*(ul_sga_dict['weight'] - ul_w['weight'])
            ul_state_dict['bias'] += 0.8*(ul_sga_dict['bias'] - ul_w['bias'])
            #ul_state_dict['weight'] -= 0.2 * ul_w['weight']
            #ul_state_dict['bias'] -= 0.2 * ul_w['bias']
            self._networks_list[cid].classifier.load_state_dict(ul_state_dict)
        end_time = time.time()
        elapsed_time = end_time - start_time  
        print('****time cost: ',elapsed_time,'****')

    def _ul_local_update(self, model, ul_iter):
        model.cuda()
        ul_classifier = deepcopy(model.classifier)
        for name,param in model.backbone.named_parameters():
            param.requires_grad=False
        optimizer = torch.optim.SGD(ul_classifier.parameters(), lr=0.01, momentum=0.9)
        for bs in range(self.args["ul_local_steps"]):
            model.backbone.eval()
            ul_classifier.train()
            _, images, labels = ul_iter.next()
            labels = labels.fill_(self.cur_ul_class)
            images, labels = images.cuda(), labels.cuda()
            features = model.backbone(images)
            logits = ul_classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return ul_classifier.state_dict()
    
    def UL_after_task(self):
        self._known_classes_set.remove(self.cur_ul_class)
        for cid in range(self.args["num_users"]):
            self._old_networks_list[cid] = self._networks_list[cid].copy().freeze()

        #model_save_path = os.path.join(self.args['save_dir'], 'UL_model')
        #torch.save(self._networks_list[0], model_save_path + '/After_UL' + str(self._cur_task) + '_Cid' + str(0) + '.pth')

##########topo eval/log
    def eval_task(self):
        tot_acc_list=[]
        cls_acc_list=[]
        task_acc_list=[]
        for cid in range(self.args["num_users"]):
            tmp_preds, tmp_targets = val(self.test_loader, self._networks_list[cid])
            assert len(tmp_preds) == len(tmp_targets), "Data length error."
            tot_acc_list.append(cal_total_acc(tmp_preds, tmp_targets, self._ul_classes_set))
            cls_acc = cal_cls_acc(tmp_preds, tmp_targets)
            cls_acc_list.append(cls_acc)
            task_acc_list.append(cal_task_acc(tmp_preds, tmp_targets, self._known_classes, self.each_task, self._ul_classes_set))
        self.tot_acc_dict["global"].append(np.around(np.mean(tot_acc_list),decimals=2))
        
        cls_acc_list = np.mean(cls_acc_list, 0)
        formatted_cls_acc = [f"{100*acc:.2f}" for acc in cls_acc_list]
        self.per_cls_acc_dict["global"].append(formatted_cls_acc)
        
        mean_dict={}
        for key in task_acc_list[0].keys():
            sum_values = np.array([d[key] for d in task_acc_list])
            mean_value = np.mean(sum_values)
            mean_dict[key] = np.around(mean_value, decimals=2)
        self.per_task_acc_dict["global"].append(mean_dict)
        