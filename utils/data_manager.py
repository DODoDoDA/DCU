import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import iCIFAR10, iCIFAR100, iMNIST, iFashionMNIST, iALLMNIST
import torch, copy
import random
import torch.backends.cudnn as cudnn
from glob import glob
import PIL
from utils.topological_methods import compute_weight_matrix
from collections import defaultdict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

def weighted_average_weights(w, weights):
    weights.cuda()
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w[0][key] * weights[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * weights[i]
    return w_avg

def average_weights(w, dp_si=0):
    """
    Returns the average of the weights.
    """
    if dp_si != 0:
        si = dp_si
        C = 1
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(0, len(w)):
                if i != 0:
                    w_avg[key] += w[i][key]
                if not 'num_batches_tracked' in key:
                    w_avg[key] += torch.normal(0, si * C, w[i][key].shape).cuda()
            if 'num_batches_tracked' in key:
                w_avg[key] = w_avg[key].true_divide(len(w))
            else:
                w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg
        
    else:
        w_avg = copy.deepcopy(w[0])
        for key in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[key] += w[i][key]
            if 'num_batches_tracked' in key:
                w_avg[key] = w_avg[key].true_divide(len(w))
            else:
                w_avg[key] = torch.div(w_avg[key], len(w))
        return w_avg


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.images = dataset.images[self.idxs]
        self.labels = dataset.labels[self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx, image, label = self.dataset[self.idxs[item]]
        return idx, image, label


def partition_data(y_train, beta=0.4, n_parties=5, return_ratio=False):
    data_size = y_train.shape[0]
    labels = np.unique(y_train)
    label_ratio = np.full(shape=(n_parties, len(labels)), fill_value=1 / n_parties)
    if beta == 0:   # for iid
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}

    elif beta > 0:  # for niid
        min_size = 0
        min_require_size = 1
        # label = np.unique(y_train).shape[0]
        net_dataidx_map = {}
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k_i, k in enumerate(labels):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array(   # 0 or x
                    [p * (len(idx_j) < data_size / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                label_ratio[:, k_i] = proportions
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    if return_ratio:
        return net_dataidx_map, label_ratio
    else:
        return net_dataidx_map


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments) #last tasks
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)
    
    def get_train_trsf(self):
        return transforms.Compose([*self._train_trsf, *self._common_trsf])
    
    def get_test_trsf(self):
        return transforms.Compose([*self._test_trsf, *self._common_trsf])

    def get_task_size(self, task):
        return self._increments[task]

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        return DummyDataset(data, targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]
    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "mnist":
        return iMNIST()
    elif name == "fmnist":
        return iFashionMNIST()
    elif name=='allmnist':
        return iALLMNIST()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
    


##########################
class AuxDataset_TP(Dataset):
    def __init__(self, root, cid, cls_id, size_per_cls, transform=None):
        self.root = os.path.join(root, cid, str(cls_id))
        img_paths = glob(os.path.join(self.root, "*.jpg"))
        img_paths = [tp for tp in img_paths 
                     if int(tp.split("-")[-1].rstrip(".jpg")) <= size_per_cls]
        self.images, self.labels = [], []
        for tp in img_paths:
            self.labels.append(int(tp.split("/")[-2]))
            with Image.open(tp) as tmp_img:
                resized_img = tmp_img.resize((32, 32), Image.BICUBIC)
                self.images.append(np.array(resized_img))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        img = self.transform(img)
        return idx, img, label

    def __len__(self):
        return len(self.images)
    
class AuxDataset(Dataset):
    def __init__(self, root, cls_id, size_per_cls, transform=None):
        self.root = os.path.join(root, str(cls_id))
        img_paths = glob(os.path.join(self.root, "*.jpg"))
        img_paths = [tp for tp in img_paths 
                     if int(tp.split("-")[-1].rstrip(".jpg")) <= size_per_cls]
        self.images, self.labels = [], []
        for tp in img_paths:
            self.labels.append(int(tp.split("/")[-2]))
            with Image.open(tp) as tmp_img:
                resized_img = tmp_img.resize((32, 32), Image.BICUBIC)
                self.images.append(np.array(resized_img))
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        label = self.labels[idx]
        img = self.transform(img)
        return idx, img, label

    def __len__(self):
        return len(self.images)



class ReservoirBuffer:
    def __init__(self, buffer_size=500):
        self.buffer_size = buffer_size
        self.buffer_img = []  
        self.buffer_label = []
        self.label_to_indices = defaultdict(list)  
        self.total_seen = 0  # for Reservoir Sampling

    def add(self, img, label):
        self.total_seen += 1
        if len(self.buffer_label) < self.buffer_size: # Buffer not full - simply append
            self.buffer_img.append(img)
            self.buffer_label.append(label)
            self.label_to_indices[label].append(len(self.buffer_label) - 1)
        else:
            # Buffer full - decide whether to replace (k=buffer_size, N=total_seen)
            replace_prob = self.buffer_size / self.total_seen
            if random.random() < replace_prob:
                idx_to_replace = random.randint(0, self.buffer_size - 1)
                old_label = self.buffer_label[idx_to_replace]
                self.label_to_indices[old_label].remove(idx_to_replace)
                # Perform replacement
                self.buffer_img[idx_to_replace] = img
                self.buffer_label[idx_to_replace] = label
                self.label_to_indices[label].append(idx_to_replace)

    def delete_by_label(self, label):
        if label not in self.label_to_indices:
            return
        # Delete from highest index first to prevent index shifting issues
        indices_to_delete = sorted(self.label_to_indices[label], reverse=True)
        for idx in indices_to_delete:
            if idx < len(self.buffer_label):
                del self.buffer_img[idx]
                del self.buffer_label[idx]
        # Rebuild label_to_indices mapping
        self.label_to_indices = defaultdict(list)
        for idx, current_label in enumerate(self.buffer_label):
            self.label_to_indices[current_label].append(idx)
        self.total_seen = len(self.buffer_label)
    def get_buffer_contents(self):
        return {
            "label_counts": {label: len(indices) for label, indices in self.label_to_indices.items()},
            "total_seen": self.total_seen
        }
    
    def get_dataset(self, trsf):
        return DummyDataset(self.buffer_img, self.buffer_label, trsf, False)
