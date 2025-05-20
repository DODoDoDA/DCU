import numpy as np
from torchvision import datasets, transforms
import os


data_dir = os.path.join("data")

def balance_dataset(data, targets, num_samples_per_class=400):
            balanced_data = []
            balanced_targets = []
            
            for class_label in np.unique(targets):
                class_data = data[targets == class_label]
                class_targets = targets[targets == class_label]
                
                if len(class_data) > num_samples_per_class:
                    indices = np.random.choice(len(class_data), num_samples_per_class, replace=False)
                    class_data = class_data[indices]
                    class_targets = class_targets[indices]
                elif len(class_data) < num_samples_per_class:
                    repeats = (num_samples_per_class // len(class_data)) + 1
                    class_data = np.tile(class_data, (repeats, 1, 1))[:num_samples_per_class]
                    class_targets = np.tile(class_targets, repeats)[:num_samples_per_class]
                
                balanced_data.append(class_data)
                balanced_targets.append(class_targets)
            
            return np.concatenate(balanced_data), np.concatenate(balanced_targets)

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iMNIST(iData):
    use_path = False
    train_trsf = [transforms.Pad(2), transforms.ToTensor(),]
    test_trsf = [transforms.Pad(2), transforms.ToTensor(),]
    common_trsf = [transforms.Normalize(mean=(0.1307,), std=(0.3081,)),]
    class_order = np.arange(10).tolist() 

    def download_data(self):
        train_dataset = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True)
        
        self.train_data, self.train_targets = train_dataset.data.numpy(), np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data.numpy(), np.array(test_dataset.targets)

class iFashionMNIST(iData):
    use_path = False
    train_trsf = [transforms.Pad(2), transforms.ToTensor(),]
    test_trsf = [transforms.Pad(2), transforms.ToTensor(),]
    common_trsf = [transforms.Normalize(mean=(0.5,), std=(0.5,)),]
    class_order = np.arange(10).tolist() 

    def download_data(self):
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data.numpy(), np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data.numpy(), np.array(test_dataset.targets)

class iALLMNIST(iData):
    use_path = False
    train_trsf = [transforms.Resize((32, 32)), transforms.ToTensor(),]
    test_trsf = [transforms.Resize((32, 32)), transforms.ToTensor(),]
    common_trsf = [transforms.Normalize(mean=(0.5,), std=(0.5,)),]
    class_order = np.arange(30).tolist() 

    def download_data(self):
        train_dataset_D = datasets.MNIST(data_dir, train=True, download=True)
        test_dataset_D = datasets.MNIST(data_dir, train=False, download=True)
        
        train_dataset_L = datasets.EMNIST(data_dir, train=True,split='balanced', download=True)
        test_dataset_L = datasets.EMNIST(data_dir, train=False,split='balanced', download=True)
        
        train_dataset_F = datasets.FashionMNIST(data_dir, train=True, download=True)
        test_dataset_F = datasets.FashionMNIST(data_dir, train=False, download=True)
        
        self.train_data_D, self.train_targets_D = train_dataset_D.data.numpy(), np.array(train_dataset_D.targets)
        self.test_data_D, self.test_targets_D = test_dataset_D.data.numpy(), np.array(test_dataset_D.targets)
        
        self.train_data_L, self.train_targets_L = train_dataset_L.data.numpy(), np.array(train_dataset_L.targets)
        self.test_data_L, self.test_targets_L = test_dataset_L.data.numpy(), np.array(test_dataset_L.targets)
        train_mask_L = (self.train_targets_L >= 10) & (self.train_targets_L <= 19)
        test_mask_L = (self.test_targets_L >= 10) & (self.test_targets_L <= 19)
        self.train_data_L = self.train_data_L[train_mask_L]
        self.train_targets_L = self.train_targets_L[train_mask_L]
        self.test_data_L = self.test_data_L[test_mask_L]
        self.test_targets_L = self.test_targets_L[test_mask_L]
        
        self.train_data_F, self.train_targets_F = train_dataset_F.data.numpy(), np.array(train_dataset_F.targets)
        self.test_data_F, self.test_targets_F = test_dataset_F.data.numpy(), np.array(test_dataset_F.targets)
        self.train_targets_F += 20  
        self.test_targets_F += 20

        self.train_data = np.concatenate([self.train_data_D, self.train_data_L, self.train_data_F], axis=0)
        self.train_targets = np.concatenate([self.train_targets_D, self.train_targets_L, self.train_targets_F], axis=0)
        self.test_data = np.concatenate([self.test_data_D, self.test_data_L, self.test_data_F], axis=0)
        self.test_targets = np.concatenate([self.test_targets_D, self.test_targets_L, self.test_targets_F], axis=0)
        
        self.train_data, self.train_targets = balance_dataset(self.train_data, self.train_targets,5000)
        self.test_data, self.test_targets = balance_dataset(self.test_data, self.test_targets,500)

class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor()
    ]
    test_trsf = [transforms.ToTensor()]
    common_trsf = [
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(data_dir, train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(data_dir, train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )
