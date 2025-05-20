import copy
import logging
import torch
from torch import nn
from torch.nn import functional as F
from .resnet import cifar_resnet18
from .CNN import conv2mnist



def get_backbone(backbone_name):
    if backbone_name == "resnet18":
        backbone = cifar_resnet18()
    elif backbone_name == "conv2mnist":
        backbone = conv2mnist()
    else:
        raise NotImplementedError(f'Unknown backbone -> {backbone_name}')

    return backbone

class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)


class CilClassifier(nn.Module):
    def __init__(self, embed_dim, nb_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = nn.ModuleList([nn.Linear(embed_dim, nb_classes).cuda()])

    def __getitem__(self, index):
        return self.heads[index]

    def __len__(self):
        return len(self.heads)

    def forward(self, x):
        logits = torch.cat([head(x) for head in self.heads], dim=1)
        return logits

    def adaption(self, nb_classes):
        self.heads.append(nn.Linear(self.embed_dim, nb_classes).cuda())


class CilModel(nn.Module):
    def __init__(self, backbone_name):
        super(CilModel, self).__init__()
        self.backbone = get_backbone(backbone_name)
        self.classifier = None
    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def forward(self, x):
        x = self.backbone(x)
        out = self.classifier(x)
        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def update_fc(self, nb_classes):
        classifier = self.generate_fc(self.feature_dim, nb_classes)
        if self.classifier is not None:
            nb_output = self.classifier.out_features
            weight = copy.deepcopy(self.classifier.weight.data)
            bias = copy.deepcopy(self.classifier.bias.data)
            classifier.weight.data[:nb_output] = weight
            classifier.bias.data[:nb_output] = bias

        del self.classifier
        self.classifier = classifier.cuda()
    def generate_fc(self, in_dim, out_dim):
        classifier = SimpleLinear(in_dim, out_dim)

        return classifier

    #def prev_model_adaption(self, nb_classes):
    #    if self.fc is None:
    #        self.fc = CilClassifier(self.feature_dim, nb_classes).cuda()
    #    else:
    #        self.fc.adaption(nb_classes)

    #def after_model_adaption(self, nb_classes, _cur_task):
    #    if _cur_task > 0:
    #        self.weight_align(nb_classes)

    #@torch.no_grad()
    #def weight_align(self, nb_new_classes):
    #    w = torch.cat([head.weight.data for head in self.fc], dim=0)
    #    norms = torch.norm(w, dim=1)
    #
    #    norm_old = norms[:-nb_new_classes]
    #    norm_new = norms[-nb_new_classes:]

    #    gamma = torch.mean(norm_old) / torch.mean(norm_new)
    #    print(f"old norm / new norm ={gamma}")
    #    self.fc[-1].weight.data = gamma * w[-nb_new_classes:]