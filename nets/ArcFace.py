import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# --------------------------------------------- ArcFace --------------------------------------------

class Classifier(nn.Module):
    """Implement of large margin arc distance (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: additive angular margin

            cos(theta + m)
    """
    def __init__(self, conf, n_classes):
        super(Classifier, self).__init__()
        self.in_features = conf.emd_size
        self.out_features = n_classes
        self.s = conf.loss_s
        self.m = conf.loss_m
        self.weight = Parameter(torch.FloatTensor(n_classes, conf.emd_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = conf.easy_margin
        self.cos_m = math.cos(conf.loss_m)
        self.sin_m = math.sin(conf.loss_m)
        self.th = math.cos(math.pi - conf.loss_m)
        self.mm = math.sin(math.pi - conf.loss_m) * conf.loss_m

        self.device = conf.device

    def forward(self, input, label):
        # cos(theta) & phi(theta)
        cosine = (F.linear(F.normalize(input), F.normalize(self.weight))).clamp(-1 + 1e-9, 1 - 1e-9)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        # if use easy_margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # torch.where(out_i = {x_i if condition_i else y_i)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output