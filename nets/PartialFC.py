"""
Obtained from
https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/partial_fc.py
https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
"""
import logging
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import Module
from torch.nn.functional import normalize, linear
from torch.nn.parameter import Parameter


class PartialFC(Module):
    """
    Author: {Xiang An, Yang Xiao, XuHan Zhu} in DeepGlint,
    Partial FC: Training 10 Million Identities on a Single Machine
    See the original paper:
    https://arxiv.org/abs/2010.05222
    """

    @torch.no_grad()
    def __init__(self, conf, n_classes, idx):
        
        super(PartialFC, self).__init__()
        
        self.num_classes: int = n_classes
        self.rank: int = conf.rank
        self.local_rank: int = conf.local_rank
        self.device: torch.device = torch.device("cuda:{}".format(self.local_rank))
        self.world_size: int = conf.world_size
        self.batch_size: int = conf.b
        self.margin_softmax: callable = get_loss(str(conf.metric).lower())
        self.sample_rate: float = conf.sample_rate
        self.embedding_size: int = conf.emd_size
        self.prefix = conf.pretrained_dir
        
        self.num_local: int = self.num_classes // self.world_size + int(self.rank < self.num_classes % self.world_size)
        self.class_start: int = self.num_classes // self.world_size * self.rank + min(self.rank, self.num_classes % self.world_size)
        self.num_sample: int = int(self.sample_rate * self.num_local)

        self.weight_name = os.path.join(self.prefix, f"rank_{self.rank}_softmax_weight.pt")
        self.weight_mom_name = os.path.join(self.prefix, f"rank_{self.rank}_softmax_weight_mom.pt")
        
        if conf.resume:
            wm_weight = torch.load(conf.WM_path)
            self.weight = wm_weight[f'metric_{idx}_weight']
            self.weight_mom = wm_weight[f'metric_{idx}_mom']
            
        else:
            self.weight = torch.normal(0, 0.01, (self.num_local, self.embedding_size), device=self.device)
            self.weight_mom: torch.Tensor = torch.zeros_like(self.weight)
            logging.info("softmax weight init successfully!")
            logging.info("softmax weight mom init successfully!")
            
        self.stream: torch.cuda.Stream = torch.cuda.Stream(self.local_rank)

        self.index = None
        if int(self.sample_rate) == 1:
            self.update = lambda: 0
            self.sub_weight = Parameter(self.weight)
            self.sub_weight_mom = self.weight_mom
        else:
            self.sub_weight = Parameter(torch.empty((0, 0)).cuda(self.local_rank))

    def save_params(self):
        """ Save softmax weight for each rank on prefix
        """
        torch.save(self.weight.data, self.weight_name)
        torch.save(self.weight_mom, self.weight_mom_name)

    @torch.no_grad()
    def sample(self, total_label):
        """
        Sample all positive class centers in each rank, and random select neg class centers to filling a fixed
        `num_sample`.
        total_label: tensor
            Label after all gather, which cross all GPUs.
        """
        index_positive = (self.class_start <= total_label) & (total_label < self.class_start + self.num_local)
        total_label[~index_positive] = -1
        total_label[index_positive] -= self.class_start
        if int(self.sample_rate) != 1:
            positive = torch.unique(total_label[index_positive], sorted=True)
            if self.num_sample - positive.size(0) >= 0:
                perm = torch.rand(size=[self.num_local], device=self.device)
                perm[positive] = 2.0
                index = torch.topk(perm, k=self.num_sample)[1]
                index = index.sort()[0]
            else:
                index = positive
            self.index = index
            total_label[index_positive] = torch.searchsorted(index, total_label[index_positive])
            self.sub_weight = Parameter(self.weight[index])
            self.sub_weight_mom = self.weight_mom[index]

    def forward(self, total_features, norm_weight):
        """ Partial fc forward, `logits = X * sample(W)`
        """
        torch.cuda.current_stream().wait_stream(self.stream)
        logits = linear(total_features, norm_weight)
        return logits

    @torch.no_grad()
    def update(self):
        """ Set updated weight and weight_mom to memory bank.
        """
        self.weight_mom[self.index] = self.sub_weight_mom
        self.weight[self.index] = self.sub_weight

    def prepare(self, label, optimizer):
        """
        get sampled class centers for cal softmax.
        label: tensor
            Label tensor on each rank.
        optimizer: opt
            Optimizer for partial fc, which need to get weight mom.
        """
        with torch.cuda.stream(self.stream):
            total_label = torch.zeros(
                size=[self.batch_size * self.world_size], device=self.device, dtype=torch.long)
            dist.all_gather(list(total_label.chunk(self.world_size, dim=0)), label)
            
            self.sample(total_label)
            
            optimizer.state.pop(optimizer.param_groups[-1]['params'][0], None)
            optimizer.param_groups[-1]['params'][0] = self.sub_weight
            optimizer.state[self.sub_weight]['momentum_buffer'] = self.sub_weight_mom
            norm_weight = normalize(self.sub_weight)
            
            return total_label, norm_weight

    def forward_backward(self, label, features, optimizer):
        """
        Partial fc forward and backward with model parallel
        label: tensor
            Label tensor on each rank(GPU)
        features: tensor
            Features tensor on each rank(GPU)
        optimizer: optimizer
            Optimizer for partial fc
        Returns:
        --------
        x_grad: tensor
            The gradient of features.
        loss_v: tensor
            Loss value for cross entropy.
        """
        total_label, norm_weight = self.prepare(label, optimizer)
        total_features = torch.zeros(
            size=[self.batch_size * self.world_size, self.embedding_size], device=self.device)
        dist.all_gather(list(total_features.chunk(self.world_size, dim=0)), features.data)
        total_features.requires_grad = True
        
        #print('Rank: {}, W size: {}'.format(self.local_rank, norm_weight.size()))
        
        logits = self.forward(total_features, norm_weight)
        logits = self.margin_softmax(logits, total_label)

        with torch.no_grad():
            max_fc = torch.max(logits, dim=1, keepdim=True)[0]
            dist.all_reduce(max_fc, dist.ReduceOp.MAX)

            # calculate exp(logits) and all-reduce
            logits_exp = torch.exp(logits - max_fc)
            logits_sum_exp = logits_exp.sum(dim=1, keepdims=True)
            dist.all_reduce(logits_sum_exp, dist.ReduceOp.SUM)

            # calculate prob
            logits_exp.div_(logits_sum_exp)

            # get one-hot
            grad = logits_exp
            index = torch.where(total_label != -1)[0]
            one_hot = torch.zeros(size=[index.size()[0], grad.size()[1]], device=grad.device)
            one_hot.scatter_(1, total_label[index, None], 1)

            # calculate loss
            loss = torch.zeros(grad.size()[0], 1, device=grad.device)
            loss[index] = grad[index].gather(1, total_label[index, None])
            dist.all_reduce(loss, dist.ReduceOp.SUM)
            loss_v = loss.clamp_min_(1e-30).log_().mean() * (-1)

            # calculate grad
            grad[index] -= one_hot
            grad.div_(self.batch_size * self.world_size)

        logits.backward(grad)
        if total_features.grad is not None:
            total_features.grad.detach_()
        x_grad: torch.Tensor = torch.zeros_like(features, requires_grad=True)
        # feature gradient all-reduce
        dist.reduce_scatter(x_grad, list(total_features.grad.chunk(self.world_size, dim=0)))
        x_grad = x_grad * self.world_size
        # backward backbone
        return x_grad, loss_v, logits_exp


def get_loss(name):
    if name == "cosface":
        return CosFace()
    elif name == "arcface":
        return ArcFace()
    else:
        raise ValueError()


class CosFace(nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine[index] -= m_hot
        ret = cosine * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine: torch.Tensor, label):
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return cosine
