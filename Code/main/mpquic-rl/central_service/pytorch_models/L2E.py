import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from torchvision import models
import torch.nn.functional as F
#from models import mmd

Input = 6

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

class L2EModel(nn.Module):
    def __init__(self, num_classes, disc = 'MMD', option='resnet18', use_bottleneck=False, bottleneck_width=256, width=64):
        super(L2EModel, self).__init__()
        self.dim = 6
        hidden_dim = bottleneck_width if use_bottleneck else self.dim
        self.base_network = nn.Sequential(
            #model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
            nn.Linear(Input, hidden_dim),
            nn.ReLU(),
            #nn.Linear(Input, hidden_dim),
        )
        self.use_bottleneck = use_bottleneck
        self.num_classes = num_classes
        self.disc = disc

        self.bottleneck_layer = nn.Sequential(
            nn.Linear(self.dim, bottleneck_width),
            nn.BatchNorm1d(bottleneck_width),
            nn.ReLU(),
            nn.Dropout(0.5)
        )


        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, width),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(width, num_classes),
            nn.Softmax()
        )

        if disc == "C-divergence":
            self.discriminator = nn.Sequential(
                nn.Linear(hidden_dim+num_classes, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, 2),
            )
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(hidden_dim, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, 2),
            )

        self.criterion = None #should be RL

    def forward(self, s_inputs, s_outputs, t_inputs, alpha=1.0):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
        s_preds = self.classifier_layer(s_feats)
        loss = self.criterion(s_preds, s_outputs)

        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            t_feats = self.bottleneck_layer(t_feats)
        if self.disc == "MMD":
            mmd_loss = mmd_rbf_noaccelerate(s_feats, t_feats)
            loss += mmd_loss * 0.1
        elif self.disc == "JS-divergence":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats], dim=0), alpha))
            domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
            loss += domain_loss * 0.1
        elif self.disc == "C-divergence":
            s_l_feats = torch.cat([s_feats, 20 * self.one_hot_embedding(s_outputs)], dim=1)  # 10 for w -> d
            t_l_feats = torch.cat([t_feats, 20 * F.softmax(self.classifier_layer(t_feats), dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_feats, t_l_feats], dim=0), alpha))
            domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
            loss += domain_loss * 0.1
        return loss

    def meta_loss(self, s_inputs, t_inputs, t_outputs):
        s_feats = self.base_network(s_inputs).view(s_inputs.shape[0], -1)
        t_feats = self.base_network(t_inputs).view(t_inputs.shape[0], -1)
        if self.use_bottleneck:
            s_feats = self.bottleneck_layer(s_feats)
            t_feats = self.bottleneck_layer(t_feats)
        t_preds = self.classifier_layer(t_feats)
        meta_loss = self.criterion(t_preds, t_outputs)

        if self.disc == "MMD":
            n = min(s_feats.shape[0], t_feats.shape[0])
            mmd_loss = mmd_rbf_noaccelerate(s_feats[:n], t_feats[:n])
            meta_loss += mmd_loss
        elif self.disc == "JS-divergence":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_feats, t_feats], dim=0), 1.0))
            domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
            meta_loss += domain_loss * 0.1
        elif self.disc == "C-divergence":
            s_l_feats = torch.cat([s_feats, 20 * F.softmax(self.classifier_layer(s_feats), dim=1)], dim=1)
            t_l_feats = torch.cat([t_feats, 20 * self.one_hot_embedding(t_outputs)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_feats, t_l_feats], dim=0), 1.0))
            domain_labels = np.array([0] * s_inputs.shape[0] + [1] * t_inputs.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=s_inputs.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
            meta_loss += domain_loss * 0.1
        return meta_loss

    def inference(self, x):
        x = self.base_network(x).view(x.shape[0], -1)
        if self.use_bottleneck:
            x = self.bottleneck_layer(x)
        return self.classifier_layer(x)

    def one_hot_embedding(self, labels):
        y = torch.eye(self.num_classes, device=labels.device)
        return y[labels]