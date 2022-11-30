# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import OrderedDict
import torch
import torch.nn as nn
from .resnet import ResNet,Bottleneck
import torchvision.models as models

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'simsiam_256bs': '/sinergia/2022-fall-sp-jiguo/pretrained/checkpoint_0099_256bs.pth.tar',
    'simsiam_512bs': '/sinergia/2022-fall-sp-jiguo/pretrained/checkpoint_0099_512bs.pth.tar',
    'resnet50': '/sinergia/2022-fall-sp-jiguo/pretrained/resnet50-19c8e357.pth',
}


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

def _simsiam(arch, block, layers, pretrained, progress, **kwargs):
    assert(arch in model_urls.keys())
    # base_encoder = models.__dict__['resnet50']
    model = ResNet(block, layers, **kwargs)
    # model = SimSiam(base_encoder)
    # print(len(model.state_dict()))
    # for name in model.state_dict():
    #     print(name)
    if pretrained:
        checkpoint = torch.load(model_urls[arch])
        state_dict = checkpoint['state_dict']
        resnet_state_dict = torch.load(model_urls['resnet50'])
        # for name in state_dict:
        #     print(name)
        # for name in resnet_state_dict:
        #     print(name)
        # for name in model.state_dict():
        #     print(name)
        new_dict = OrderedDict()
        # print(len(state_dict.keys()))
        # print(len(resnet_state_dict))
        # print(len(model.state_dict()))
        # new_dict = {k.replace('module.encoder',''):v for k,v in state_dict.items() if k.replace('module.encoder','') in resnet_state_dict}
        # for k,v in state_dict.items():
        #     if k.replace('module.encoder','') in resnet_state_dict:
        #         resnet_state_dict[k.replace('module.encoder','')] = state_dict[k]
        for k,v in state_dict.items():
            # print(k.replace('module.encoder',''),'--')
            if k.replace('module.encoder.','') in model.state_dict():
                new_dict[k.replace('module.encoder.','')] = state_dict[k]
        new_dict['fc.weight']=resnet_state_dict['fc.weight']
        new_dict['fc.bias'] = resnet_state_dict['fc.bias']
        # for name in resnet_state_dict:
        #     print(name)
        # print(len(model.state_dict()),len(new_dict))
        assert(len(model.state_dict())==len(new_dict))
        model.load_state_dict(new_dict)
        # res50_simsiam = model.encoder
        # for name in res50_simsiam.state_dict():
        #     print(name)
    # print(model)
    return model


def resnet50_simsiam(pretrained=True, progress=True, **kwargs):
    r"""Simsiam with ResNet-50 as backbone
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _simsiam('simsiam_256bs', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)