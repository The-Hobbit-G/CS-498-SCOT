r"""Helper functions"""

import logging
import re

import torch.nn.functional as F
import torch

### Hook the models
from collections import OrderedDict
class ModuleHook:
    #### Example usage: encoder_hook, encoder_layers = hook_model(encoder, include_class_name=False)
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, include_class_name=False):
    features = OrderedDict()
    layers = OrderedDict()
    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                hook_layers(layer, prefix=prefix + [name])
                if include_class_name:
                    features["-".join(prefix + [name]) + f":{layer.__class__.__name__}"] = ModuleHook(layer)
                    layers["-".join(prefix + [name]) + f":{layer.__class__.__name__}"] = layer.__repr__()
                else:
                    features["-".join(prefix + [name])] = ModuleHook(layer)
                    layers["-".join(prefix + [name])] = layer.__repr__()

    hook_layers(model)

    def hook(layer):
        assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
        out = features[layer].features
        assert out is not None, "There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook, layers


def init_logger(logfile):
    r"""Initialize logging settings"""
    logging.basicConfig(filemode='w',
                        filename=logfile,
                        level=logging.INFO,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M:%S')

    # Configuration on console logs
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_args(args):
    r"""Log program arguments"""
    logging.info('\n+========== SCOT Arguments ===========+')
    for arg_key in args.__dict__:
        logging.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    logging.info('+================================================+\n')


def where(predicate):
    r"""Returns indices which match given predicate"""
    matching_idx = predicate.nonzero()
    n_match = len(matching_idx)
    if n_match != 0:
        matching_idx = matching_idx.t().squeeze(0)
    return matching_idx


def intersect1d(tensor1, tensor2):
    r"""Takes two 1D tensor and returns tensor of common values"""
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort()[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def parse_hyperpixel(hyperpixel_ids):
    r"""Parse given hyperpixel list (string -> int)"""
    return list(map(int, re.findall(r'\d+', hyperpixel_ids)))


def get_bbox_mask(mask, thres=0.0):
    """mask:HxW"""
    pos = (mask>=thres).nonzero()
    pmin = pos.min(0)[0] # top-left
    pmax = pos.max(0)[0] # bottom-right
    bbox = torch.tensor([pmin[1],pmin[0],pmax[1],pmax[0]])

    return bbox


def resize(img, kps, side_thres=300):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    kps = kps.float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    #if side_max > side_thres:
    if True:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
        kps *= inter_ratio
    return img.squeeze(0), kps, inter_ratio


def resize_mask(mask, imsize):
    mask = mask.unsqueeze(0).float()
    mask = F.interpolate(mask, size=(imsize[1],imsize[2]),
                        mode='bilinear', align_corners=False)
    return mask[0][0]


def resize_TSS(img, side_thres=300):
    r"""Resize given image with imsize: (1, 3, H, W)"""
    imsize = torch.tensor(img.size()).float()
    side_max = torch.max(imsize)
    inter_ratio = 1.0
    #if side_max > side_thres:
    if True:
        inter_ratio = side_thres / side_max
        img = F.interpolate(img,
                            size=(int(imsize[2] * inter_ratio), int(imsize[3] * inter_ratio)),
                            mode='bilinear',
                            align_corners=False)
        return img.squeeze(0), imsize[1:], img.size()[1:], inter_ratio


