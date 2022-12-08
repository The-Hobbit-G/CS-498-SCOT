import resnet
from util import *
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res_model = resnet.resnet50(pretrained=True).to(device)

reshook,reslayers = hook_model(res_model.layer1)
layer = nn.Linear(10,200000)

layer_hook = ModuleHook(layer)

for i in range(200):
    a = torch.randn((1,10))
    tic = time.time()
    for j in range(200):
        layer(a)
    toc = time.time()
    print(f'time passing a layer {toc-tic}')
