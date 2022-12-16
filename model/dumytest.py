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





def get_pred(color, classes):
    color = F.to_tensor(color).unsqueeze(0)
    masks = torch.zeros((len(classes)+1, *color.shape[-2:]), dtype=torch.uint8)
    for i, clas in enumerate(classes.keys()):
        mask = (color*255).to(torch.uint8)==torch.tensor(clas).unsqueeze(1).unsqueeze(2)
        mask = (mask.sum(0)==3).to(torch.uint8)
        masks[i,] = mask
    masks[-1,] = (masks.sum(0)==0).to(torch.uint8)
    return masks

def get_mask(pred):
    '''
    Expects predictions without the BG class. Converts Soft probabilities to one-hot. Adds BG
    '''
    # b ,c, h, w
    pred = torch.cat([pred, torch.zeros((pred.shape[0],1,*pred.shape[2:])).cuda()], dim=1)
    prob, clas = pred.max(dim=1, keepdims=True)
    mask = pred.ge(prob).int()
    return mask

def get_color(mask, classes):
    '''
    Expects one-hot masks with the BG. Converts them to RGB
    '''
    #mask = mask[:,1:,]
    device = mask.device
    colors = torch.tensor([*list(classes.keys()), (0,0,0)])
    if mask.shape[1] == 0:
        mask = torch.zeros((mask.shape[0], 1, *mask.shape[2:]))
    val, clas = mask.max(dim=1)
    clas[val==0] = -1
    im_color = colors[clas,:].permute(0,3,1,2).to(device)
    im_color = im_color.to(torch.float32)/255.0
    return im_color
 
def accuracy(pred, gt):
    mask = get_mask(pred)
    fg = 1 - gt[:,-1:,:,:]
    mask = mask * fg
    
    c = gt.shape[1]
    n = fg.sum(-1).sum(-1)
    
    true = ((mask == gt).sum(1, keepdims=True)==c).sum(-1).sum(-1)
    inter = ((mask + gt)>=1.9).sum(-1).sum(-1)
    uni = ((mask + gt)>=0.9).sum(-1).sum(-1)
    acc = true.float() / n
    return acc, inter, uni
