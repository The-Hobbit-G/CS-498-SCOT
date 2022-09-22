"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add

import torch.nn.functional as F
import torch
import gluoncvth as gcv
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt

from . import geometry
from . import util
from . import rhm_map
#from torchvision.models import resnet
from . import resnet
from data.dataset import Normalize, UnNormalize


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def show_from_cv(img, kps,title):
    assert(type(title)==str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape, kps)
    for i in range(kps.shape[1]):
        cv2.circle(img,(kps[0][i],kps[1][i]),3,(0,0,255),-1)
    cv2.imwrite('/home/jianting/SCOT/visualization/'+title+'.jpg',img)


class SCOT_CAM:
    r"""SCOT framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device, cam):
        r"""Constructor for SCOT framework"""

        # Feature extraction network initialization.
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]
        
        elif backbone == 'fcn101':
            self.backbone = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device).pretrained
            if len(cam)==0:
                self.backbone1 = gcv.models.get_fcn_resnet101_voc(pretrained=True).to(device)
                self.backbone1.eval()
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.backbone.eval()

        # Hyperpixel id and pre-computed jump and receptive field size initialization
        # Reference: https://fomoro.com/research/article/receptive-field-calculator
        # (the jump and receptive field sizes for 'fcn101' are heuristic values)
        self.hyperpixel_ids = util.parse_hyperpixel(hyperpixel_ids) #string -> int
        #self.jsz = jump size, self.rfsz = receptive field size
        if backbone in ['resnet50', 'resnet101']:
            ##For ResNet the jump size(stride) in the feature map of the final convolutional layer is 16
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 16, 16]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)
        elif backbone in ['resnet50_ft', 'resnet101_ft']:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)
        else:
            self.jsz = torch.tensor([4, 4, 4, 4, 8, 8, 8, 8, 8, 8]).to(device)
            self.rfsz = torch.tensor([11, 19, 27, 35, 43, 59, 75, 91, 107, 139]).to(device)

        # Miscellaneous
        self.hsfilter = geometry.gaussian2d(7).to(device)
        self.device = device
        self.benchmark = benchmark
        self.detransform = UnNormalize() ##For visualizing the imgs

    def __call__(self, *args, **kwargs):
        r"""Forward pass"""
        maptype = args[6]
        src_bbox = args[7]
        trg_bbox = args[8]
        src_mask = args[9]
        trg_mask = args[10]
        backbone = args[11]
        src_kps = args[12]
        trg_kps = args[13]
        src_kps_feat = src_kps/self.jsz[self.hyperpixel_ids[0]]
        trg_kps_feat = trg_kps/self.jsz[self.hyperpixel_ids[0]]
        print('src_kps size: {}, trg_kps size:{}'.format(src_kps_feat.size(),trg_kps_feat.size()))
        print('image0 size: {}, image1 size: {}'.format(args[0].size(),args[1].size()))
        print('image type:{}'.format(type(args[0]),type(args[1])))

        ##visualize the souce and target images
        source_image = self.detransform(args[0])
        target_image = self.detransform(args[1])
        scr_image = tensor_to_np(source_image)
        trg_image = tensor_to_np(target_image)
        show_from_cv(scr_image,src_kps.cpu().numpy().astype(int),'source_image')
        show_from_cv(trg_image,trg_kps.cpu().numpy().astype(int),'target_image')
        #########

        src_hyperpixels = self.extract_hyperpixel(args[0], maptype, src_bbox, src_mask, backbone)
        trg_hyperpixels = self.extract_hyperpixel(args[1], maptype, trg_bbox, trg_mask, backbone)
        src_featmap = src_hyperpixels[-1]
        trg_featmap = trg_hyperpixels[-1]
        print('----src,trg featmap size : {},{}'.format(src_featmap.size(),trg_featmap.size()))
        C_mat = torch.einsum('kij,kmn -> ijmn',(src_featmap,trg_featmap))/(torch.norm(src_featmap,2)*torch.norm(trg_featmap,2))
        print('C_mat size: {}'.format(C_mat.size()))
        plt.figure(1)
        for i in range(src_kps_feat.size()[1]):
            plt.subplot(1,src_kps_feat.size()[1],i+1)
            plt.imshow(C_mat[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/C_correspondence_src')
        '''
        C_mat_trg = C_mat.permute(2,3,0,1)
        plt.figure(2)
        for i in range(trg_kps_feat.size()[1]):
            plt.subplot(1,trg_kps_feat.size()[1],i+1)
            plt.imshow(C_mat_trg[int(trg_kps_feat[0][i]),int(trg_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/C_correspondence_trg')
        '''

        confidence_ts = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, src_kps_feat, trg_kps_feat, C_mat, self.hsfilter, args[2], args[3], args[4], args[5])
        return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]


    def extract_hyperpixel(self, img, maptype, bbox, mask, backbone="resnet101"):
        r"""Given image, extract desired list of hyperpixels"""
        hyperfeats, rfsz, jsz, feat_map, fc = self.extract_intermediate_feat(img.unsqueeze(0), return_hp=True, backbone=backbone)
        print('image size:{}, feature map size:{}'.format(img.size(),feat_map.size()))
        hpgeometry = geometry.receptive_fields(rfsz, jsz, hyperfeats.size()).to(self.device)
        
        hyperfeats_orisize = copy.deepcopy(hyperfeats)
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], -1).t() ##heperfeats size: (3136,50,75)->(3136,3750)->(3750,3136)

        # Prune boxes on margins (Otherwise may cause error)
        if self.benchmark in ['TSS']:
            hpgeometry, valid_ids = geometry.prune_margin(hpgeometry, img.size()[1:], 10)
            hyperfeats = hyperfeats[valid_ids, :]

        weights = torch.ones(len(hyperfeats),1).to(hyperfeats.device) ##Since hyperfeats.size() = (3750,3136), len(hyperfeats)=3750
        print('--**--**weights size: {}'.format(weights.size()))
        if maptype in [1]: # weight points
            ##Mind that the given mask is None in our case since we don't specify args.cam as shown in evaluate_map_CAM.py
            if mask is None:
                # get CAM mask
                if backbone=='fcn101':
                    mask = self.get_FCN_map(img.unsqueeze(0), feat_map, fc, sz=(img.size(1),img.size(2)))
                else:
                    mask = self.get_CAM_multi(img.unsqueeze(0), feat_map, fc, sz=(img.size(1),img.size(2)), top_k=2)
                scale = 1.0
            else:
                scale = 255.0

            # print('--**--**mask size:{}'.format(mask.size))
            hpos = geometry.center(hpgeometry)
            hselect = mask[hpos[:,1].long(),hpos[:,0].long()].to(hpos.device)
            weights = 0.5*torch.ones(len(hyperfeats),1).to(hpos.device)

            weights[hselect>0.4*scale,:] = 0.8
            weights[hselect>0.5*scale,:] = 0.9
            weights[hselect>0.6*scale,:] = 1.0
            ##weights are gotten by applying staircase function to CAM and they will be used as prior distribution for OT problem
        
        return hpgeometry, hyperfeats, img.size()[1:][::-1], weights, hyperfeats_orisize


    def extract_intermediate_feat(self, img, return_hp=True, backbone='resnet101'):
        r"""Extract desired a list of intermediate features"""
        """This is the pre-processing (Input feature extraction) We extract features from feature maps from different layers"""

        feats = []
        rfsz = self.rfsz[self.hyperpixel_ids[0]]
        jsz = self.jsz[self.hyperpixel_ids[0]]
        #The receptive field size and the jump size are decided by the first layer in the hyperpixel id(the shallowest one)

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in self.hyperpixel_ids:
                feats.append(feat.clone())
                #if hid + 1 == max(self.hyperpixel_ids):
                #    break
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        # GAP feature map
        feat_map = feat
        if backbone!='fcn101':
            x = self.backbone.avgpool(feat)
            x = torch.flatten(x, 1)
            fc = self.backbone.fc(x)
        else:
            fc = None

        if not return_hp: # only return feat_map and fc
            return feat_map,fc

        # Up-sample & concatenate features to construct a hyperimage
        for idx, feat in enumerate(feats):
            if idx == 0:
                continue
            feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
        feats = torch.cat(feats, dim=1)
        ## feats.size() = (1 ,channels, h, w)  eg:(1,3136,50,75)
        ## So if we return feats[0], we will get an extracted feature map with size (3136, 50, 75) 
        # and this feature map is gotten by upsampling stacking the feature maps selected layers
        # In this way, we can make use of the multi-level representations

        print('****feats size:{}'.format(feats.size()))

        return feats[0], rfsz, jsz, feat_map, fc
    

    def get_CAM(self, feat_map, fc, sz, top_k=2):
        logits = F.softmax(fc, dim=1)
        scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
        pred_labels = pred_labels[0]
        bz, nc, h, w = feat_map.size()

        output_cam = []
        for label in pred_labels:
            cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
            cam = cam.view(1,1,h,w)
            cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
            cam = (cam-cam.min()) / cam.max()
            output_cam.append(cam)
        output_cam = torch.stack(output_cam,dim=0) # kxHxW
        output_cam = output_cam.max(dim=0)[0] # HxW

        return output_cam


    def get_CAM_multi(self, img, feat_map, fc, sz, top_k=2):
        # [img_h,img_w] = img.size()
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                if scale*scale*sz[0]*sz[1] > 800*800:
                    scale = min(800/img_h,800/img_w)
                    scale = min(1.5,scale)
                img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
                feat_map, fc = self.extract_intermediate_feat(img,return_hp=False)

            logits = F.softmax(fc, dim=1)
            scores, pred_labels = torch.topk(logits, k=top_k, dim=1)
            pred_labels = pred_labels[0]
            bz, nc, h, w = feat_map.size()

            output_cam = []
            for label in pred_labels:
                cam = self.backbone.fc.weight[label,:].unsqueeze(0).mm(feat_map.view(nc,h*w))
                cam = cam.view(1,1,h,w)
                cam = F.interpolate(cam, (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
                cam = (cam-cam.min()) / cam.max()
                output_cam.append(cam)
            output_cam = torch.stack(output_cam,dim=0) # kxHxW
            output_cam = output_cam.max(dim=0)[0] # HxW
            
            map_list.append(output_cam)
        map_list = torch.stack(map_list,dim=0)
        #map_list size = (3,199,300), same as the image size
        # print('--**map list size:{}'.format(map_list.size()))
        sum_cam = map_list.sum(0)
        norm_cam = sum_cam / (sum_cam.max()+1e-5)
        # print('--**mask size:{}'.format(norm_cam.size()))
        #norm_cam(mask) size is (199*300)=(image_H, image_W)

        return norm_cam


    def get_FCN_map(self, img, feat_map, fc, sz):
        #scales = [1.0,1.5,2.0]
        scales = [1.0]
        map_list = []
        for scale in scales:
            if scale*scale*sz[0]*sz[1] > 1200*800:
                scale = 1.5
            img = F.interpolate(img, (int(scale*sz[0]),int(scale*sz[1])), None, 'bilinear', True) # 1x3xHxW
            #feat_map, fc = self.extract_intermediate_feat(img,return_hp=False,backbone='fcn101')
            feat_map = self.backbone1.evaluate(img)
            
            predict = torch.max(feat_map, 1)[1]
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map
