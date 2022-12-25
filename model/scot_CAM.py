"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add

import torch.nn.functional as F
import torch
import torchvision.utils
import time
import torchvision.transforms.functional as TF
# import gluoncvth as gcv
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from torchvision import transforms

from . import geometry
from . import util
from . import rhm_map
#from torchvision.models import resnet
from . import resnet
from . import simsiam
from data.dataset import Normalize, UnNormalize
from scipy.optimize import linear_sum_assignment
from .clip import *


clip_urls = {
    'resnet50_clip': '/scratch/2022-fall-sp-jiguo/pretrained/CLIP_RN50.pt',
    'resnet101_clip': '/scratch/2022-fall-sp-jiguo/pretrained/CLIP_RN101.pt',
}


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def show_from_cv(img, kps):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape, kps)
    img_list = []
    for i in range(kps.shape[1]):
        img_i = cv2.circle(img.copy(),(kps[0][i],kps[1][i]),3,(0,0,255),-1)
        img_i = cv2.cvtColor(img_i, cv2.COLOR_RGB2BGR)
        img_list.append(img_i)
    #cv2.imwrite('/scratch/2022-fall-sp-jiguo/SCOT/visualization/'+title+'.jpg',img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_list


def denormalize_gta(image):
    im = TF.normalize(image, mean=[-1, -1, -1], std=[2, 2, 2])
    return im


class SCOT_CAM:
    r"""SCOT framework"""
    def __init__(self, backbone, hyperpixel_ids, benchmark, device, cam):
        r"""Constructor for SCOT framework"""

        # Feature extraction network initialization.
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True).to(device)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet50_simsiam':
            self.backbone = simsiam.resnet50_simsiam(pretrained=True).to(device)
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

        elif backbone == 'resnet50_densecl_IN':
            self.backbone = resnet.resnet50_densecl(pretrained=True,pretrained_dataset='ImageNet').to(device)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet50_densecl_COCO':
            self.backbone = resnet.resnet50_densecl(pretrained=True,pretrained_dataset='COCO').to(device)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet101_densecl_IN':
            self.backbone = resnet.resnet101_densecl(pretrained=True).to(device)
            nbottlenecks = [3, 4, 23, 3]

        elif backbone == 'resnet50_clip':
            clip50_dict = torch.jit.load(clip_urls[backbone],map_location=device).state_dict()
            clip50 = build_model(clip50_dict).to(device)
            self.backbone1 = clip50.visual.to(device)
            for p in self.backbone1.parameters():
                p.data = p.data.to(torch.float32)
            self.backbone1.eval()
            self.backbone = resnet.resnet50(pretrained=True).to(device) ##in order to get fixed fc layers for extracting CAM
            self.hook, self.layers = util.hook_model(self.backbone1)
            nbottlenecks = [3, 4, 6, 3]

        elif backbone == 'resnet101_clip':
            clip101_dict = torch.jit.load(clip_urls[backbone],map_location=device).state_dict()
            clip101 = build_model(clip101_dict).to(device)
            self.backbone1 = clip101.visual.to(device)
            for p in self.backbone1.parameters():
                p.data = p.data.to(torch.float32)
            self.backbone1.eval()
            self.backbone = resnet.resnet101(pretrained=True).to(device) ##in order to get fixed fc layers for extracting CAM
            self.hook, self.layers = util.hook_model(self.backbone1)
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
        if backbone in ['resnet50', 'resnet101','resnet50_simsiam','resnet50_densecl_IN','resnet50_densecl_COCO','resnet101_densecl_IN','resnet50_clip','resnet101_clip']:
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
        print(self.backbone)

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
        # src_kps_feat = src_kps/self.jsz[self.hyperpixel_ids[0]]
        # trg_kps_feat = trg_kps/self.jsz[self.hyperpixel_ids[0]]
        factorization = args[14]
        k=args[15]
        activation = args[16]
        normalization = args[17]
        # print('src_kps size: {}, trg_kps size:{}'.format(src_kps_feat.size(),trg_kps_feat.size()))
        # print('image0 size: {}, image1 size: {}'.format(args[0].size(),args[1].size()))
        # print('image type:{}'.format(type(args[0]),type(args[1])))

        ##visualize the souce and target images
        # source_image = self.detransform(args[0])
        # target_image = self.detransform(args[1])
        # scr_image = tensor_to_np(source_image)
        # trg_image = tensor_to_np(target_image)
        # scr_image_with_rps = show_from_cv(scr_image,src_kps.cpu().numpy().astype(int))
        # trg_image_with_rps = show_from_cv(trg_image,trg_kps.cpu().numpy().astype(int))
        #########
        tic = time.time()
        if backbone in ['resnet50_clip','resnet101_clip']:
            if self.benchmark == 'gta':
                source_image = denormalize_gta(args[0])
                target_image = denormalize_gta(args[1])
            else:
                source_image = self.detransform(args[0])
                target_image = self.detransform(args[1])
            clip_normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
             std=[0.26862954, 0.26130258, 0.27577711])
            source_image = clip_normalize(source_image)
            target_image = clip_normalize(target_image)
            src_hyperpixels = self.extract_hyperpixel(source_image, maptype, src_bbox, src_mask, backbone)
            trg_hyperpixels = self.extract_hyperpixel(target_image, maptype, trg_bbox, trg_mask, backbone)  
        else:
            src_hyperpixels = self.extract_hyperpixel(args[0], maptype, src_bbox, src_mask, backbone)
            trg_hyperpixels = self.extract_hyperpixel(args[1], maptype, trg_bbox, trg_mask, backbone)

        # print(f'extract hyper-pixel time: {time.time()-tic}')
        src_featmap = src_hyperpixels[-1]
        trg_featmap = trg_hyperpixels[-1]
        # print('----src,trg featmap size : {},{}'.format(src_featmap.size(),trg_featmap.size()))

        # num_kps = src_kps_feat.size()[1]
        
        """Visualize cross-similarity C"""
        
        # C_mat = torch.einsum('kij,kmn -> ijmn',(src_featmap,trg_featmap))/(torch.norm(src_featmap,2)*torch.norm(trg_featmap,2))
        # print('C_mat size: {}'.format(C_mat.size()))
        

        """Visualize cross-similarity OT matrix T and p(m|D) after RHM"""
        ##cross-similarity T for the source image
        confidence_ts, OT_mat, C_2dim = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter, args[2], args[3], args[4], args[5],k,factorization,activation,normalization)
        # print('**confidence_ts size: {}'.format(confidence_ts.size()))
        # confidence_ts_orisize = confidence_ts.view_as(C_mat)
        # OT_mat_orisize = OT_mat.view_as(C_mat)
        # print('**confidence_ts original size:{}'.format(confidence_ts_orisize.size())
        

        return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]

        ##try removing RHM
        # return OT_mat, src_hyperpixels[0], trg_hyperpixels[0]

    def visualize_sim(self,sample,idx,maptype,exp1,exp2,eps,savepath,backbone="resnet101",sim='All',simi='OT',choice='cross'):
        """Given two images(src & trg), the backbone and the sim choice, visualize sim(backbone(src).T @ backbone(trg))"""
        src_img = sample['src_img']
        trg_img = sample['trg_img']
        src_bbox = sample['src_bbox']
        trg_bbox = sample['trg_bbox']
        src_mask = sample['src_mask']
        trg_mask = sample['trg_mask']
        src_kps = sample['src_kps']
        trg_kps = sample['trg_kps']
        vis_idx = str(idx)
        # print(src_kps.shape,trg_kps.shape)
        # print(src_kps,trg_kps)
        pair_class = sample['pair_class']
        pair_classid = str(sample['pair_classid'])
        sim_list = ['Correlation','OT','RHM','All']
        assert(sim in sim_list)
        choice_list = ['cross','self']
        assert(choice in choice_list)
        src_kps_feat = src_kps/self.jsz[self.hyperpixel_ids[0]]
        trg_kps_feat = trg_kps/self.jsz[self.hyperpixel_ids[0]]
        num_kps = src_kps_feat.size()[1]
        source_image = self.detransform(src_img)
        target_image = self.detransform(trg_img)
        scr_image = tensor_to_np(source_image)
        trg_image = tensor_to_np(target_image)
        scr_image_with_rps_list = show_from_cv(scr_image, src_kps.cpu().numpy().astype(int))
        trg_image_with_rps_list = show_from_cv(trg_image, trg_kps.cpu().numpy().astype(int))
        assert(len(scr_image_with_rps_list)==len(trg_image_with_rps_list)==num_kps)
        

        src_hyperpixels = self.extract_hyperpixel(src_img, maptype, src_bbox, src_mask, backbone)
        trg_hyperpixels = self.extract_hyperpixel(trg_img, maptype, trg_bbox, trg_mask, backbone)
        src_featmap = src_hyperpixels[-1]
        trg_featmap = trg_hyperpixels[-1]

        if choice == 'cross':
            C_mat = torch.einsum('kij,kmn -> ijmn',(src_featmap,trg_featmap))/(torch.norm(src_featmap,2)*torch.norm(trg_featmap,2))
            confidence_ts, OT_mat, C_2dim = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter, simi, exp1, exp2, eps)
            confidence_ts_orisize = confidence_ts.view_as(C_mat)
            OT_mat_orisize = OT_mat.view_as(C_mat)
        else:
            C_mat = torch.einsum('kij,kmn -> ijmn',(src_featmap,src_featmap))/(torch.norm(src_featmap,2)*torch.norm(src_featmap,2))
            confidence_ts, OT_mat, C_2dim = rhm_map.rhm(src_hyperpixels, src_hyperpixels, self.hsfilter, simi, exp1, exp2, eps)
            confidence_ts_orisize = confidence_ts.view_as(C_mat)
            OT_mat_orisize = OT_mat.view_as(C_mat)
        
        visual_dic = {'Correlation':C_mat, 'OT':OT_mat_orisize, 'RHM':confidence_ts_orisize}
        if sim == 'All':
            plt.figure(figsize=(3*num_kps,3*5))
            # plt.subplot(4,num_kps,1)
            # plt.title('source image')
            # plt.axis('off')
            # plt.imshow(scr_image_with_rps)
            # plt.subplot(4,num_kps,2)
            # plt.title('target image')
            # plt.axis('off')
            # plt.imshow(trg_image_with_rps)
            for i in range(num_kps):
            ##Correlation matrix C
                plt.subplot(5,num_kps,i+1)
                plt.title('source image')
                plt.axis('off')
                plt.imshow(scr_image_with_rps_list[i])

                plt.subplot(5,num_kps,i+1+num_kps)
                plt.title('target image')
                plt.axis('off')
                plt.imshow(trg_image_with_rps_list[i])

                plt.subplot(5,num_kps,i+1+2*num_kps)
                plt.title('Correlation matrix')
                plt.axis('off')
                plt.imshow(C_mat[int(min(max(src_kps_feat[0][i],0),C_mat.shape[0]-1)),\
                    int(min(max(src_kps_feat[1][i],0),C_mat.shape[1]-1)),:,:].cpu().numpy())
                # plt.colorbar()
                ##OT matrix T
                plt.subplot(5,num_kps,i+1+3*num_kps)
                plt.title('OT matrix')
                plt.axis('off')
                plt.imshow(OT_mat_orisize[int(min(max(src_kps_feat[0][i],0),OT_mat_orisize.shape[0]-1)),\
                    int(min(max(src_kps_feat[1][i],0),OT_mat_orisize.shape[1]-1)),:,:].cpu().numpy()) 
                # plt.colorbar()
                #RHM confidence
                plt.subplot(5,num_kps,i+1+4*num_kps)
                plt.title('RHM')
                plt.axis('off')
                plt.imshow(confidence_ts_orisize[int(min(max(src_kps_feat[0][i],0),confidence_ts_orisize.shape[0]-1)),\
                    int(min(max(src_kps_feat[1][i],0),confidence_ts_orisize.shape[1]-1)),:,:].cpu().numpy())
                # plt.colorbar()
            plt.savefig(savepath+pair_class+vis_idx+backbone+'_all_three_matrices_'+choice+'_similarity')
        else:
            plt.figure(figsize=(3*num_kps,3*3))
            # plt.subplot(2,num_kps,1)
            # plt.title('source image')
            # plt.axis('off')
            # plt.imshow(scr_image_with_rps)
            # plt.subplot(2,num_kps,2)
            # plt.title('target image')
            # plt.axis('off')
            # plt.imshow(trg_image_with_rps)
            for i in range(num_kps):
            ##Correlation matrix C
                plt.subplot(3,num_kps,i+1)
                plt.title('source image')
                plt.axis('off')
                plt.imshow(scr_image_with_rps_list[i])

                plt.subplot(3,num_kps,i+1+num_kps)
                plt.title('target image')
                plt.axis('off')
                plt.imshow(trg_image_with_rps_list[i])

                plt.subplot(3,num_kps,i+1+2*num_kps)
                plt.title(sim+' matrix')
                plt.axis('off')
                plt.imshow(visual_dic[sim][int(min(max(src_kps_feat[0][i],0),visual_dic[sim].shape[0]-1)),\
                    int(min(max(src_kps_feat[1][i],0),visual_dic[sim].shape[1]-1)),:,:].cpu().numpy())
                # plt.colorbar()
            plt.savefig(savepath+pair_class+vis_idx+backbone+'_'+sim+'_matrices_'+choice+'_similarity')



    def visualize_G(self, sample, visual_idx, maptype, savepath, k_list=[3,7,20,35], backbone="resnet101", f = 'KMeans', choice='src'):
        src_img = sample['src_img']
        trg_img = sample['trg_img']
        src_bbox = sample['src_bbox']
        trg_bbox = sample['trg_bbox']
        src_mask = sample['src_mask']
        trg_mask = sample['trg_mask']
        pair_class = sample['pair_class']
        vis_idx = str(visual_idx)
        f_list = ['PCA' ,'NMF','KMeans','All']
        assert(f in f_list)
        choice_list = ['src','trg']
        assert(choice in choice_list)
        num_k = len(k_list)
        source_image = self.detransform(src_img)
        target_image = self.detransform(trg_img)
        scr_image = tensor_to_np(source_image)
        trg_image = tensor_to_np(target_image)
        if choice == 'src':
            hyperpixels = self.extract_hyperpixel(src_img, maptype, src_bbox, src_mask, backbone)
            shown_image = scr_image
        elif choice == 'trg':
            hyperpixels = self.extract_hyperpixel(trg_img, maptype, trg_bbox, trg_mask, backbone)
            shown_image = trg_image
        else:
            raise Exception('image not in src or trg')

        hyperfeat_orisize = hyperpixels[-1]
        n_rows = [int(i**0.5) for i in k_list]
        
        if f == 'KMeans':
            L_list_kmeans,GT_list_kmeans = self.visualize_k_means(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            for i in range(num_k):
                img_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_kmeans[i]),n_rows[i])
                torchvision.utils.save_image(img_grid,savepath+pair_class+vis_idx+'_'+choice+'_'+backbone+'_G_mat_k_means_k={}.jpg'.format(k_list[i]))
        elif f == 'PCA':
            L_list_pca,GT_list_pca = self.visualize_pca(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            for i in range(num_k):
                img_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_pca[i]),n_rows[i])
                torchvision.utils.save_image(img_grid,savepath+pair_class+vis_idx+'_'+choice+'_'+backbone+'_G_mat_pca_k={}.jpg'.format(k_list[i]))
        elif f == 'NMF':
            L_list_nmf,GT_list_nmf = self.visualize_nmf(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            for i in range(num_k):
                img_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_nmf[i]),n_rows[i])
                torchvision.utils.save_image(img_grid,savepath+pair_class+vis_idx+'_'+choice+'_'+backbone+'_G_mat_nmf_k={}.jpg'.format(k_list[i]))
        elif f == 'All':
            L_list_kmeans,GT_list_kmeans = self.visualize_k_means(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            L_list_pca,GT_list_pca = self.visualize_pca(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            L_list_nmf,GT_list_nmf = self.visualize_nmf(hyperpixels[1],hyperfeat_orisize[0,:,:],k_list)
            for i in range(num_k):
                # kmeans_grid = torchvision.utils.make_grid(GT_list_kmeans[i],n_rows[i])
                # kmeans_grid = kmeans_grid.cpu().numpy()
                Li_pca = L_list_pca[i]
                Li_kmeans = L_list_kmeans[i]
                Li_nmf = L_list_nmf[i]

                Li_pca_norm = np.linalg.norm(Li_pca, ord=2, axis=0)
                Li_kmeans_norm = np.linalg.norm(Li_kmeans, ord=2, axis=0)
                Li_nmf_norm = np.linalg.norm(Li_nmf, ord=2, axis=0)

                Li_pca_norm = np.expand_dims(Li_pca_norm,axis=1)
                Li_kmeans_norm = np.expand_dims(Li_kmeans_norm,axis=0)
                Li_nmf_norm = np.expand_dims(Li_nmf_norm,axis=0)
                
                #correlation matrix L1.T@L2-->k*k matrix
                # print(Li_pca.shape,Li_kmeans.shape,Li_nmf.shape)
                C_pca_kmeas = (Li_pca.T@Li_kmeans)/(Li_pca_norm@Li_kmeans_norm)
                C_pca_nmf = (Li_pca.T@Li_nmf)/(Li_pca_norm@Li_nmf_norm)
                # print('max and min of C_pca_kmeas are:{},{}'.format(np.max(C_pca_kmeas),np.min(C_pca_kmeas)))
                # print('max and min of C_pca_nmf are:{},{}'.format(np.max(C_pca_nmf),np.min(C_pca_nmf)))

                # C_pca_kmeas = torch.pow(torch.clamp(C_pca_kmeas, min=0), 1.0)
                # C_pca_nmf = torch.pow(torch.clamp(C_pca_nmf, min=0), 1.0)
                C_pca_kmeas = np.power(np.maximum(C_pca_kmeas,0),1.0)
                C_pca_nmf = np.power(np.maximum(C_pca_nmf,0),1.0)

                # print('max and min of C_pca_kmeas are:{},{}'.format(np.max(C_pca_kmeas),np.min(C_pca_kmeas)))
                # print('max and min of C_pca_nmf are:{},{}'.format(np.max(C_pca_nmf),np.min(C_pca_nmf)))



                row_ind_pca_kmeans, col_ind_pca_kmeans = linear_sum_assignment(1-C_pca_kmeas)
                row_ind_pca_nmf, col_ind_pca_nmf = linear_sum_assignment(1-C_pca_nmf)
                GT_list_kmeans[i] = GT_list_kmeans[i][col_ind_pca_kmeans,:]
                GT_list_nmf[i] = GT_list_nmf[i][col_ind_pca_nmf,:]
                
                pca_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_pca[i]),n_rows[i]).permute(1,2,0)
                kmeans_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_kmeans[i]),n_rows[i]).permute(1,2,0)
                nmf_grid = torchvision.utils.make_grid(torch.from_numpy(GT_list_nmf[i]),n_rows[i]).permute(1,2,0)
                pca_grid = pca_grid.numpy().astype(np.float64)
                kmeans_grid = kmeans_grid.numpy().astype(np.float64)
                nmf_grid = nmf_grid.numpy().astype(np.float64)
                plt.figure(figsize=(4*4,4))
                plt.subplot(1,4,1)
                plt.title('Image')
                plt.axis('off')
                plt.imshow(shown_image)
                plt.subplot(1,4,2)
                plt.title('G_mat with PCA factorization')
                plt.axis('off')
                plt.imshow(pca_grid)
                plt.subplot(1,4,3)
                plt.title('G_mat with KMeans factorization')
                plt.axis('off')
                plt.imshow(kmeans_grid)
                plt.subplot(1,4,4)
                plt.title('G_mat with NMF factorization')
                plt.axis('off')
                plt.imshow(nmf_grid)
                plt.savefig(savepath+pair_class+vis_idx+'_'+choice+'_'+backbone+'_G_mat_all3_k={}'.format(k_list[i]))

    def visualize_k_means(self, hyperfeats, C_orisize, k_list):
        num_k = len(k_list)
        hyperfeats = F.relu(hyperfeats)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        L_list = []
        for k in k_list:
            km = KMeans(n_clusters=k,max_iter=500).fit(hyperfeats)
            # print(km.cluster_centers_.shape) #k*C
            # G = hyperfeats@np.linalg.pinv(km.cluster_centers_)
            # print(G)
            # G_list.append(G)
            GT = np.zeros((k,hyperfeats.shape[0])) #GT shape:k*HW
            assert(hyperfeats.shape[0]==km.labels_.shape[0]) # ==HW
            for label in range(km.labels_.shape[0]):
                GT[km.labels_[label],label]=1
            # print(GT)
            # L = hyperfeats.T @ np.linalg.pinv(GT)
            L = km.cluster_centers_.T #C*k

            # print(np.linalg.norm(hyperfeats.T @ np.linalg.pinv(GT)-L))
            # print(np.linalg.norm(L))
            print('KMeans reconstruction error and hyperfeats norm with k={}'.format(k))
            print(np.linalg.norm(km.cluster_centers_.T@GT - hyperfeats.T))
            print(np.linalg.norm(hyperfeats))
            
            # L = hyperfeats.T @ GT.T
            L_list.append(L)
            GT = GT.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1])
            GT_list.append(GT)
        assert(len(GT_list)==num_k)
        # plt.figure()
        # for i in range(num_k):
        #     img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
        #     torchvision.utils.save_image(img_grid,'/scratch/2022-fall-sp-jiguo/SCOT/visualization/G_mat_k_means_k={}.jpg'.format(k_list[i]))
        return  L_list,GT_list

    def visualize_pca(self, hyperfeats, C_orisize, k_list):
        num_k = len(k_list)
        hyperfeats = F.relu(hyperfeats)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        L_list = []
        for k in k_list:
            pca = PCA(n_components=k)
            # pca.fit(hyperfeats)
            # G = pca.transform(hyperfeats)
            # singular_values = pca.singular_values_
            # print(singular_values)
            # print(G)

            '''
            U,S,Vh = np.linalg.svd(hyperfeats.T)
            L = U[:,0:k]@np.diag(S[0:k])
            G = Vh[0:k,:].T
            '''


            G = pca.fit_transform(hyperfeats) #G shape: HW*k
            # L = hyperfeats.T @ np.linalg.pinv(G.T)
            L = pca.components_.T  #C*k

            
            
            # print(np.linalg.norm(L-hyperfeats.T @ np.linalg.pinv(G.T)))
            # print(np.linalg.norm(L))
            print('PCA reconstruction error and hyperfeats norm with k={}'.format(k))
            print(np.linalg.norm(L@G.T-hyperfeats.T))
            print(np.linalg.norm(hyperfeats))

            L_list.append(L)
            
            # GT = torch.from_numpy(G.T.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1]))
            GT = G.T.reshape(k,C_orisize.shape[0],C_orisize.shape[1])
            GT_rgb = np.zeros((k,3,C_orisize.shape[0],C_orisize.shape[1]))
            for i in range(k):
                #illustrate negative values in red and positive values in blue
                GT_rgb[i,0,:,:] = np.abs(np.minimum(GT[i,:,:],0))
                GT_rgb[i,2,:,:] = np.maximum(GT[i,:,:],0)
            # print(GT)
            GT_list.append(GT_rgb)
        assert(len(GT_list)==num_k)
        # plt.figure()
        # for i in range(num_k):
        #     img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
        #     torchvision.utils.save_image(img_grid,'/scratch/2022-fall-sp-jiguo/SCOT/visualization/G_mat_pca_k={}.jpg'.format(k_list[i]))
        return L_list,GT_list

    def visualize_nmf(self, hyperfeats, C_orisize, k_list):
        num_k = len(k_list)
        hyperfeats = F.relu(hyperfeats)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        L_list = []
        for k in k_list:
            nmf = NMF(n_components=k)
            G = nmf.fit_transform(hyperfeats) #G shape: HW*k
            # L = hyperfeats.T @ np.linalg.pinv(G.T)
            L = nmf.components_.T
            # print(np.linalg.norm(L-hyperfeats.T @ np.linalg.pinv(G.T)))
            # print(np.linalg.norm(L))
            print('NMF reconstruction error and hyperfeats norm with k={}'.format(k))
            print(np.linalg.norm(nmf.components_.T@G.T-hyperfeats.T))
            print(np.linalg.norm(hyperfeats))

            L_list.append(L)
            GT = G.T.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1])
            GT_list.append(GT)
        assert(len(GT_list)==num_k)
        # plt.figure()
        # for i in range(num_k):
        #     img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
        #     torchvision.utils.save_image(img_grid,'/scratch/2022-fall-sp-jiguo/SCOT/visualization/G_mat_nmf_k={}.jpg'.format(k_list[i]))
        return L_list,GT_list


    def plot_selfsim_statistic(self, C, T, RHM, C_orisize, scr_image_with_rps):
        """Visualize the entropy, mean, sum, std of self-similarity C(C_2dim_selfsim) and T(OT_mat_selfsim) over axis =1 so that [HW*HW] --> [HW,1]"""
        print(C[C<0],T[T<0],RHM[RHM<0])
        ##Compute the entropy of C using its histogram


        # PC = F.softmax(C,dim=1)
        n_bins = 64
        C_min = C.cpu().numpy().min()
        C_max = C.cpu().numpy().max()
        for i in range(C.shape[0]):
            his = torch.histogram(C[i,:].cpu(),bins=n_bins,range=(C_min,C_max))
            row_his = his.hist
            row_his = row_his/row_his.sum()
            row_his = row_his.view(1,-1)
            # print(row_his.shape)
            if i == 0:
                C_his = row_his
            else:
                C_his = torch.cat((C_his,row_his),0)
        # print(C_his.shape, C.shape)
        assert(C_his.shape[0]==C.shape[0])


        PT = F.softmax(T,dim=1)
        PRHM = F.softmax(RHM,dim=1)
        # if activation == 'softmax':
        #     C = PC
        # elif activation == 'Relu':
        #     C =  F.relu(C,dim=1)

        # lnPC = torch.log(PC)
        lnC_his = torch.log(C_his+1e-20)


        lnPT = torch.log(PT)
        lnPRHM = torch.log(PRHM)

        C_entropy = -torch.sum(C_his*lnC_his,dim =1)/np.log(n_bins)
        T_entropy = -torch.sum(PT*lnPT,dim =1)
        RHM_entropy = -torch.sum(PRHM*lnPRHM,dim =1)

        # print(C_entropy.max(),T_entropy.max(),RHM_entropy.max())
        # print(C_entropy.min(),T_entropy.min(),RHM_entropy.min())

        # if activation == 'softmax':
        #     C_entropy = F.softmax(C_entropy,dim=0)
        #     T_entropy = F.softmax(T_entropy,dim=0)
        #     RHM_entropy = F.softmax(RHM_entropy,dim=0)
        # elif activation == 'ReLU':
        #     C_entropy = F.relu(C_entropy)
        #     T_entropy = F.relu(T_entropy)
        #     RHM_entropy = F.relu(RHM_entropy)


        # print(C_entropy,T_entropy,RHM_entropy)


        # print(T_entropy.max(),T_entropy.min())
        # print(C_entropy.dtype,T_entropy.dtype,RHM_entropy.dtype)
        # C_entropy = C_entropy/C_entropy.max()
        # T_entropy = T_entropy/T_entropy.max()
        # RHM_entropy = RHM_entropy/RHM_entropy.max()
        # print(C_entropy.dtype,T_entropy.dtype,RHM_entropy.dtype)
        C_entropy = C_entropy.view_as(C_orisize)
        T_entropy = T_entropy.view_as(C_orisize)
        RHM_entropy = RHM_entropy.view_as(C_orisize)
        # print(C_entropy,T_entropy,RHM_entropy)
        # print(T_entropy.max(),T_entropy.min())
        # print(T_entropy.cpu().numpy().dtype)


        C_mean = torch.mean(C,dim=1).view_as(C_orisize)
        T_mean = torch.mean(T,dim=1).view_as(C_orisize)
        RHM_mean = torch.mean(RHM,dim=1).view_as(C_orisize)
        # C_mean = C_mean/C_mean.max()
        # T_mean = T_mean/T_mean.max()
        # RHM_mean = RHM_mean/RHM_mean.max()

        C_sum = torch.sum(C,dim=1).view_as(C_orisize)
        T_sum = torch.sum(T,dim=1).view_as(C_orisize)
        RHM_sum = torch.sum(RHM,dim=1).view_as(C_orisize)
        # C_sum = C_sum/C_sum.max()
        # T_sum = T_sum/T_sum.max()
        # RHM_sum = RHM_sum/RHM_sum.max()

        # print(T_mean, T_sum)

        C_std = torch.std(C,dim=1).view_as(C_orisize)
        T_std = torch.std(T,dim=1).view_as(C_orisize)
        RHM_std = torch.std(RHM,dim=1).view_as(C_orisize)
        # C_std = C_std/C_std.max()
        # T_std = T_std/T_std.max()
        # RHM_std = RHM_std/RHM_std.max()

        plt.figure(figsize=(3*3,3*5))
        plt.subplot(5,3,1)
        plt.title('source image')
        plt.axis('off')
        plt.imshow(scr_image_with_rps)

        plt.subplot(5,3,4)
        plt.title('C entropy')
        plt.axis('off')
        plt.imshow(C_entropy.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,5)
        plt.title('T entropy')
        plt.axis('off')
        plt.imshow(T_entropy.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,6)
        plt.title('RHM entropy')
        plt.axis('off')
        plt.imshow(RHM_entropy.cpu().numpy())
        plt.colorbar()

        plt.subplot(5,3,7)
        plt.title('C mean')
        plt.axis('off')
        plt.imshow(C_mean.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,8)
        plt.title('T mean')
        plt.axis('off')
        plt.imshow(T_mean.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,9)
        plt.title('RHM mean')
        plt.axis('off')
        plt.imshow(RHM_mean.cpu().numpy())
        plt.colorbar()
        
        plt.subplot(5,3,10)
        plt.title('C sum')
        plt.axis('off')
        plt.imshow(C_sum.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,11)
        plt.title('T sum')
        plt.axis('off')
        plt.imshow(T_sum.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,12)
        plt.title('RHM sum')
        plt.axis('off')
        plt.imshow(RHM_sum.cpu().numpy())
        plt.colorbar()

        plt.subplot(5,3,13)
        plt.title('C std')
        plt.axis('off')
        plt.imshow(C_std.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,14)
        plt.title('T std')
        plt.axis('off')
        plt.imshow(T_std.cpu().numpy())
        plt.colorbar()
        plt.subplot(5,3,15)
        plt.title('RHM std')
        plt.axis('off')
        plt.imshow(RHM_std.cpu().numpy())
        plt.colorbar()
        plt.savefig('/scratch/2022-fall-sp-jiguo/SCOT/visualization/self_similarity_statistics')


    def extract_hyperpixel(self, img, maptype, bbox, mask, backbone="resnet101"):
        r"""Given image, extract desired list of hyperpixels"""
        # tic = time.time()
        hyperfeats, rfsz, jsz, feat_map, fc = self.extract_intermediate_feat(img.unsqueeze(0), return_hp=True, backbone=backbone)

        ###feat_map_fix,fc_fix are generated to fix the CAM(The CAM should always come from SCOT with resnet50 pretrained on ImageNet in a supervised way as backbone)
        if backbone in ['resnet101','resnet101_densecl_IN']:
            _, _, _, feat_map_fix, fc_fix = self.extract_intermediate_feat(img.unsqueeze(0), return_hp=True, backbone='resnet101')
        elif backbone in ['resnet101_clip','resnet50_clip']:
            feat_map_fix, fc_fix = feat_map, fc
        else:
            _, _, _, feat_map_fix, fc_fix = self.extract_intermediate_feat(img.unsqueeze(0), return_hp=True, backbone='resnet50')
        # toc = time.time()
        # print(f'time for extracting intermediate features is {toc -tic}')
        # print('image size:{}, feature map size:{}'.format(img.size(),feat_map.size()))
        # print('max and min of hyperfeats are: {},{}'.format(torch.max(hyperfeats),torch.min(hyperfeats)))
        hpgeometry = geometry.receptive_fields(rfsz, jsz, hyperfeats.size()).to(self.device)
        
        # hyperfeats_orisize = copy.deepcopy(hyperfeats)
        hyperfeats_orisize = hyperfeats
        hyperfeats = hyperfeats.view(hyperfeats.size()[0], -1).t() ##heperfeats size: (3136,50,75)->(3136,3750)->(3750,3136)=(HW,C)

        # Prune boxes on margins (Otherwise may cause error)
        if self.benchmark in ['TSS']:
            hpgeometry, valid_ids = geometry.prune_margin(hpgeometry, img.size()[1:], 10)
            hyperfeats = hyperfeats[valid_ids, :]

        weights = torch.ones(len(hyperfeats),1).to(hyperfeats.device) ##Since hyperfeats.size() = (3750,3136), len(hyperfeats)=3750
        # print('--**--**weights size: {}'.format(weights.size()))
        if maptype in [1]: # weight points
            ##Mind that the given mask is None in our case since we don't specify args.cam as shown in evaluate_map_CAM.py
            if mask is None:
                # get CAM mask
                if backbone in ['fcn101']:
                    mask = self.get_FCN_map(img.unsqueeze(0), feat_map, fc, sz=(img.size(1),img.size(2)))
                else:
                    #adding an backbone attribute to determin the generation of CAM(using different fully connection layers)
                    mask = self.get_CAM_multi(img.unsqueeze(0), feat_map_fix, fc_fix, sz=(img.size(1),img.size(2)), top_k=2)
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
        #Because all the other hyperpixels will be resized into the same size with the first hyperpixel layer

        '''Add the clip backbone'''
        '''
        remember that the rfsz, jsz are fixed, we don't have to care about them
        featmap,fc are the output of the final cov layers and the fc layers respectively and they are only useful
        for constructing the CAM. Since we wanna fix our CAM with the resnet imagenet supervised trained one, we should
        fix the featmap and fc as well, which means when we use clip backbone, they should be generated from the 
        self.backbone1 instead of self.backbone
        '''

        clip_feats = []

        # tic_pass = time.time()
        if backbone in ['resnet50_clip','resnet101_clip']:
            # hook,layers = util.hook_model(self.backbone1)
            # tic_backbone1 = time.time()
            self.backbone1(img)
        #     print(f'time for passing the image {time.time()-tic_backbone1}')
        # print(f'time for passing the image {time.time()-tic_pass}')



        #For resnet part
        # Layer 0
        tic_res = time.time()
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        ## Try hooking clip
        # print(f'resnet conv time {time.time()-tic_res}')
        if 0 in self.hyperpixel_ids:
            feats.append(feat.clone())

            if backbone in ['resnet50_clip','resnet101_clip']:
                assert(list(self.layers.keys())[9]=='avgpool')
                tic_clip = time.time()
                clip_feat = self.hook(list(self.layers.keys())[9])
                # print(f'clip hook time {time.time()-tic_clip}')
                clip_feats.append(clip_feat)



        '''
        "Insert the forward propagation for clip backbone here"
        #layer 0 for clip resnet
        if backbone in ['resnet50_clip','resnet101_clip']:
            # clip_img = copy.deepcopy(img)
            # clip_img = clip_img.half()
            clip_feat = self.backbone1.conv1.forward(img)
            clip_feat = self.backbone1.bn1.forward(clip_feat)
            clip_feat = self.backbone1.relu1.forward(clip_feat)
            clip_feat = self.backbone1.conv2.forward(clip_feat)
            clip_feat = self.backbone1.bn2.forward(clip_feat)
            clip_feat = self.backbone1.relu2.forward(clip_feat)
            clip_feat = self.backbone1.conv3.forward(clip_feat)
            clip_feat = self.backbone1.bn3.forward(clip_feat)
            clip_feat = self.backbone1.relu3.forward(clip_feat)
            clip_feat = self.backbone1.avgpool.forward(clip_feat)

            if 0 in self.hyperpixel_ids:
                clip_feats.append(clip_feat.clone())
        '''
        




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


                #Try hooking clip
                if backbone in ['resnet50_clip','resnet101_clip']:
                    layer_name = 'layer'+str(lid)+'-'+str(bid)+'-bn3'
                    assert(layer_name in self.layers.keys())
                    clip_feat = self.hook(layer_name)
                    if bid == 0:
                        res_name = 'layer'+str(lid)+'-'+str(bid)+'-downsample'
                    else:
                        res_name = 'layer'+str(lid)+'-'+str(bid-1)
                    assert(res_name in self.layers.keys())
                    clip_res = self.hook(res_name)
                    clip_feat += clip_res
                    clip_feats.append(clip_feat)


            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

            #Implement CLIP resnet encoder the same way of implementing other backbones without using hook
            '''
            if backbone in ['resnet50_clip','resnet101_clip']:
                clip_res = clip_feat
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].conv1.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].bn1.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].relu1.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].conv2.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].bn2.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].relu2.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].avgpool.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].conv3.forward(clip_feat)
                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].bn3.forward(clip_feat)

                if bid == 0:
                    clip_res = self.backbone1.__getattr__('layer%d' % lid)[bid].downsample.forward(clip_res)
                
                clip_feat += clip_res

                if hid + 1 in self.hyperpixel_ids:
                    clip_feats.append(clip_feat.clone())

                clip_feat = self.backbone1.__getattr__('layer%d' % lid)[bid].relu3.forward(clip_feat)
            '''


        # GAP feature map
        feat_map = feat
        if backbone not in ['fcn101']:
            x = self.backbone.avgpool(feat)
            x = torch.flatten(x, 1)
            fc = self.backbone.fc(x)
        else:
            fc = None

        if not return_hp: # only return feat_map and fc
            return feat_map,fc

        
        ## feats.size() = (1 ,channels, h, w)  eg:(1,3136,50,75)
        ## So if we return feats[0], we will get an extracted feature map with size (3136, 50, 75) 
        # and this feature map is gotten by upsampling stacking the feature maps selected layers
        # In this way, we can make use of the multi-level representations
        if backbone in ['resnet50_clip','resnet101_clip']:
            for idx, clip_feat in enumerate(clip_feats):
                if idx == 0:
                    continue
                clip_feats[idx] = F.interpolate(clip_feat, tuple(clip_feats[0].size()[2:]), None, 'bilinear', True)
            clip_feats = torch.cat(clip_feats,dim=1)
            clip_feats = clip_feats.type(torch.float32)
            # print(clip_feats.dtype)

            # print('clip_feats size:{}'.format(clip_feats.size()))

            return clip_feats[0],rfsz, jsz, feat_map, fc
        else:
            # Up-sample & concatenate features to construct a hyperimage
            for idx, feat in enumerate(feats):
                if idx == 0:
                    continue
                feats[idx] = F.interpolate(feat, tuple(feats[0].size()[2:]), None, 'bilinear', True)
            feats = torch.cat(feats, dim=1)
            # print('****feats size:{}'.format(feats.size()))
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
        img_h,img_w = img.size(1),img.size(2)
        scales = [1.0,1.5,2.0]
        map_list = []
        for scale in scales:
            if scale>1.0:
                #in our case, there won't be scale*scale*sz[0]*sz[1] > 800*800
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
            # feat_map = self.backbone.get_final_fp(img)
            # print('FCN feature map size: {}'.format(feat_map.shape))
            
            predict = torch.max(feat_map, 1)[1] #[1] means get the indices of all the max values(from the torch.return_types.max)
            mask = predict-torch.min(predict)
            mask_map = mask / torch.max(mask)
            mask_map = F.interpolate(mask_map.unsqueeze(0).double(), (sz[0],sz[1]), None, 'bilinear', True)[0,0] # HxW
    
        return mask_map
