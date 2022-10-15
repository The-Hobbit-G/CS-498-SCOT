"""Implementation of : Semantic Correspondence as an Optimal Transport Problem"""

from functools import reduce
from operator import add

import torch.nn.functional as F
import torch
import torchvision.utils
import gluoncvth as gcv
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

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
    #cv2.imwrite('/home/jianting/SCOT/visualization/'+title+'.jpg',img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


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
        scr_image_with_rps = show_from_cv(scr_image,src_kps.cpu().numpy().astype(int),'source_image')
        trg_image_with_rps = show_from_cv(trg_image,trg_kps.cpu().numpy().astype(int),'target_image')
        #########

        src_hyperpixels = self.extract_hyperpixel(args[0], maptype, src_bbox, src_mask, backbone)
        trg_hyperpixels = self.extract_hyperpixel(args[1], maptype, trg_bbox, trg_mask, backbone)
        src_featmap = src_hyperpixels[-1]
        trg_featmap = trg_hyperpixels[-1]
        print('----src,trg featmap size : {},{}'.format(src_featmap.size(),trg_featmap.size()))

        num_kps = src_kps_feat.size()[1]
        
        """Visualize cross-similarity C"""
        
        C_mat = torch.einsum('kij,kmn -> ijmn',(src_featmap,trg_featmap))/(torch.norm(src_featmap,2)*torch.norm(trg_featmap,2))
        print('C_mat size: {}'.format(C_mat.size()))
        

        '''
        C_mat_trg = C_mat.permute(2,3,0,1)
        plt.figure(2)
        for i in range(trg_kps_feat.size()[1]):
            plt.subplot(1,trg_kps_feat.size()[1],i+1)
            plt.imshow(C_mat_trg[int(trg_kps_feat[0][i]),int(trg_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/C_correspondence_trg')
        '''

        """Visualize cross-similarity OT matrix T and p(m|D) after RHM"""
        ##cross-similarity T for the source image
        confidence_ts, OT_mat, C_2dim = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter, args[2], args[3], args[4], args[5])
        print('**confidence_ts size: {}'.format(confidence_ts.size()))
        confidence_ts_orisize = confidence_ts.view_as(C_mat)
        OT_mat_orisize = OT_mat.view_as(C_mat)
        print('**confidence_ts original size:{}'.format(confidence_ts_orisize.size()))

        plt.figure(figsize=(3*num_kps,3*4))
        plt.subplot(4,num_kps,1)
        plt.title('source image')
        plt.imshow(scr_image_with_rps)
        plt.subplot(4,num_kps,2)
        plt.title('target image')
        plt.imshow(trg_image_with_rps)

        for i in range(num_kps):
            ##Correlation matrix C
            plt.subplot(4,num_kps,i+1+num_kps)
            plt.title('Correlation matrix')
            plt.imshow(C_mat[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
            ##OT matrix T
            plt.subplot(4,num_kps,i+1+2*num_kps)
            plt.title('OT matrix')
            plt.imshow(OT_mat_orisize[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy()) 
            #RHM confidence
            plt.subplot(4,num_kps,i+1+3*num_kps)
            plt.title('RHM')
            plt.imshow(confidence_ts_orisize[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/all_three_matrices_cross_matrix')


        
        '''
        plt.figure(1)
        for i in range(num_kps):
            plt.subplot(1,num_kps,i+1)
            plt.imshow(C_mat[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/C_correspondence_src')

        plt.figure(3)
        for i in range(num_kps):
            plt.subplot(1,num_kps,i+1)
            plt.imshow(confidence_ts_orisize[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/RHM_confidence_src')
        '''


        """Visualize self-similarity C"""
        ##self-similarity C for the source image
        C_self_src = torch.einsum('kij,kmn -> ijmn',(src_featmap,src_featmap))/(torch.norm(src_featmap,2)*torch.norm(src_featmap,2))
        

    
        """Visualize self-similarity OT matrix T and p(m|D) after RHM"""
        ##self-similarity T for the source image
        confidence_ts_selfsim, OT_mat_selfsim, C_2dim_selfsim = rhm_map.rhm(src_hyperpixels, src_hyperpixels, self.hsfilter, args[2], args[3], args[4], args[5])
        confidence_ts_selfsim_orisize = confidence_ts_selfsim.view_as(C_self_src)
        OT_mat_selfsim_orisize = OT_mat_selfsim.view_as(C_self_src)


        plt.figure(figsize=(3*num_kps,3*4))
        plt.subplot(4,num_kps,1)
        plt.title('source image')
        plt.imshow(scr_image_with_rps)
        # plt.subplot(4,num_kps,2)
        # plt.title('target image')
        # plt.imshow(trg_image_with_rps)

        for i in range(num_kps):
            ##Correlation matrix C
            plt.subplot(4,num_kps,i+1+num_kps)
            plt.title('Correlation matrix')
            plt.imshow(C_self_src[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
            ##OT matrix T
            plt.subplot(4,num_kps,i+1+2*num_kps)
            plt.title('OT matrix')
            plt.imshow(OT_mat_selfsim_orisize[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy()) 
            #RHM confidence
            plt.subplot(4,num_kps,i+1+3*num_kps)
            plt.title('RHM')
            plt.imshow(confidence_ts_selfsim_orisize[int(src_kps_feat[0][i]),int(src_kps_feat[1][i]),:,:].cpu().numpy())
        plt.savefig('/home/jianting/SCOT/visualization/all_three_matrices_self_similarity')


        
        """Visualize the entropy, mean, sum, std of self-similarity C(C_2dim_selfsim) and T(OT_mat_selfsim) over axis =1 so that [HW*HW] --> [HW,1]"""
        self.plot_selfsim_statistic(C_2dim_selfsim,OT_mat_selfsim,confidence_ts_selfsim,C_self_src[:,:,0,0],scr_image_with_rps)

        
        """Visualize the result of applying K-means, PCA, NNMF at src_hyperfeats"""
        
        # k_list = [3,7,20,35]
        '''
        # k_list = [200]
        k_list = [4,9,16,25,36]
        n_rows = [2,3,4,5,6]
        #visualize K-means results
        # src_hyperfeatures = src_hyperpixels[1]
        # print(src_hyperfeatures)
        # print(src_hyperfeatures[src_hyperfeatures<0])
       
        self.visualize_k_means(src_hyperpixels[1],C_self_src[:,:,0,0],k_list,n_rows)
        #visualize PCA results
        self.visualize_pca(src_hyperpixels[1],C_self_src[:,:,0,0],k_list,n_rows)
        #visualize NMF results
        self.visualize_nmf(src_hyperpixels[1],C_self_src[:,:,0,0],k_list,n_rows)

        
        """Visualize different sim with different backbone"""
        ##You can choose different backbones by changing the 'backbone' attribute and different similarity matrix by changing the 'sim' attribute
        sim = 'Correlation' #'OT','RHM'
        self.visualize_sim(args[0],args[1],maptype, src_bbox, trg_bbox, src_mask, trg_mask,args[3], args[4], args[5],backbone, sim, args[2])

        """Visualize G from fs = FG.T with different choices of factorization and different backbones"""
        ## bbox and mask have to be in line with img(src for args[0], trg for args[1])
        ## You can choose different backbones by changing the 'backbone' attribute and different factorization by changing fac
        fac = 'KMeans' # 'PCA','NMF'
        self.visualize_G(args[0], maptype, src_bbox, src_mask, k_list, backbone, fac)
        '''
        

        return confidence_ts, src_hyperpixels[0], trg_hyperpixels[0]

    def visualize_sim(self,src_img,trg_img,maptype,src_bbox,trg_bbox,src_mask,trg_mask,exp1,exp2,eps,backbone="resnet101",sim='Correlation',simi='OT'):
        """Given two images(src & trg), the backbone and the sim choice, visualize sim(backbone(src).T @ backbone(trg))"""
        sim_list = ['Correlation','OT','RHM']
        assert(sim in sim_list)
        source_image = self.detransform(src_img)
        target_image = self.detransform(trg_img)
        scr_image = tensor_to_np(source_image)
        trg_image = tensor_to_np(target_image)
        src_hyperpixels = self.extract_hyperpixel(src_img, maptype, src_bbox, src_mask, backbone)
        trg_hyperpixels = self.extract_hyperpixel(trg_img, maptype, trg_bbox, trg_mask, backbone)
        confidence_ts, OT_mat, C_2dim = rhm_map.rhm(src_hyperpixels, trg_hyperpixels, self.hsfilter, simi, exp1, exp2, eps)
        visual_dic = {'Correlation':C_2dim, 'OT':OT_mat, 'RHM':confidence_ts}
        plt.figure(figsize=(3*2,3*2))
        plt.subplot(2,2,1)
        plt.title('source image')
        plt.imshow(scr_image)
        plt.subplot(2,2,2)
        plt.title('target image')
        plt.imshow(trg_image)
        plt.subplot(2,2,3)
        plt.title(sim+'matrix')
        plt.imshow(visual_dic[sim])
        plt.savefig('/home/jianting/SCOT/visualization/'+backbone+''+sim+' matrix')

    def visualize_G(self, img, maptype, bbox, mask, k_list=[3,7,20,35], backbone="resnet101", f = 'KMeans'):
        f_list = ['KMeans','PCA','NMF']
        assert(f in f_list)
        hyperpixels = self.extract_hyperpixel(img, maptype, bbox, mask, backbone)
        if f == 'KMeans':
            self.visualize_k_means(hyperpixels[1],k_list)
        elif f == 'PCA':
            self.visualize_pca(hyperpixels[1],k_list)
        else:
            self.visualize_nmf(hyperpixels[1],k_list)


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

        print(C_entropy.max(),T_entropy.max(),RHM_entropy.max())
        print(C_entropy.min(),T_entropy.min(),RHM_entropy.min())

        # if activation == 'softmax':
        #     C_entropy = F.softmax(C_entropy,dim=0)
        #     T_entropy = F.softmax(T_entropy,dim=0)
        #     RHM_entropy = F.softmax(RHM_entropy,dim=0)
        # elif activation == 'ReLU':
        #     C_entropy = F.relu(C_entropy)
        #     T_entropy = F.relu(T_entropy)
        #     RHM_entropy = F.relu(RHM_entropy)


        print(C_entropy,T_entropy,RHM_entropy)
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

        print(T_mean, T_sum)

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
        plt.savefig('/home/jianting/SCOT/visualization/self_similarity_statistics')

    def visualize_k_means(self, hyperfeats, C_orisize, k_list,n_rows = [2,3,4,5,6]):
        num_k = len(k_list)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        for k in k_list:
            km = KMeans(n_clusters=k).fit(hyperfeats)
            # print(km.cluster_centers_,km.cluster_centers_.shape)
            # G = hyperfeats@np.linalg.pinv(km.cluster_centers_)
            # print(G)
            # G_list.append(G)
            GT = np.zeros((k,hyperfeats.shape[0])) #GT shape:k*HW
            assert(hyperfeats.shape[0]==km.labels_.shape[0])
            for label in range(km.labels_.shape[0]):
                GT[km.labels_[label],label]=1
            # print(GT)
            GT = torch.from_numpy(GT.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1]))
            GT_list.append(GT)
        assert(len(GT_list)==num_k)
        plt.figure()
        for i in range(num_k):
            img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
            torchvision.utils.save_image(img_grid,'/home/jianting/SCOT/visualization/G_mat_k_means_k={}.jpg'.format(k_list[i]))

    def visualize_pca(self, hyperfeats, C_orisize, k_list,n_rows = [1,2,4,6]):
        num_k = len(k_list)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        for k in k_list:
            pca = PCA(n_components=k)
            G = pca.fit_transform(hyperfeats) #G shape: HW*k
            GT = torch.from_numpy(G.T.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1]))
            GT_list.append(GT)
        assert(len(GT_list)==num_k)
        plt.figure()
        for i in range(num_k):
            img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
            torchvision.utils.save_image(img_grid,'/home/jianting/SCOT/visualization/G_mat_pca_k={}.jpg'.format(k_list[i]))

    def visualize_nmf(self, hyperfeats, C_orisize, k_list,n_rows = [1,2,4,6]):
        num_k = len(k_list)
        hyperfeats = F.relu(hyperfeats)
        hyperfeats = hyperfeats.cpu().numpy()
        C_orisize = C_orisize.cpu().numpy()
        GT_list = []
        for k in k_list:
            nmf = NMF(n_components=k)
            G = nmf.fit_transform(hyperfeats) #G shape: HW*k
            GT = torch.from_numpy(G.T.reshape(k,1,C_orisize.shape[0],C_orisize.shape[1]))
            GT_list.append(GT)
        assert(len(GT_list)==num_k)
        plt.figure()
        for i in range(num_k):
            img_grid = torchvision.utils.make_grid(GT_list[i],n_rows[i])
            torchvision.utils.save_image(img_grid,'/home/jianting/SCOT/visualization/G_mat_nmf_k={}.jpg'.format(k_list[i]))

        






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
