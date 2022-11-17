"""Implementation of optimal transport+geometric post-processing (Hough voting)"""

import math
import copy
import cv2
from matplotlib import pyplot as plt
import numpy as np

import torch.nn.functional as F
import torch

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment

from kmeans_pytorch import kmeans

from . import geometry


def perform_sinkhorn(C,epsilon,mu,nu,a=[],warm=False,niter=1,tol=10e-9):
    """Main Sinkhorn Algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        a = a.cuda()

    K = torch.exp(-C/epsilon)

    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%2==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)
            if i>0 and (Err[i,0]) < tol:
                break

        a = mu / torch.mm(K, b)

        if i%2==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)
            if i>0 and (Err[i,1]) < tol:
                break

        PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    del a; del b; del K
    return PI,mu,nu,Err


def appearance_similarity(src_feats, trg_feats, exp1=3):
    r"""Semantic appearance similarity (exponentiated cosine)"""
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)
    sim = torch.matmul(src_feats, trg_feats.t()) / \
          torch.matmul(src_feat_norms, trg_feat_norms)
    sim = torch.pow(torch.clamp(sim, min=0), exp1)

    return sim

def pca_(src_feats,trg_feats,src_feat_norms,trg_feat_norms,k):

    '''pca+hungarian matching with L=XG'''
    '''
    pca_src = PCA(n_components=k)
    pca_trg = PCA(n_components=k)
    pca_src.fit(src_feats)
    pca_trg.fit(trg_feats)

    GT_src = pca_src.components_ #k*C
    L_src = src_feats@GT_src.T #(HW*C)@(C*k)-->HW*k
    GT_trg = pca_trg.components_ #k*c
    L_trg = trg_feats@GT_trg.T

    # print(GT_src.shape,L_src.shape,GT_trg.shape,L_trg.shape)
    print(GT_src@GT_src.T)
    # L_src_norm = np.linalg.norm(L_src, ord=2, axis=0)
    # L_trg_norm = np.linalg.norm(L_trg, ord=2, axis=0)
    # L_src_norm = np.expand_dims(L_src_norm,axis = 1)
    # L_trg_norm = np.expand_dims(L_trg_norm,axis = 0)
    GT_src_norm = np.linalg.norm(GT_src, ord=2, axis=1)
    GT_trg_norm = np.linalg.norm(GT_trg, ord=2, axis=1)
    GT_src_norm = np.expand_dims(GT_src_norm,axis = 1)
    GT_trg_norm = np.expand_dims(GT_trg_norm,axis = 0)

    G_corr = (GT_src@GT_trg.T)/(GT_src_norm@GT_trg_norm)
    print(np.max(G_corr),np.min(G_corr))

    row_ind,col_ind = linear_sum_assignment(1-G_corr)
    assert(row_ind.shape[0]==col_ind.shape[0]==L_src.shape[1]==L_trg.shape[1])
    L_trg = L_trg[:,col_ind]
    G_corr_new = np.zeros((row_ind.shape[0],col_ind.shape[0]))
    G_corr_new[row_ind,col_ind] = 1.0

    sim = torch.from_numpy(L_src).to(torch.float32).cuda()@torch.from_numpy(G_corr_new).to(torch.float32).cuda()@\
        torch.from_numpy(L_trg.T).to(torch.float32).cuda()/torch.matmul(src_feat_norms, trg_feat_norms)

    '''

    ##svd 
    U_src,S_src,Vh_src = torch.linalg.svd(src_feats.T) # src_feats:HW*C --> src_feats.T:C*HW
    L_src = U_src[:,0:k]@torch.diag(S_src[0:k])
    G_src = Vh_src[0:k,:].T
    U_trg,S_trg,Vh_trg = torch.linalg.svd(trg_feats.T) # src_feats:HW*C --> src_feats.T:C*HW
    L_trg = U_trg[:,0:k]@torch.diag(S_trg[0:k])
    G_trg = Vh_trg[0:k,:].T


    #Compute norm for hungarian matching
    # L_src_norm = torch.norm(L_src, p=2, dim=0).unsqueeze(1)
    # L_trg_norm = torch.norm(L_trg, p=2, dim=0).unsqueeze(0)


    # G_src = pca_src.fit_transform(src_feats) ##HW*k
    # L_src = pca_src.components_.T ##C*k
    # G_trg = pca_trg.fit_transform(trg_feats)  ##HW*k
    # L_trg = pca_trg.components_.T ##C*k

    # L_src_norm = np.linalg.norm(L_src, ord=2, axis=0)
    # L_trg_norm = np.linalg.norm(L_trg, ord=2, axis=0)
    # L_src_norm = np.expand_dims(L_src_norm,axis = 1)
    # L_trg_norm = np.expand_dims(L_trg_norm,axis = 0)

    # L_corr = (L_src.T@L_trg)/(L_src_norm@L_trg_norm)
    # L_corr = L_corr.cpu().numpy()

    ##construct cost matrix for Hungarian matching
    # print(np.max(1-L_corr),np.min(1-L_corr))
    # L_corr = np.power(np.maximum(L_corr,0),1.0)

    # row_ind,col_ind = linear_sum_assignment(1-L_corr)
    # assert(row_ind.shape[0]==col_ind.shape[0]==G_src.shape[1]==G_trg.shape[1])
    # G_trg = G_trg[:,col_ind]
    # L_corr_new = np.zeros((row_ind.shape[0],col_ind.shape[0]))
    # L_corr_new[row_ind,col_ind] = 1.0


    #home for svd and hungarian matching with L = U@torch.diag(S) and G = Vh.T
    # sim = G_src@torch.from_numpy(L_corr_new).to(torch.float32).cuda()@G_trg.T/torch.matmul(src_feat_norms, trg_feat_norms)

    # print(G_src.dtype,L_corr_new.dtype,G_trg.dtype)
    # print(torch.from_numpy(G_src).dtype,torch.from_numpy(L_corr_new).dtype,torch.from_numpy(G_trg).dtype,src_feat_norms.dtype,trg_feat_norms.dtype)

    ##sim for pca+hungarian matching
    # sim = torch.from_numpy(G_src).to(torch.float32).cuda()@torch.from_numpy(L_corr_new).to(torch.float32).cuda()@\
    #     torch.from_numpy(G_trg.T).to(torch.float32).cuda()/torch.matmul(src_feat_norms, trg_feat_norms)
    




    '''To do: try reconstruction mathods later'''
    src_feats_recon = L_src@G_src.T
    trg_feats_recon = L_trg@G_trg.T
    sim = sim = torch.matmul(src_feats_recon.t(), trg_feats_recon) / \
            torch.matmul(src_feat_norms, trg_feat_norms)
    
    
    return sim

def kmeans_(src_feats_np,trg_feats_np,src_feat_norms,trg_feat_norms,k):
    km_src = KMeans(n_clusters=k,max_iter=500).fit(src_feats_np)
    km_trg = KMeans(n_clusters=k,max_iter=500).fit(trg_feats_np)

    HW_src = np.arange(km_src.labels_.shape[0])
    HW_trg = np.arange(km_trg.labels_.shape[0])

    G_src = np.zeros((km_src.labels_.shape[0],k)) ##HW*k
    G_src[HW_src,km_src.labels_[HW_src]] = 1.0
    L_src = km_src.cluster_centers_.T ##C*k
    G_trg = np.zeros((km_trg.labels_.shape[0],k))##HW*k
    G_trg[HW_trg,km_trg.labels_[HW_trg]] = 1.0
    L_trg = km_trg.cluster_centers_.T ##C*k

    src_feats_recon = torch.from_numpy(G_src).to(torch.float32).cuda()@torch.from_numpy(L_src.T).to(torch.float32).cuda()
    trg_feats_recon = torch.from_numpy(G_trg).to(torch.float32).cuda()@torch.from_numpy(L_trg.T).to(torch.float32).cuda()

    sim = sim = torch.matmul(src_feats_recon, trg_feats_recon.t()) / \
            torch.matmul(src_feat_norms, trg_feat_norms)

    '''
    L_src_norm = np.linalg.norm(L_src, ord=2, axis=0)
    L_trg_norm = np.linalg.norm(L_trg, ord=2, axis=0)
    L_src_norm = np.expand_dims(L_src_norm,axis = 1)
    L_trg_norm = np.expand_dims(L_trg_norm,axis = 0)

    L_corr = (L_src.T@L_trg)/(L_src_norm@L_trg_norm)
    # L_corr = np.power(np.maximum(L_corr,0),1.0)

    row_ind,col_ind = linear_sum_assignment(1-L_corr)
    assert(row_ind.shape[0]==col_ind.shape[0]==G_src.shape[1]==G_trg.shape[1])
    G_trg = G_trg[:,col_ind]
    L_corr_new = np.zeros((row_ind.shape[0],col_ind.shape[0]))
    L_corr_new[row_ind,col_ind] = 1.0


    sim = torch.from_numpy(G_src).to(torch.float32).cuda()@torch.from_numpy(L_corr_new).to(torch.float32).cuda()@\
        torch.from_numpy(G_trg.T).to(torch.float32).cuda()/torch.matmul(src_feat_norms, trg_feat_norms)
    '''
    return sim

def nmf_(src_feats_np,trg_feats_np,src_feat_norms,trg_feat_norms,k):
    nmf_src = NMF(n_components=k,max_iter=500)
    nmf_trg = NMF(n_components=k,max_iter=500)

    G_src = nmf_src.fit_transform(src_feats_np)  ##HW*k
    L_src = nmf_src.components_.T ##C*k
    G_trg = nmf_trg.fit_transform(trg_feats_np)  ##HW*k
    L_trg = nmf_trg.components_.T ##C*k

    src_feats_recon = torch.from_numpy(G_src).to(torch.float32).cuda()@torch.from_numpy(L_src.T).to(torch.float32).cuda()
    trg_feats_recon = torch.from_numpy(G_trg).to(torch.float32).cuda()@torch.from_numpy(L_trg.T).to(torch.float32).cuda()
    sim = sim = torch.matmul(src_feats_recon, trg_feats_recon.t()) / \
            torch.matmul(src_feat_norms, trg_feat_norms)


    '''
    L_src_norm = np.linalg.norm(L_src, ord=2, axis=0)
    L_trg_norm = np.linalg.norm(L_trg, ord=2, axis=0)
    L_src_norm = np.expand_dims(L_src_norm,axis = 1)
    L_trg_norm = np.expand_dims(L_trg_norm,axis = 0)

    L_corr = (L_src.T@L_trg)/(L_src_norm@L_trg_norm)
    # L_corr = np.power(np.maximum(L_corr,0),1.0)

    row_ind,col_ind = linear_sum_assignment(1-L_corr)
    
    assert(row_ind.shape[0]==col_ind.shape[0]==G_src.shape[1]==G_trg.shape[1])
    G_trg = G_trg[:,col_ind]
    L_corr_new = np.zeros((row_ind.shape[0],col_ind.shape[0]))
    L_corr_new[row_ind,col_ind] = 1.0

    sim = torch.from_numpy(G_src).to(torch.float32).cuda()@torch.from_numpy(L_corr_new).to(torch.float32).cuda()@\
        torch.from_numpy(G_trg.T).to(torch.float32).cuda()/torch.matmul(src_feat_norms, trg_feat_norms)
    '''
    return sim



def appearance_similarityOT(src_feats, trg_feats, k, factorization, exp1=1.0, exp2=1.0, eps=0.05, src_weights=None, trg_weights=None,activation = 'No',normalization='No'):
    r"""Semantic Appearance Similarity"""
    #st_weights = src_weights.mm(trg_weights.t())
    # print('size of fs:{}'.format(src_feats.size()))
    # print('size of ft:{}'.format(trg_feats.size()))
    if activation == 'ReLU':
        src_feats = F.relu(src_feats)
        trg_feats = F.relu(trg_feats)

    if normalization == 'Mutual':
        src_mean = torch.mean(src_feats,dim=0)
        trg_mean = torch.mean(trg_feats,dim=0)
        multual_mean = (src_mean*src_feats.shape[0]+trg_mean*trg_feats.shape[0])/(src_feats.shape[0]+trg_feats.shape[0])
        src_feats = src_feats-multual_mean
        trg_feats = trg_feats-multual_mean
    elif normalization == 'Single':
        src_mean = torch.mean(src_feats,dim=0)
        trg_mean = torch.mean(trg_feats,dim=0)
        src_feats = src_feats-src_mean
        trg_feats = trg_feats-trg_mean
    

    ##The norm should be computed along channel(C). So each hyper-pixel should be normalized with a specific norm
    src_feat_norms = torch.norm(src_feats, p=2, dim=1).unsqueeze(1)
    trg_feat_norms = torch.norm(trg_feats, p=2, dim=1).unsqueeze(0)

    src_feats_np = src_feats.cpu().numpy()
    trg_feats_np = trg_feats.cpu().numpy()

    if factorization == 'PCA':
        sim = pca_(src_feats,trg_feats,src_feat_norms,trg_feat_norms,k)
    elif factorization == 'KMeans':
        sim = kmeans_(src_feats_np,trg_feats_np,src_feat_norms,trg_feat_norms,k)
    elif factorization == 'No':
        sim = torch.matmul(src_feats, trg_feats.t()) / \
            torch.matmul(src_feat_norms, trg_feat_norms)
    elif factorization == 'NMF' and activation == 'ReLU':
        sim = nmf_(src_feats_np,trg_feats_np,src_feat_norms,trg_feat_norms,k)
    else:
        raise Exception('Factorization not feasible')
        
    
    sim = torch.pow(torch.clamp(sim, min=0), 1.0)
    # print('size of C: {}'.format(sim.size()))
    #sim = sim*st_weights
    '''
    Visualize the similarity matrix
    '''
    cost = 1-sim

    n1 = len(src_feats)
    mu = (torch.ones((n1,))/n1).cuda()
    mu = src_weights / src_weights.sum()
    n2 = len(trg_feats)
    nu = (torch.ones((n2,))/n2).cuda()
    nu = trg_weights / trg_weights.sum()
    ## ---- <Run Optimal Transport Algorithm> ----
    #mu = mu.unsqueeze(1)
    #nu = nu.unsqueeze(1)
    with torch.no_grad():
        epsilon = eps
        cnt = 0
        while True:
            PI,a,b,err = perform_sinkhorn(cost, epsilon, mu, nu)
            #PI = sinkhorn_stabilized(mu, nu, cost, reg=epsilon, numItermax=50, method='sinkhorn_stabilized', cuda=True)
            if not torch.isnan(PI).any():
                if cnt>0:
                    print(cnt)
                break
            else: # Nan encountered caused by overflow issue is sinkhorn
                epsilon *= 2.0
                #print(epsilon)
                cnt += 1

    PI = n1*PI # re-scale PI 
    #exp2 = 1.0 for spair-71k, TSS
    #exp2 = 0.5 # for pf-pascal and pfwillow
    PI = torch.pow(torch.clamp(PI, min=0), exp2)
    # print('size of T: {}'.format(PI.size()))

    # C_orisize = sim.view()

    #return OT and C
    return PI,sim



def hspace_bin_ids(src_imsize, src_box, trg_box, hs_cellsize, nbins_x):
    r"""Compute Hough space bin id for the subsequent voting procedure"""
    src_ptref = torch.tensor(src_imsize, dtype=torch.float).to(src_box.device)
    src_trans = geometry.center(src_box)
    trg_trans = geometry.center(trg_box)
    xy_vote = (src_ptref.unsqueeze(0).expand_as(src_trans) - src_trans).unsqueeze(2).\
                  repeat(1, 1, len(trg_box)) + \
              trg_trans.t().unsqueeze(0).repeat(len(src_box), 1, 1)

    bin_ids = (xy_vote / hs_cellsize).long()

    return bin_ids[:, 0, :] + bin_ids[:, 1, :] * nbins_x


def build_hspace(src_imsize, trg_imsize, ncells):
    r"""Build Hough space where voting is done"""
    hs_width = src_imsize[0] + trg_imsize[0]
    hs_height = src_imsize[1] + trg_imsize[1]
    hs_cellsize = math.sqrt((hs_width * hs_height) / ncells)
    nbins_x = int(hs_width / hs_cellsize) + 1
    nbins_y = int(hs_height / hs_cellsize) + 1

    return nbins_x, nbins_y, hs_cellsize


def rhm(src_hyperpixels, trg_hyperpixels, hsfilter, sim, exp1, exp2, eps,k=4,factorization='No',activation='No',normalization='No',ncells=8192):
    r"""Regularized Hough matching"""
    # Unpack hyperpixels
    # src_hpgeomt, src_hpfeats, src_imsize, src_weights = src_hyperpixels
    # trg_hpgeomt, trg_hpfeats, trg_imsize, trg_weights = trg_hyperpixels

    src_hpgeomt, src_hpfeats, src_imsize, src_weights, src_hpfeats_orisize = src_hyperpixels
    trg_hpgeomt, trg_hpfeats, trg_imsize, trg_weights, trg_hpfeats_orisize = trg_hyperpixels

    # Prepare for the voting procedure
    if sim in ['cos', 'cosGeo']:
        votes = appearance_similarity(src_hpfeats, trg_hpfeats, exp1)
    if sim in ['OT', 'OTGeo']:
        # votes = appearance_similarityOT(src_hpfeats, trg_hpfeats, exp1, exp2, eps, src_weights, trg_weights)
        votes,sim_mat_C = appearance_similarityOT(src_hpfeats, trg_hpfeats, k, factorization, exp1, exp2, eps, src_weights, trg_weights, activation,normalization)
    if sim in ['OT', 'cos', 'cos2']:
        return votes


    nbins_x, nbins_y, hs_cellsize = build_hspace(src_imsize, trg_imsize, ncells)
    bin_ids = hspace_bin_ids(src_imsize, src_hpgeomt, trg_hpgeomt, hs_cellsize, nbins_x)
    hspace = src_hpgeomt.new_zeros((len(votes), nbins_y * nbins_x))

    # Proceed voting
    hbin_ids = bin_ids.add(torch.arange(0, len(votes)).to(src_hpgeomt.device).
                           mul(hspace.size(1)).unsqueeze(1).expand_as(bin_ids))
    hspace = hspace.view(-1).index_add(0, hbin_ids.view(-1), votes.view(-1)).view_as(hspace)
    hspace = torch.sum(hspace, dim=0)

    # Aggregate the voting results
    hspace = F.conv2d(hspace.view(1, 1, nbins_y, nbins_x),
                      hsfilter.unsqueeze(0).unsqueeze(0), padding=3).view(-1)

    return votes * torch.index_select(hspace, dim=0, index=bin_ids.view(-1)).view_as(votes), votes, sim_mat_C


