r"""Visualize different factorization methods(PCA,NMF,Kmeans) based on different backbones"""

import argparse
from audioop import cross
import datetime
import os
import logging
import time

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from model import scot_CAM, util
from data import dataset, download

import numpy as np


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel,
        logpath, visual_idx, factorization, choice, klist, args, beamsearch=False, model=None, dataloader=None):
    r"""Runs Semantic Correspondence as an Optimal Transport Problem"""

    # 1. Logging initialization
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not beamsearch:
        logfile = 'logs/{}_{}_{}_{}_exp{}-{}_e{}_m{}_{}_{}'.format(benchmark,backbone,args.split,args.sim,args.exp1,args.exp2,args.eps,args.classmap,args.cam,args.hyperpixel)
        print(logfile)
        util.init_logger(logfile)
        util.log_args(args)

    # 2. Evaluation benchmark initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dataloader is None:
        download.download_dataset(os.path.abspath(datapath), benchmark)
        #split = 'val' if beamsearch else 'test'
        split = args.split
        dset = download.load_dataset(benchmark, datapath, thres, device, split, args.cam)
        # dataloader = DataLoader(dset, batch_size=1, num_workers=0)
        data = dset.__getitem__(visual_idx)
    # print(data.keys())


    # 3. Model initialization
    if model is None:
        model = scot_CAM.SCOT_CAM(backbone, hyperpixel, benchmark, device, args.cam)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)

    threshold = 0.0

    # print(data['src_img'].shape,data['trg_img'].shape)
    data['src_img']=data['src_img'].unsqueeze(0)
    data['trg_img']=data['trg_img'].unsqueeze(0)
    # print(data['src_kps'].shape)

    data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'])
    data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'])
    src_size = data['src_img'].size()
    trg_size = data['trg_img'].size()

    if len(args.cam)>0:
        data['src_mask'] = util.resize_mask(data['src_mask'],src_size)
        data['trg_mask'] = util.resize_mask(data['trg_mask'],trg_size)
        data['src_bbox'] = util.get_bbox_mask(data['src_mask'], thres=threshold).to(device)
        data['trg_bbox'] = util.get_bbox_mask(data['trg_mask'], thres=threshold).to(device)
    else:
        data['src_mask'] = None
        data['trg_mask'] = None

    data['alpha'] = alpha

    savepath = '/home/jianting/SCOT/visualization/Gmat_example/'
    factorization_list = ['PCA' ,'NMF','KMeans','All']
    assert(factorization in factorization_list)
    choice_list = ['src','trg']
    assert(choice in choice_list)
    klist = util.parse_hyperpixel(klist)
    # k_list = [4,9,16,25,36]
    with torch.no_grad():
        model.visualize_G(data, visual_idx, args.classmap, savepath, klist, backbone, factorization, choice)
        # conf, trg_indices = torch.max(confidence_ts, dim=1)
        # unique, inv = torch.unique(trg_indices, sorted=False, return_inverse=True)
        # trgpt_list.append(len(unique))
        # srcpt_list.append(len(confidence_ts))



if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description='SCOT in pytorch')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id')
    parser.add_argument('--datapath', type=str, default='./Datasets_SCOT')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--backbone', type=str, default='resnet101')
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--hyperpixel', type=str, default='')
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--split', type=str, default='test', help='trn,val.test')
    parser.add_argument('--visual_idx',type=int,default=0, help='index of the image pair you want to visualize')
    parser.add_argument('--facorization',type=str,default='All',help='Choose the factorization methods you want to visualize, PCA,NMF,or Kmeans')
    parser.add_argument('--choice',type=str,default='src', help='Choice for visualizing cross or self similarity')
    parser.add_argument('--klist',type=str,default='(4,9,16,25,36)', help='dimension of the factorized matrix M (X=L@M.T)')

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres,
        alpha=args.alpha, hyperpixel=args.hyperpixel, logpath=args.logpath, visual_idx=args.visual_idx, 
        factorization=args.facorization, choice=args.choice, klist=args.klist, args=args, beamsearch=False)

    util.log_args(args)