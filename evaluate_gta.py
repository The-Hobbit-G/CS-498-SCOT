r"""Runs Semantic Correspondence as an Optimal Transport Problem on GTA and Cityscape datasets"""

import argparse
import datetime
import os
import logging
import time

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from model import scot_CAM, geometry, evaluation, util
from data import dataset, download

import numpy as np


def run(datapath, benchmark, backbone, thres, alpha, hyperpixel, factorization,activation,normalization,k,
        logpath, args, beamsearch=False, model=None, dataloader=None):
    r"""Runs Semantic Correspondence as an Optimal Transport Problem"""

    # tic1 = time.time()
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
        # download.download_dataset(os.path.abspath(datapath), benchmark)
        #split = 'val' if beamsearch else 'test'
        split = args.split
        dset = download.load_dataset(benchmark, datapath, thres, device, split, args.cam)
        dataloader = DataLoader(dset, batch_size=1, num_workers=0)

    # for idx, data in enumerate(dataloader):
    #     print('idx:{}'.format(idx))
    #     print(data)
    #     if idx>1:
    #         break


    ''''''
    # 3. Model initialization
    if model is None:
        model = scot_CAM.SCOT_CAM(backbone, hyperpixel, benchmark, device, args.cam)
    else:
        model.hyperpixel_ids = util.parse_hyperpixel(hyperpixel)

    # 4. Evaluator initialization
    factorization_list = ['PCA','NMF','KMeans','No']
    assert(factorization in factorization_list)
    activation_list = ['ReLU','No']
    assert (activation in activation_list)
    normalization_list = ['Mutual','No','Single']
    assert(normalization in normalization_list)

    # print(f'load time is {time.time()-tic1}')


    evaluator = evaluation.Evaluator(benchmark, device)

    # zero_pcks = 0
    # srcpt_list = []
    # trgpt_list = []
    # time_list = []
    # PCK_list = []
    zero_acc = 0
    time_list = []
    # print(f'the length of dataloader is {len(dataloader)}')
    ##Now we need each pixel in the src img(gta) to be key points
    datalen = len(dataloader)
    for idx, data in enumerate(dataloader):
        gta_img,gta_ano,cs_img,cs_ano = data[0].to(device),data[1].to(device),data[2].to(device),data[3].to(device)
        assert(gta_ano.shape==cs_ano.shape)
        # print(gta_img.shape,cs_img.shape)
        #shape of gta_img = shape of cs_img = [1, 3, 196, 392], shape of gta_ano = shape of cs_ano = [1, 20, 196, 392]
        #gta_ano and cs_ano have 20 class labels including 1 background label
        # print(gta_img[0],cs_img[0])
        """
        In spair, data['bbox'] and data['kps'] are useless during evaluation and data['mask'] is None.
        So we don't have to care about any of them here (Could simply set them all to None)
        """
        tic = time.time()
        # print('src_kps_size:{}'.format(data['src_kps'].size()))

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            confidence_ts, src_box, trg_box = model(gta_img[0], cs_img[0], args.sim, args.exp1, args.exp2, args.eps, args.classmap, None, None, None, None, backbone, None, None,factorization,k,activation,normalization)
            # print(confidence_ts)
            conf, trg_indices = torch.max(confidence_ts, dim=1)
            unique, inv = torch.unique(trg_indices, sorted=False, return_inverse=True)
            # trgpt_list.append(len(unique))
            # srcpt_list.append(len(confidence_ts))
        if idx == 0:
            gta_shape = gta_img.shape
            cs_shape = cs_img.shape
            pixels = torch.zeros([2,gta_img.shape[2]*gta_img.shape[3]])
            x_ = np.arange(gta_img.shape[-1])
            x_cor = np.repeat(x_,gta_img.shape[-2])
            y_cor = torch.arange(gta_img.shape[-2]).repeat(gta_img.shape[-1])
            # print(pixels.shape,x_cor.shape,y_cor.shape)
            pixels[0,:] = torch.from_numpy(x_cor).to(torch.float32)
            pixels[1,:] = y_cor
            pixels = pixels.to(device)
            # print(pixels[:,500:520])
        assert(gta_img.shape==gta_shape==cs_img.shape==cs_shape)
            
        # c) Predict the correspondence of all pixels in src_img & evaluate performance
        prd_pixels = geometry.predict_kps(src_box, trg_box, pixels, confidence_ts) #get the correpondence pixel in cs of every pixel in gta
        prd_pixels[0] = torch.clamp(prd_pixels[0],min=0,max=cs_img.shape[-1]-1)
        prd_pixels[1] = torch.clamp(prd_pixels[1],min=0,max=cs_img.shape[-2]-1)
        
        # print(prd_pixels.shape,prd_pixels)
        gta_prd = torch.zeros_like(gta_ano)
        gta_prd[:,:,pixels[1].long(),pixels[0].long()] = cs_ano[:,:,prd_pixels[1].long(),prd_pixels[0].long()]
        toc = time.time()
        time_list.append(toc-tic)

        gta_classes = gta_ano[0].sum(-1).sum(-1)
        cs_classes = cs_ano[0].sum(-1).sum(-1)
        common_classes = gta_classes*cs_classes
        common_classes = torch.clamp(common_classes,max=1)
        # print(common_classes.shape)
        acc,inter,uni = evaluator.evaluate_acc(gta_prd,gta_ano,common_classes)

        if acc == 0:
            zero_acc += 1

        # d) Log results
        if not beamsearch:
            evaluator.log_acc(idx, datalen)
        
        
    
    #save_file = logfile.replace('logs/','')
    #np.save('PCK_{}.npy'.format(save_file), PCK_list)
    if beamsearch:
        return (sum(evaluator.eval_buf['true']) / sum(evaluator.eval_buf['common_area'])) * 100.
    else:
        # logging.info('source points:'+str(sum(srcpt_list)*1.0/len(srcpt_list)))
        # logging.info('target points:'+str(sum(trgpt_list)*1.0/len(trgpt_list)))
        logging.info('avg running time:'+str(sum(time_list)/len(time_list)))
        evaluator.log_acc(len(dset), datalen, average=True)
        logging.info('Total Number of 0.00 acc images:'+str(zero_acc))
    


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

    # Algorithm parameters
    parser.add_argument('--sim', type=str, default='OTGeo', help='Similarity type: OT, OTGeo, cos, cosGeo')
    parser.add_argument('--exp1', type=float, default=1.0, help='exponential factor on initial cosine cost')
    parser.add_argument('--exp2', type=float, default=1.0, help='exponential factor on final OT scores')
    parser.add_argument('--eps', type=float, default=0.05, help='epsilon for Sinkhorn Regularization')
    parser.add_argument('--classmap', type=int, default=1, help='class activation map: 0 for none, 1 for using CAM')
    parser.add_argument('--cam', type=str, default='', help='activation map folder, empty for end2end computation')
    parser.add_argument('--facorization',type=str,default='No',help='Choose the factorization methods you want to visualize, PCA,NMF,or Kmeans')
    parser.add_argument('--k',type=int,default=9, help='dimension of the factorized matrix M (X=L@M.T)')
    parser.add_argument('--activation',type=str,default='No', help='decide whether to apply a ReLU application to the hyperfeats')
    parser.add_argument('--normalization',type=str,default='No', help='decide whether to apply a mutual mean normalization to the hyperfeats')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    run(datapath=args.datapath, benchmark=args.dataset, backbone=args.backbone, thres=args.thres,
        alpha=args.alpha, hyperpixel=args.hyperpixel,factorization=args.facorization,activation=args.activation,normalization=args.normalization,k=args.k,logpath=args.logpath, args=args, beamsearch=False)

    util.log_args(args)
