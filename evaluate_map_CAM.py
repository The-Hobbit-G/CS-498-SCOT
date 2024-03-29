r"""Runs Semantic Correspondence as an Optimal Transport Problem"""

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

    zero_pcks = 0
    srcpt_list = []
    trgpt_list = []
    time_list = []
    PCK_list = []
    # print(f'the length of dataloader is {len(dataloader)}')
    for idx, data in enumerate(dataloader):
        threshold = 0.0
        # print('idx:{}'.format(idx))
        # if idx>10:
        #     break
        
        # a) Retrieve images and adjust their sizes to avoid large numbers of hyperpixels(The coordinates of kps will also be adjusted with the same ratio)
        data['src_img'], data['src_kps'], data['src_intratio'] = util.resize(data['src_img'], data['src_kps'][0])
        data['trg_img'], data['trg_kps'], data['trg_intratio'] = util.resize(data['trg_img'], data['trg_kps'][0])
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
        tic = time.time()
        # print('src_kps_size:{}'.format(data['src_kps'].size()))
        ##src_kps and trg_kps are stored in a width,height fashion like [[w1,w2,w3],[h1,h2,h3]]

        # b) Feed a pair of images to Hyperpixel Flow model
        with torch.no_grad():
            confidence_ts, src_box, trg_box = model(data['src_img'], data['trg_img'], args.sim, args.exp1, args.exp2, args.eps, args.classmap, data['src_bbox'], data['trg_bbox'], data['src_mask'], data['trg_mask'], backbone, data['src_kps'], data['trg_kps'],factorization,k,activation,normalization)
            # print(confidence_ts)
            conf, trg_indices = torch.max(confidence_ts, dim=1)
            #conf--the conf score of the pixel in trg img that has the highest matching conf score with a perticular pixel in the hyper-feats of scr img
            #trg_indices -- the index of that perticular hyperpixel
            unique, inv = torch.unique(trg_indices, sorted=False, return_inverse=True)
            trgpt_list.append(len(unique))# we only count the unique matching pixel(which means we only accept one-to-one matching)
            srcpt_list.append(len(confidence_ts))

        # c) Predict key-points & evaluate performance
        prd_kps = geometry.predict_kps(src_box, trg_box, data['src_kps'], confidence_ts)
        toc = time.time()
        # print(toc-tic)
        time_list.append(toc-tic)
        pair_pck = evaluator.evaluate(prd_kps, data)
        PCK_list.append(pair_pck)
        # print(f'evaluate time: {time.time()-toc}')
        if pair_pck==0:
            zero_pcks += 1

        # d) Log results
        if not beamsearch:
            evaluator.log_result(idx, data=data)
    
    #save_file = logfile.replace('logs/','')
    #np.save('PCK_{}.npy'.format(save_file), PCK_list)
    if beamsearch:
        return (sum(evaluator.eval_buf['pck']) / len(evaluator.eval_buf['pck'])) * 100.
    else:
        logging.info('source points:'+str(sum(srcpt_list)*1.0/len(srcpt_list)))
        logging.info('target points:'+str(sum(trgpt_list)*1.0/len(trgpt_list)))
        logging.info('avg running time:'+str(sum(time_list)/len(time_list)))
        evaluator.log_result(len(dset), data=None, average=True)
        logging.info('Total Number of 0.00 pck images:'+str(zero_pcks))
    


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
