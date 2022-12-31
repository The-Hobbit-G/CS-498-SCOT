"""Provides memory buffer and logger for evaluation"""

import logging

# from skimage import draw
import numpy as np
import torch


class Evaluator:
    r"""To evaluate and log evaluation metrics: PCK, LT-ACC, IoU"""
    def __init__(self, benchmark, device):
        r"""Constructor for Evaluator"""
        self.eval_buf = {
            'pfwillow': {'pck': [], 'cls_pck': dict()},
            'pfpascal': {'pck': [], 'cls_pck': dict()},
            'spair':    {'pck': [], 'cls_pck': dict()},
            'gta' : {'acc': [],'true':[], 'common_area':[], 'acc_cs':[], 'true_cs':[], 'common_area_cs':[]}
        }

        self.eval_funct = {
            'pfwillow': self.eval_pck,
            'pfpascal': self.eval_pck,
            'spair': self.eval_pck,
            'gta': self.eval_acc
        }

        self.log_funct = {
            'pfwillow': self.log_pck,
            'pfpascal': self.log_pck,
            'spair': self.log_pck,
            'gta': self.log_acc
        }

        self.eval_buf = self.eval_buf[benchmark]
        self.eval_funct = self.eval_funct[benchmark]
        self.log_funct = self.log_funct[benchmark]
        self.benchmark = benchmark
        self.device = device

    def evaluate(self, prd_kps, data):
        r"""Compute desired evaluation metric"""
        return self.eval_funct(prd_kps, data)
    
    def evaluate_acc(self,pred, gt, common_classes):
        r""""Compute the accuracy for gta and cityscape dataset """
        return self.eval_funct(pred, gt, common_classes)

    def log_result(self, idx, data, average=False):
        r"""Print results: PCK, or LT-ACC & IoU """
        return self.log_funct(idx, data, average)

    def eval_pck(self, prd_kps, data):
        r"""Compute percentage of correct key-points (PCK) based on prediction"""
        pckthres = data['pckthres'][0] * data['trg_intratio']
        ncorrt = correct_kps(data['trg_kps'].cuda(), prd_kps, pckthres, data['alpha'])
        pair_pck = int(ncorrt) / int(data['trg_kps'].size(1))

        self.eval_buf['pck'].append(pair_pck)

        if self.eval_buf['cls_pck'].get(data['pair_class'][0]) is None:
            self.eval_buf['cls_pck'][data['pair_class'][0]] = []
        self.eval_buf['cls_pck'][data['pair_class'][0]].append(pair_pck)

        return pair_pck

    def log_pck(self, idx, data, average):
        r"""Log percentage of correct key-points (PCK)"""
        if average:
            pck = sum(self.eval_buf['pck']) / len(self.eval_buf['pck'])
            for cls in self.eval_buf['cls_pck']:
                cls_avg = sum(self.eval_buf['cls_pck'][cls]) / len(self.eval_buf['cls_pck'][cls])
                logging.info('%15s: %3.3f' % (cls, cls_avg))
            logging.info(' * Average: %3.3f' % pck)

            return pck

        logging.info('[%5d/%5d]: \t [Pair PCK: %3.3f]\t[Average: %3.3f] %s' %
                     (idx + 1,
                      data['datalen'],
                      self.eval_buf['pck'][idx],
                      sum(self.eval_buf['pck']) / len(self.eval_buf['pck']),
                      data['pair_class'][0]))
        return None

    def eval_acc(self, pred, gt, common_classes):
        # mask = get_mask(pred)
        mask = pred
        fg = 1 - gt[:,-1:,:,:]#get the foreground
        mask = mask * fg

        gt_common_classes = torch.mul(gt,common_classes.view(1,common_classes.shape[0],1,1))
        fg_common_classes = gt_common_classes[:,:-1,:,:] * fg

        # print(fg.shape, gt_common_classes.shape, fg_common_classes.shape)
        
        c = gt.shape[1] #num of classes (including the background channel)
        n = fg.sum(-1).sum(-1)[0] #num of foreground pixels

        normalized_n = fg_common_classes.sum(-1).sum(-1).sum(-1)

        # print(n[0].item(),normalized_n[0].item())
        
        true = ((mask == gt).sum(1, keepdims=True)==c).sum(-1).sum(-1)[0]
        inter = ((mask + gt)>=1.9).sum(-1).sum(-1)
        uni = ((mask + gt)>=0.9).sum(-1).sum(-1)
        acc = true.float() / n

        # print(true[0].item(),acc[0].item())

        # normalized_acc = true.float() / normalized_n

        self.eval_buf['acc'].append(acc.item())
        # self.eval_buf['normalized_acc'].append(normalized_acc)
        self.eval_buf['true'].append(true.item())
        self.eval_buf['common_area'].append(normalized_n.item())

        return acc, inter, uni


    def eval_acc_cs(self, pred, gt, common_classes):
        # mask = get_mask(pred)
        mask = pred
        fg = 1 - gt[:,-1:,:,:]#get the foreground
        mask = mask * fg

        gt_common_classes = torch.mul(gt,common_classes.view(1,common_classes.shape[0],1,1))
        fg_common_classes = gt_common_classes[:,:-1,:,:] * fg

        # print(fg.shape, gt_common_classes.shape, fg_common_classes.shape)
        
        c = gt.shape[1] #num of classes (including the background channel)
        n = fg.sum(-1).sum(-1)[0] #num of foreground pixels

        normalized_n = fg_common_classes.sum(-1).sum(-1).sum(-1)

        # print(n[0].item(),normalized_n[0].item())
        
        true = ((mask == gt).sum(1, keepdims=True)==c).sum(-1).sum(-1)[0]
        inter = ((mask + gt)>=1.9).sum(-1).sum(-1)
        uni = ((mask + gt)>=0.9).sum(-1).sum(-1)
        acc = true.float() / n

        # print(true[0].item(),acc[0].item())

        # normalized_acc = true.float() / normalized_n

        self.eval_buf['acc_cs'].append(acc.item())
        # self.eval_buf['normalized_acc'].append(normalized_acc)
        self.eval_buf['true_cs'].append(true.item())
        self.eval_buf['common_area_cs'].append(normalized_n.item())

        return acc, inter, uni

    def log_acc(self,idx,datalen,average=False):
        if average:
            avg_acc = sum(self.eval_buf['acc']) / len(self.eval_buf['acc'])
            normalized_avg_acc = sum(self.eval_buf['true'])/sum(self.eval_buf['common_area'])
            avg_acc_cs = sum(self.eval_buf['acc_cs']) / len(self.eval_buf['acc_cs'])
            normalized_avg_acc_cs = sum(self.eval_buf['true_cs'])/sum(self.eval_buf['common_area_cs'])
            logging.info(' * Average Acc(gta): %3.3f' % avg_acc)
            logging.info(' * Normalized Average Acc(gta): %3.3f' % normalized_avg_acc)
            logging.info(' * Average Acc(cs): %3.3f' % avg_acc_cs)
            logging.info(' * Normalized Average Acc(cs): %3.3f' % normalized_avg_acc_cs)

            return avg_acc, normalized_avg_acc


        logging.info('[%5d/%5d]: \t [Acc: %3.3f]\t[Average Acc(gta): %3.3f] \t[Normalized Average Acc(gta): %3.3f]\
            \t[Average Acc(cs): %3.3f] \t[Normalized Average Acc(cs): %3.3f]' %
                     (idx + 1,
                      datalen,
                      self.eval_buf['acc'][idx],
                      sum(self.eval_buf['acc']) / len(self.eval_buf['acc']),
                      sum(self.eval_buf['true'])/sum(self.eval_buf['common_area']),
                      sum(self.eval_buf['acc_cs']) / len(self.eval_buf['acc_cs']),
                      sum(self.eval_buf['true_cs'])/sum(self.eval_buf['common_area_cs']),))


        return None


def correct_kps(trg_kps, prd_kps, pckthres, alpha=0.1):
    r"""Compute the number of correctly transferred key-points"""
    l2dist = torch.pow(torch.sum(torch.pow(trg_kps - prd_kps, 2), 0), 0.5)
    thres = pckthres.expand_as(l2dist).float()
    correct_pts = torch.le(l2dist, thres * alpha)

    return torch.sum(correct_pts)


