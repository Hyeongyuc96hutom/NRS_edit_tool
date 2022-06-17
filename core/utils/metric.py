import torch
import math
import numpy as np
from pycm import *
from core.utils.misc import *

import matplotlib.pyplot as plt

class MetricHelper():
    """
        Help metric computation.
    """
    def __init__(self):
        # TODO 필요한거 더 넣기
        self.EXCEPTION_NUM = -100
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        self.pred_list = []
        self.gt_list = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.epoch = 0

    def write_preds(self, pred_list, gt_list):
        for pred, gt in zip(pred_list, gt_list):
            self.pred_list.append(pred.item())
            self.gt_list.append(gt.item())

    def write_loss(self, loss_val, task='train'):
        if task == 'train':
            self.train_loss.append(loss_val)
        elif task == 'val':
            self.valid_loss.append(loss_val)
            self.epoch += 1
        elif task == 'test':
            self.test_loss.append(loss_val)

    def calc_metric(self):
        classes = [self.IB_CLASS, self.OOB_CLASS]

        cm = ConfusionMatrix(self.gt_list, self.pred_list, classes=classes) # pycm
        
        try: # for [1,1,1] [1,1,1] error solving
            cm.relabel(mapping={0:self.IB_CLASS, 1:self.OOB_CLASS}) # when [1,1,0] [0,1,0] return => cm.classes : [0, 1]
        except:
            cm.relabel(mapping={'0':self.IB_CLASS, '1':self.OOB_CLASS}) # when [1,1,1] [1,1,1] return => cm.classes : ['0', '1']

        metrics = {
            'TP': cm.TP[self.OOB_CLASS],
            'TN': cm.TN[self.OOB_CLASS],
            'FP': cm.FP[self.OOB_CLASS],
            'FN': cm.FN[self.OOB_CLASS],
            'Accuracy': cm.ACC[self.OOB_CLASS],
            'Precision': cm.PPV[self.OOB_CLASS],
            'Recall': cm.TPR[self.OOB_CLASS],
            'F1-Score': cm.F1[self.OOB_CLASS],
            'Jaccard': cm.J[self.OOB_CLASS],
        }

        # np casting for zero divide to inf
        TP = np.float16(metrics['TP'])
        FP = np.float16(metrics['FP'])
        TN = np.float16(metrics['TN'])
        FN = np.float16(metrics['FN'])

        metrics['CR'] = (TP - FP) / (FN + TP + FP) # 잘못예측한 OOB / predict OOB + 실제 OOB # Confidence Ratio
        metrics['OR'] = FP / (FN + TP + FP) # Over estimation ratio
        metrics['Mean_metric'] = (metrics['CR'] + (1-metrics['OR'])) / 2 # for train

        # Predict / GT CLASS elements num
        metrics['gt_IB']= self.gt_list.count(self.IB_CLASS)
        metrics['gt_OOB']= self.gt_list.count(self.OOB_CLASS)
        metrics['predict_IB']= self.pred_list.count(self.IB_CLASS)
        metrics['predict_OOB']= self.pred_list.count(self.OOB_CLASS)
        
        # exception
        for k, v in metrics.items():
            if v == 'None': # ConfusionMetrix return
                metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v) or np.isnan(v): # numpy return
                metrics[k] = self.EXCEPTION_NUM

        self.pred_list = []
        self.gt_list = []

        return metrics

    def aggregate_calc_metric(self, metrics_list):
        advanced_metrics = {
            'TP': 0,
            'TN': 0,
            'FP': 0,
            'FN': 0,
            'Accuracy': self.EXCEPTION_NUM,
            'Precision': self.EXCEPTION_NUM,
            'Recall': self.EXCEPTION_NUM,
            'F1-Score': self.EXCEPTION_NUM,
            'Jaccard': self.EXCEPTION_NUM, # TO-DO should calc
            'gt_IB': 0,
            'gt_OOB': 0,
            'predict_IB': 0,
            'predict_OOB': 0,
        }

        # sum of TP/TN/FP/FN
        for metrics in metrics_list:
            advanced_metrics['TP'] += metrics['TP']
            advanced_metrics['TN'] += metrics['TN']
            advanced_metrics['FP'] += metrics['FP']
            advanced_metrics['FN'] += metrics['FN']
            
            # sum IB / OOB
            advanced_metrics['gt_IB'] += metrics['gt_IB']
            advanced_metrics['gt_OOB'] += metrics['gt_OOB']
            advanced_metrics['predict_IB'] += metrics['predict_IB']
            advanced_metrics['predict_OOB'] += metrics['predict_OOB']

        # np casting for zero divide to inf
        TP = np.float16(advanced_metrics['TP'])
        FP = np.float16(advanced_metrics['FP'])
        TN = np.float16(advanced_metrics['TN'])
        FN = np.float16(advanced_metrics['FN'])

        advanced_metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)
        advanced_metrics['Precision'] = TP / (TP + FP)
        advanced_metrics['Recall'] = TP / (TP + FN)
        advanced_metrics['F1-Score'] = 2 * ((advanced_metrics['Precision'] * advanced_metrics['Recall']) / (advanced_metrics['Precision'] + advanced_metrics['Recall']))

        advanced_metrics['CR'] = (TP - FP) / (FN + TP +FP) # 잘못예측한 OOB / predict OOB + 실제 OOB # Confidence Ratio
        advanced_metrics['OR'] = FP / (FN + TP + FP)  # Over estimation ratio
        advanced_metrics['Mean_metric'] = (advanced_metrics['CR'] + (1-advanced_metrics['OR'])) / 2 # for train

        # calc mCR, mOR
        advanced_metrics['mCR'] = np.mean([metrics['CR'] for metrics in metrics_list])
        advanced_metrics['mOR'] = np.mean([metrics['OR'] for metrics in metrics_list])

        # calc mPrecision, mRecall
        advanced_metrics['mPrecision'] = np.mean([metrics['Precision'] for metrics in metrics_list])
        advanced_metrics['mRecall'] = np.mean([metrics['Recall'] for metrics in metrics_list])

        # calc Jaccard index (https://neo4j.com/docs/graph-data-science/current/alpha-algorithms/jaccard/)
        advanced_metrics['Jaccard'] = np.float16(advanced_metrics['TP']) / np.float16(advanced_metrics['predict_OOB'] + advanced_metrics['gt_OOB'] - advanced_metrics['TP'])

        # exception
        for k, v in advanced_metrics.items():
            if v == 'None': # ConfusionMetrix return
                advanced_metrics[k] = self.EXCEPTION_NUM
            elif np.isinf(v) or np.isnan(v): # numpy return
                advanced_metrics[k] = self.EXCEPTION_NUM

        return advanced_metrics

    def save_metric(self, metric, epoch, args, save_path, task='OOB'):
        if task=='OOB':
            save_OOB_result_csv(metric=metric, epoch=epoch, args=args, save_path=save_path)

    def save_loss_pic(self, save_path):
        fig = plt.figure(figsize=(32, 16))

        plt.ylabel('Loss', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)

        plt.plot(range(self.epoch), self.train_loss)
        plt.plot(range(self.epoch), self.valid_loss)
        
        plt.legend(['Train', 'Val'], fontsize=20)
        plt.savefig(save_path + '/loss.png')

        plt.close(fig)
