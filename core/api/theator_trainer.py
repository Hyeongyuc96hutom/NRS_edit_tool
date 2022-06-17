import os
import math
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.api import BaseTrainer
from core.model import get_model, get_loss
from core.dataset import *
from core.utils.metric import MetricHelper


class TheatorTrainer(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
 
        # TODO 필요한거 더 추가하기
        self.args = args

        random.seed(self.args.random_seed)
        np.random.seed(self.args.random_seed)
        os.environ["PYTHONHASHSEED"]=str(self.args.random_seed)
        torch.manual_seed(self.args.random_seed)
        torch.cuda.manual_seed(self.args.random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=True
        
        self.save_hyperparameters() # save with hparams

        self.model = get_model(self.args)
        self.loss_fn = get_loss(self.args)

        self.metric_helper = MetricHelper()
        self.best_val_loss = math.inf

        self.sanity_check = True
        self.restore_path = None
        self.iter_end_epoch = 20

    def on_epoch_start(self):
        if self.current_epoch > 1 and (self.current_epoch + 1) % self.iter_end_epoch == 0:
            # self.trainer.accelerator.setup_optimizers(self.trainer) # latest ver.
            self.trainer.accelerator_backend.setup_optimizers(self.trainer)

    def setup(self, stage):
        '''
            Called one each GPU separetely - stage defines if we are at fit or test step.
            We wet up only relevant datasets when stage is specified (automatically set by pytorch-lightning).
        '''
        # training stage
        if stage == 'fit' or stage is None:
            if self.args.dataset == 'ROBOT':
                self.trainset = RobotDataset(self.args, state='train') # train dataset setting
                self.valset = RobotDataset(self.args, state='val') # val dataset setting
            elif self.args.dataset == 'LAPA':
                self.trainset = LapaDataset(self.args, state='train') 
                self.valset = LapaDataset(self.args, state='val')

        # testing stage
        if stage in (None, 'test'):
            if self.args.dataset == 'ROBOT':
                self.testset = RobotDataset(self.args, state='val')
            elif self.args.dataset == 'LAPA':
                self.testset = LapaDataset(self.args, state='val')

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            sampler=MPerClassSampler(self.trainset.label_list, 
                                     self.args.batch_size//2, self.args.batch_size)
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

    def test_dataloader(self):
        # TODO testset이 따로 있으면 그걸로 하기
        return DataLoader(
            self.testset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

    def forward(self, x):
        """
            forward data
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
            forward for mini-batch
        """
        img_path, x, y = batch

        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return  {
            'loss': loss,
        }

    def training_epoch_end(self, outputs):
        train_loss, cnt = 0, 0

        for output in outputs:
            train_loss += output['loss'].cpu().data.numpy()
            cnt += 1

        train_loss_mean = train_loss/cnt

        # write train loss
        self.metric_helper.write_loss(train_loss_mean, task='train')

    def validation_step(self, batch, batch_idx): # val - every batch
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
            'img_path': img_path,
            'y': y,
            'y_hat': y_hat.argmax(dim=1).detach().cpu(),
            'logit': y_hat
        }

    def validation_epoch_end(self, outputs): # val - every epoch
        if self.sanity_check:
            self.sanity_check = False
        else:
            self.restore_path = os.path.join(self.args.save_path, self.logger.log_dir)
            metrics = self.metric_helper.calc_metric() # 매 epoch 마다 metric 계산 (TP, TN, .. , accuracy, precision, recaull, f1-score)
        
            val_loss, cnt = 0, 0
            for output in outputs: 
                val_loss += output['val_loss'].cpu().data.numpy()
                cnt += 1

            val_loss_mean = val_loss/cnt
            metrics['Loss'] = val_loss_mean

            '''
                metrics = {
                    'TP': cm.TP[self.OOB_CLASS],
                    'TN': cm.TN[self.OOB_CLASS],
                    'FP': cm.FP[self.OOB_CLASS],
                    'FN': cm.FN[self.OOB_CLASS],
                    'Accuracy': cm.ACC[self.OOB_CLASS],
                    'Precision': cm.PPV[self.OOB_CLASS],
                    'Recall': cm.TPR[self.OOB_CLASS],
                    'F1-Score': cm.F1[self.OOB_CLASS],
                    'OOB_metric':
                    'Over_estimation':
                    'Under_estimation':
                    'Correspondence_estimation':
                    'UNCorrespondence_estimation':
                    'Loss':
            }
            '''

            self.log_dict(metrics, on_epoch=True, prog_bar=True)
            
            # save result.csv 
            self.metric_helper.save_metric(metric=metrics, epoch=self.current_epoch, args=self.args, save_path=os.path.join(self.args.save_path, self.logger.log_dir))

            # write val loss
            self.metric_helper.write_loss(val_loss_mean, task='val')
            
            self.metric_helper.save_loss_pic(save_path=os.path.join(self.args.save_path, self.logger.log_dir))

            if not self.args.use_lightning_style_save:
                if self.best_val_loss > val_loss_mean : # math.inf 보다 현재 epoch val loss 가 작으면,
                    self.best_val_loss = val_loss_mean # self.best_val_loss 업데이트. 
                    self.save_checkpoint()

                if self.current_epoch + 1 == self.args.max_epoch: # max_epoch 모델 저장
                    # TODO early stopping 적용시 구현 필요
                    self.best_val_loss = val_loss_mean
                    self.save_checkpoint()

            # Re-labeling
            if self.current_epoch > 1 and (self.current_epoch + 1) % self.iter_end_epoch == 0:
                d_loader = DataLoader(self.trainset, batch_size=self.args.batch_size*20, shuffle=False, num_workers=self.args.num_workers)
                
                change_list = []

                for _, img, lbs in tqdm(d_loader):
                    img = img.cuda()
                    outputs = self.model(img)
                    ids = list(torch.argmax(outputs, -1).cpu().data.numpy())
                    change_list += ids

                self.trainset.change_labels(change_list)


    def test_step(self, batch, batch_idx):
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu())

        self.log('test_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'test_loss': loss,
        }

    def test_epoch_end(self, outputs):
        metrics = self.metric_helper.calc_metric()
        
        for k, v in metrics.items():
            if k in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                self.log('test_'+k, v, on_epoch=True, prog_bar=True)
