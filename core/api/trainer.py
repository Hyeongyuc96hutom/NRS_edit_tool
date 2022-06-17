import os
import math
import numpy as np
import csv
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from core.api import BaseTrainer
from core.model import get_model, get_loss
from core.dataset import *
from core.utils.metric import MetricHelper



class CAMIO(BaseTrainer):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
 
        # TODO 필요한거 더 추가하기
        self.args = args
        self.save_hyperparameters() # save with hparams

        self.model = get_model(self.args)
        self.loss_fn = get_loss(self.args)

        self.metric_helper = MetricHelper()
        self.hem_helper = HEMHelper(self.args)

        self.best_val_loss = math.inf
        self.best_mean_metric = -2 # repvgg use only

        self.sanity_check = True
        self.restore_path = self.args.restore_path # inference module args / save path of hem df 
        
        self.cur_step = 0
        self.last_epoch = -1
        self.hem_extract_mode = self.args.hem_extract_mode

        # hem-online
        if 'online' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)
        
        # hem-offline // hem_train, general_train 에서는 Hem 생성하지 않음.
        elif 'offline' in self.args.hem_extract_mode:
            self.hem_helper.set_method(self.hem_extract_mode)
            self.last_epoch = self.args.max_epoch - 1

        self.emb_only = self.args.use_emb_only

    def on_epoch_start(self):
        if hasattr(self, 'cur_stage'):
            print('restore path : ', self.restore_path)
            if self.cur_stage > 1 and self.restore_path is not None and self.current_epoch == 0:
                d_loader = DataLoader(self.trainset, 
                                    batch_size=self.args.batch_size, 
                                    shuffle=False, 
                                    num_workers=self.args.num_workers)
                
                change_list = []
                
                self.model.eval()
                with torch.no_grad():
                    for _, img, lbs in tqdm(d_loader):
                        img = img.cuda()
                        outputs = self.model(img)
                        ids = list(torch.argmax(outputs, -1).cpu().data.numpy())
                        change_list += ids

                self.trainset.label_list = change_list
                self.args.restore_path = None
                self.model = get_model(self.args)
        
    def change_deploy_mode(self, pt_path=None):
        if 'repvgg' in self.args.model:
            if pt_path: # load from pt path
                self.model.load_feature_module(pt_path)
            else: # load from rule of online (restore)
                self.model.change_deploy_mode()

        if 'multi' in self.args.model:
            self.model.change_deploy_mode()

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
        if 'hem-focus' in self.hem_extract_mode:
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=FocusSampler(self.trainset.label_list,
                                     self.args)
            )
        elif self.args.experiment_type == 'theator':
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                sampler=MPerClassSampler(self.trainset.label_list, 
                                        self.args.batch_size//2, self.args.batch_size)
            )
        else:
            return DataLoader(
                self.trainset,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                shuffle=True,
                drop_last=True,
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
        if self.emb_only:
            emb = self.model(x)
            
            sim_dist = torch.zeros((emb.size(0), self.model.proxies.size(1))).to(emb.device)
        
            for d in range(sim_dist.size(1)):
                sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, self.model.proxies[:, d].unsqueeze(0))
                
            return sim_dist
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        if 'hem-emb' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
            
        elif 'hem-focus' in self.hem_extract_mode and self.training:
            img_path, x, y = batch
            
            loss = self.hem_helper.compute_hem(self.model, x, y, self.loss_fn)
        else:
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

        # save number of train dataset (rs, nrs)
        if self.current_epoch == self.last_epoch:
            transet_rs_count, trainset_nrs_count = self.trainset.number_of_rs_nrs()
            valset_rs_count, valset_nrs_count = self.valset.number_of_rs_nrs()
            
            save_data = {
                'train_dataset': {
                    'rs': transet_rs_count,
                    'nrs': trainset_nrs_count 
                },
                'target_hem_count': {
                    'rs': transet_rs_count // 3,
                    'nrs': trainset_nrs_count // 3
                }
            }

            val_save_data = {
                'val_dataset': {
                    'rs': valset_rs_count,
                    'nrs': valset_nrs_count 
                },
            }

            patient_per_dic = self.valset.number_of_patient_rs_nrs()
            
            val_save_data.update(patient_per_dic)

            with open(os.path.join(self.restore_path, 'DATASET_COUNT.json'), 'w') as f:
                json.dump(save_data, f, indent=2)
            
            with open(os.path.join(self.restore_path, 'PATIENTS_DATASET_COUNT.json'), 'w') as f:
                json.dump(val_save_data, f, indent=2)

    def validation_step(self, batch, batch_idx): # val - every batch
        img_path, x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.metric_helper.write_preds(y_hat.argmax(dim=1).detach().cpu(), y.cpu()) # MetricHelper 에 저장

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        return {
            'val_loss': loss,
        }

    def validation_epoch_end(self, outputs): # val - every epoch
        if self.sanity_check:
            print('sanity check')
            self.restore_path = os.path.join(self.args.save_path, self.logger.log_dir) # hem-df path / inference module restore path
            print('SANITY RESTORE PATH : ', self.restore_path)

            self.sanity_check = False

        else:
            self.restore_path = os.path.join(self.args.save_path, self.logger.log_dir) # hem-df path / inference module restore path
            
            metrics = self.metric_helper.calc_metric() # 매 epoch 마다 metric 계산 (TP, TN, .. , accuracy, precision, recaull, f1-score)
        
            val_loss, cnt = 0, 0
            for output in outputs: 
                val_loss += output['val_loss'].cpu().data.numpy()
                cnt += 1

            val_loss_mean = val_loss/cnt
            metrics['Loss'] = val_loss_mean

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
                    
            # repvgg는 별도로 torch style save
            elif 'repvgg' in self.args.model:
                if self.best_mean_metric < metrics['Mean_metric']: # 기존 best mean metric 보다 현재 epoch 의 mean metric 이 더 크다면, 
                    self.best_mean_metric = metrics['Mean_metric'] # best mean metric = 현재 mean metric.
                    self.save_checkpoint()
            
            elif 'multi' in self.args.model:
                if self.best_mean_metric < metrics['Mean_metric']:
                    self.best_mean_metric = metrics['Mean_metric']
                    self.save_checkpoint_multi()
                      
            '''
            # Hard Example Mining (Offline)
            if self.current_epoch == self.last_epoch:
                if self.args.stage not in ['hem_train', 'general_train'] and self.args.hem_extract_mode == 'all-offline': 

                    self.hem_helper.set_restore_path(self.restore_path)

                    # softmax_hem_df, voting_hem_df, vi_hem_df = self.hem_helper.compute_hem(self.model, outputs)
                    # softmax_hem_df, voting_hem_df, vi_hem_df = self.hem_helper.compute_hem(self.model, self.valset)
                    # softmax_diff_small_hem_final_df, softmax_diff_large_hem_final_df, voting_hem_final_df, vi_small_hem_final_df, vi_large_hem_final_df
                    softmax_diff_small_hem_df, softmax_diff_large_hem_df, voting_hem_df, vi_small_hem_df, vi_large_hem_df = self.hem_helper.compute_hem(self.model, self.valset)
                    
                    softmax_diff_small_hem_df.to_csv(os.path.join(self.restore_path, 'softmax_diff_small_{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
                    softmax_diff_large_hem_df.to_csv(os.path.join(self.restore_path, 'softmax_diff_large_{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)

                    voting_hem_df.to_csv(os.path.join(self.restore_path, 'voting_{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
                    
                    vi_small_hem_df.to_csv(os.path.join(self.restore_path, 'vi_small_{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
                    vi_large_hem_df.to_csv(os.path.join(self.restore_path, 'vi_large_{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)

                elif self.args.stage not in ['hem_train', 'general_train'] and 'offline' in self.args.hem_extract_mode: 
                    self.hem_helper.set_restore_path(self.restore_path)

                    # hem_df = self.hem_helper.compute_hem(self.model, outputs)
                    hem_df = self.hem_helper.compute_hem(self.model, self.valset)
                    hem_df.to_csv(os.path.join(self.restore_path, '{}-{}-{}.csv'.format(self.args.model, self.args.hem_extract_mode, self.args.fold)), header=False) # restore_path (mobilenet_v3-hem-vi-fold-1.csv)
            '''
            
            
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
