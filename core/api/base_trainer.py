import abc
import os
import torch
import natsort
from glob import glob

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# from core.accessory.RepVGG.repvgg import repvgg_model_convert


class BaseTrainer(pl.LightningModule):
    """
        Base Trainer Class based on Pytorch Lightning
    """
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def setup(self):
        pass
    
    @abc.abstractmethod
    def train_dataloader(self):
        pass
    
    @abc.abstractmethod
    def val_dataloader(self):
        pass
    
    @abc.abstractmethod
    def test_dataloader(self):
        pass
    
    @abc.abstractmethod
    def forward(self):
        pass
    
    @abc.abstractmethod
    def training_step(self):
        pass
    
    @abc.abstractmethod
    def validation_step(self):
        pass
    
    @abc.abstractmethod
    def validation_epoch_end(self):
        pass
    
    @abc.abstractmethod
    def test_step(self):
        pass
    
    @abc.abstractmethod
    def test_epoch_end(self):
        pass

    # @classmethod
    def configure_optimizers(self):
        """
            Details for return values
            1) First list : list of optimizers
            2) Second list : list of schedulers
        """
        # ver 1 - yaml
        opt_name = self.args.optimizer

        if opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )
        elif opt_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.args.init_lr,
                weight_decay=self.args.weight_decay,
            )

        schdlr_name = self.args.lr_scheduler

        if schdlr_name == 'step_lr':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=self.args.lr_scheduler_step, 
                gamma=self.args.lr_scheduler_factor,
                )
        elif schdlr_name == 'mul_lr':
            scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
                optimizer,
                lr_lambda=lambda epoch: self.args.lr_scheduler_factor,
            )
        elif schdlr_name == 'mul_step_lr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.args.lr_milestones,
                gamma=self.args.lr_scheduler_factor,
            )
        elif schdlr_name == 'reduced':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.args.lr_scheduler_factor,
                patience=self.args.lr_scheduler_step,
                threshold=1e-4,
                threshold_mode='rel',
                min_lr=1e-7,
                verbose=True,
            )
        elif schdlr_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.t_max_iter,
                eta_min=0,
                last_epoch=-1,
                verbose=True,
            )
        else:
            scheduler = None

        print('[+] Optimizer and Scheduler are set ', optimizer, scheduler)
        
        if scheduler is not None:
            if schdlr_name == 'reduced':
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
                    'monitor': 'val_loss'
                }
            else: 
                return [optimizer], [scheduler]
        else:
            return [optimizer]
    
    # @classmethod
    def configure_callbacks(self):
        """
            Return the list of utilities such as Earlystopping, ModelCheckpoint, ...
        """
        callbacks = []
        
        if self.args.use_early_stop:
            early_stop = EarlyStopping(
                monitor=self.args.early_stop_monitor, 
                mode=self.args.ealry_stop_mode,
                patience=self.args.ealry_stop_patience,
                )
            callbacks.append(early_stop)

        lrMonitor = LearningRateMonitor(
            logging_interval='epoch',
        )

        callbacks.append(lrMonitor)
        
        if self.args.use_lightning_style_save:            
            checkpoint = ModelCheckpoint(
                filename='{epoch}-{Mean_metric:.4f}-best',
                save_top_k=self.args.save_top_n,
                save_last=True,
                verbose=True,
                monitor='Mean_metric',
                mode='max')

            checkpoint.CHECKPOINT_NAME_LAST = '{epoch}-{val_loss:.4f}-last'
        
            callbacks.append(checkpoint)

        print('[+] Callbacks are set ', callbacks)
        
        return callbacks

    # torch style save checkpoint
    # repvgg use only
    def save_checkpoint(self):
        saved_pt_list = glob(os.path.join(self.logger.log_dir, 'checkpoints', '*pt'))

        print('saved_pt_list ====> ', saved_pt_list)

        if len(saved_pt_list) > self.args.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.args.save_top_n+1)]:
                os.remove(li)

        save_path = '{}/epoch:{}-Mean_metric:{:.4f}.pt'.format(
                    os.path.join(self.logger.log_dir, 'checkpoints'),
                    self.current_epoch,
                    self.best_mean_metric,
                )

        # _ = repvgg_model_convert(self.model.feature_module, save_path=save_path)        

        print('[+] save checkpoint(torch ver.) : ', save_path)
        
        # os.makedirs(os.path.join(self.args.save_path), exist_ok=True)

        # saved_pt_list = glob(os.path.join(self.args.save_path, self.logger.log_dir, 'checkpoints', '*pt'))

        # print('saved_pt_list ====> ', saved_pt_list)

        # if len(saved_pt_list) > self.args.save_top_n:
        #     saved_pt_list = natsort.natsorted(saved_pt_list)

        #     for li in saved_pt_list[:-(self.args.save_top_n+1)]:
        #         os.remove(li)

        # save_path = '{}/epoch:{}-loss_val:{:.4f}.pt'.format(
        #             os.path.join(self.args.save_path, self.logger.log_dir, 'checkpoints'),
        #             self.current_epoch,
        #             self.best_val_loss.item()
        #         )

        # if self.args.num_gpus > 1:
        #     torch.save(self.model.module.state_dict(), save_path)
        # else:
        #     torch.save(self.model.state_dict(), save_path)

        # print('[+] save checkpoint(torch ver.) : ', save_path)

    def save_checkpoint_multi(self):
        saved_pt_list = glob(os.path.join(self.logger.log_dir, 'checkpoints', '*pt'))

        print('saved_pt_list ====> ', saved_pt_list)

        if len(saved_pt_list) > self.args.save_top_n:
            saved_pt_list = natsort.natsorted(saved_pt_list)

            for li in saved_pt_list[:-(self.args.save_top_n+1)]:
                os.remove(li)

        save_path = '{}/epoch:{}-Mean_metric:{:.4f}.pt'.format(
                    os.path.join(self.logger.log_dir, 'checkpoints'),
                    self.current_epoch,
                    self.best_mean_metric,
                )

        # _ = repvgg_model_convert(self.model.repvgg_feature, save_path=save_path)        

        print('[+] save checkpoint(torch ver.) : ', save_path)