import hydra
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from core.modules import CustomCLIP, contrastive_loss
from core.utils import set_parameters, get_parameters
from core.logger_utils import get_logger

import wandb


class Client(pl.LightningModule):
    def __init__(self, cfg, custom_clip, param, running_args):
        super().__init__()
        self.cfg = cfg
        self.loss_func = contrastive_loss
        self.param = param
        self.model = custom_clip
        
        self.train_loss_list = []
        self.val_acc_batch = []
        self.test_acc_batch = []
        self.running_args = running_args
        self.client_idx = running_args["client_idx"]
        self.round_idx = running_args["round_idx"]
        
        self.mylogger = get_logger(f"{cfg.output_dir}/{cfg.logger.project}_{cfg.logger.name}.log")
        
    def on_train_start(self):
        self.model = set_parameters(self.model, self.param)
        
        if self.round_idx == 1:
            enabled = set()
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    enabled.add(name)
            self.mylogger.info(f"Parameters to be updated: {enabled}")
    
    def on_test_start(self):
        self.model = set_parameters(self.model, self.param)
    
    def configure_optimizers(self):
        params_to_train = []
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(True)
                params_to_train.append(param)
            elif "image_encoder" in name:
                param.requires_grad_(True)
                params_to_train.append(param)
            else:
                param.requires_grad_(False)
        
        # if self.round_idx == 1:
        #     enabled = set()
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             enabled.add(name)
        #     self.mylogger.info(f"Parameters to be updated: {enabled}")
        
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params_to_train)
        
        if self.cfg.optimizer.scheduler:
            scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                },
            }
        
        else:
            return optimizer
    
    def on_train_epoch_start(self):
        self.train_loss_list = []
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        image, label = image.to(self.device), label.to(self.device)
        
        model_ret = self.model(image, label)
        loss = model_ret["loss"]
        self.train_loss_list.append(loss)
        
        return loss

    def on_train_epoch_end(self):
        # outputs 是一个由 validation_step 返回的字典组成的列表
        avg_train_loss = torch.stack(self.train_loss_list).mean()
        self.mylogger.info(f"Round[{self.round_idx}]-Client[{self.client_idx}] - Epoch[{self.current_epoch}/{self.trainer.max_epochs}] train_loss: {avg_train_loss}")
        if self.cfg.logger.wandb_enable:
            wandb.log({f"Client{self.client_idx}|Train_loss:": avg_train_loss})
        return
    
    def on_test_epoch_start(self):
        self.test_acc_batch = []
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        model_ret = self.model(x, y)
        logits = model_ret["logits"]
        loss = model_ret["loss"]
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.test_acc_batch.append(acc)
        
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        avg_test_acc = torch.stack(self.test_acc_batch).mean()
        self.mylogger.info(f"------------------------------")
        self.mylogger.info(f'Round[{self.round_idx}] - test_acc: {avg_test_acc}')
        self.mylogger.info(f"------------------------------")
        if self.cfg.logger.wandb_enable:
            wandb.log({f"Server_Test_Acc:": avg_test_acc})
    
    def on_validation_epoch_start(self):
        self.val_acc_batch = []
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        model_ret = self.model(x, y)
        logits = model_ret["logits"]
        loss = model_ret["loss"]
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.val_acc_batch.append(acc)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        # outputs 是一个由 validation_step 返回的字典组成的列表
        avg_val_acc = torch.stack(self.val_acc_batch).mean()
        self.mylogger.info(f'Round[{self.round_idx}]-Client[{self.client_idx}] - val_acc: {avg_val_acc}')
        if self.cfg.logger.wandb_enable:
            wandb.log({f"Client{self.client_idx}|Val_Acc:": avg_val_acc})
    
    

