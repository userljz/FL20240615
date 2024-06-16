import hydra
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl

from core.modules import CustomCLIP, contrastive_loss
from core.utils import set_parameters, get_parameters
from core.logger_utils import get_logger


class Client(pl.LightningModule):
    def __init__(self, cfg, custom_clip, param, running_args=None):
        super().__init__()
        self.cfg = cfg
        self.loss_func = contrastive_loss
        self.param = param
        self.model = custom_clip
        
        self.val_acc_batch = []
        self.test_acc_batch = []
        
        self.mylogger = get_logger(cfg.output_dir + cfg.logger.project + cfg.logger.name)
        
    def on_train_start(self):
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
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        
        optimizer = hydra.utils.instantiate(self.cfg.optimizer, params_to_train)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        image, label = image.to(self.device), label.to(self.device)
        
        model_ret = self.model(image, label)
        loss = model_ret["loss"]
        self.mylogger.info(f"train_loss: {loss}")
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        model_ret = self.model(x, y)
        logits = model_ret["logits"]
        loss = model_ret["loss"]
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.test_acc_batch.append(acc)
        
        self.log('test_acc_batch', acc, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}
    
    def on_test_epoch_end(self):
        avg_test_acc = torch.stack(self.test_acc_batch).mean()
        print(f"----- test_acc_epoch: {avg_test_acc} -----")
        self.log('test_acc_epoch', avg_test_acc, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        model_ret = self.model(x, y)
        logits = model_ret["logits"]
        loss = model_ret["loss"]
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.val_acc_batch.append(acc)
        
        self.log('val_acc_batch', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def on_validation_epoch_end(self):
        # outputs 是一个由 validation_step 返回的字典组成的列表
        avg_val_acc = torch.stack(self.val_acc_batch).mean()
        print(f"----- val_acc_epoch: {avg_val_acc} -----")
        # Log average accuracy for the entire validation epoch
        self.log('val_acc_epoch', avg_val_acc, prog_bar=True, logger=True)
    
    

