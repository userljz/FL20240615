import random
import numpy as np
import torch
import os
import hydra
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as pl

from core.system import Client, Server
from core.modules import DataModule, CustomCLIP
from core.utils import get_parameters, FitRes, set_random_seed, set_device, select_round_clients, print_cfg, is_main_process
from core.logger_utils import get_logger
from core.data_utils import load_dataloader_from_generate


def client_fn(cfg, param, running_args, custom_clip):
    """
    Instantiate the Client for each Round
    :param cfg: Config
    :param param: 需要 Client 端进行实例化时初始化的参数
    :return: Client
    """
    client = Client(cfg, custom_clip, param, running_args)
    
    return client


def train_fl(cfg):
    if is_main_process():
        print_cfg(cfg)
        if cfg.logger.wandb_enable:
            config_dict = OmegaConf.to_container(cfg, resolve=True)
            wandb.init(project=cfg.logger.project, name=cfg.logger.name, config=config_dict)
        mylogger = get_logger(f"{cfg.output_dir}/{cfg.logger.project}_{cfg.logger.name}.log")

    # set_random_seed(seed=cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)
    set_device(cfg.device)
    

    server = Server(cfg)
    
    running_args = {}
    
    train_loaders, val_loaders, test_loader = load_dataloader_from_generate(dataset_name=cfg.dataset.dataset_name,
                                                                            model_name=cfg.clip.backbone,
                                                                            batch_size=cfg.dataset.batch_size,
                                                                            dirichlet_alpha=cfg.fl.dirichlet_alpha,
                                                                            dataloader_num=cfg.fl.client_num,
                                                                            dataset_root=cfg.dataset.dataset_root)
    custom_clip = CustomCLIP(cfg)
    test_accu_list = []
    # ===== FL Communication Start =====
    for round_i in range(1, cfg.fl.round + 1):
        if is_main_process():
            mylogger.info(f"===== Round-{round_i} Start =====")
        results = []
        running_args["round_idx"] = round_i
        
        if round_i == 1:
            param = get_parameters(custom_clip)
            num_sanity_val_steps = 2
        else:
            num_sanity_val_steps = 0
            
        client_list_use = select_round_clients(cfg.fl.client_num, cfg.fl.select_client_num)
        if is_main_process():
            mylogger.info("----------")
            mylogger.info(f"Round-{round_i} Selected Clients:")
            mylogger.info(f"{client_list_use}")
            mylogger.info("----------")
        
        if cfg.clip.momentum_ref:
            # 动量更新 Ref Text
            momentum_ref_list, num_samples_list = [], []
            
        for client_idx in client_list_use:
            if is_main_process():
                mylogger.info(f"\n===== Round-{round_i}|Client-{client_idx} Start =====")
            running_args["client_idx"] = client_idx
            datamodule = DataModule(cfg, client_idx, train_loaders, val_loaders, test_loader)
            client = client_fn(cfg, param, running_args, custom_clip)
            trainer = hydra.utils.instantiate(cfg.trainer, num_sanity_val_steps=num_sanity_val_steps)
            trainer.fit(client, datamodule=datamodule)
            
            _param = get_parameters(client)
            train_loader = datamodule.train_dataloader()
            num_samples = len(train_loader.dataset)
            fit_res = FitRes(_param, num_samples)
            results.append(fit_res)
            
            if cfg.clip.momentum_ref:
                with torch.no_grad():
                    _text_feat = client.model.get_text_features()
                    momentum_ref_list.append(_text_feat.detach().cpu())
                    num_samples_list.append(num_samples)
        
            
        
        # ===== Aggregation =====
        param = server.server_conduct(results)
        
        # ===== Test Global =====
        client_agg = client_fn(cfg, param, running_args, custom_clip)
        datamodule = DataModule(cfg, None, None, None, test_loader)
        trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.test(client_agg, datamodule=datamodule)
        test_accu_list.append(client_agg.avg_test_acc)
        
        # ===== Momentum Update Ref =====
        if cfg.clip.momentum_ref:
            with torch.no_grad():
                weights_tensor = torch.tensor(num_samples_list).view(-1, 1, 1)
                stacked_tensors = torch.stack(momentum_ref_list).cpu()
                
                momentum_ref = (stacked_tensors * weights_tensor).sum(dim=0) / sum(num_samples_list)
                ref = custom_clip.prompt_learner.class_text_features.cpu()
                
                momentum_ref = momentum_ref / momentum_ref.norm(dim=-1, keepdim=True)
                ref = ref / ref.norm(dim=-1, keepdim=True)
                
                new_ref = (1-float(cfg.clip.momentum_weight)) * ref + float(cfg.clip.momentum_weight) * momentum_ref
                
                # print(f"weights_tensor.requires_grad: {weights_tensor.requires_grad}")
                # print(f"stacked_tensors.requires_grad: {stacked_tensors.requires_grad}")
                # print(f"momentum_ref.requires_grad: {momentum_ref.requires_grad}")
                # print(f"ref.requires_grad: {ref.requires_grad}")
                # print(f"new_ref.requires_grad: {new_ref.requires_grad}")
    
                for i, (key, val) in enumerate(custom_clip.state_dict().items()):
                    if 'prompt_learner.class_text_features' in key:
                        # print(f"{new_ref.shape = }")
                        # print(f"{param[i].shape = }")
                        param[i] = new_ref.detach().cpu().numpy()
                        break
    
    max_test_acc = max(test_accu_list)
    if is_main_process():
        mylogger.info(f"Max Test Acc: {max_test_acc}")
        if cfg.logger.wandb_enable:
            wandb.log({f"Max_Test_Acc": max_test_acc})
            
    return
