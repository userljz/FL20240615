import random
import numpy as np
import torch
import os
import hydra

from core.system import Client, Server
from core.modules import DataModule, CustomCLIP
from core.utils import get_parameters, FitRes, set_random_seed, set_device, select_round_clients
from core.logger_utils import get_logger


def client_fn(cfg, param, running_args):
    """
    Instantiate the Client for each Round
    :param cfg: Config
    :param param: 需要 Client 端进行实例化时初始化的参数
    :return: Client
    """
    custom_clip = CustomCLIP(cfg)
    client = Client(cfg, custom_clip, param, running_args)
    
    return client


def train_fl(cfg):
    set_random_seed(seed=cfg.seed)
    set_device(cfg.device)
    
    server = Server(cfg)
    mylogger = get_logger(f"{cfg.output_dir}/{cfg.logger.project}_{cfg.logger.name}.log")
    running_args = {}
    
    # ===== FL Communication Start =====
    for round_i in range(1, cfg.fl.round + 1):
        mylogger.info(f"===== Round-{round_i} Start =====")
        results = []
        running_args["round_idx"] = round_i
        
        if round_i == 1:
            param = get_parameters(CustomCLIP(cfg))
            num_sanity_val_steps = 2
        else:
            num_sanity_val_steps = 0
            
        client_list_use = select_round_clients(cfg.fl.client_num, cfg.fl.select_client_num)
        mylogger.info("----------")
        mylogger.info(f"Round-{round_i} Selected Clients:")
        mylogger.info(f"{client_list_use}")
        mylogger.info("----------")
        
        for client_idx in client_list_use:
            mylogger.info(f"\n===== Round-{round_i}|Client-{client_idx} Start =====")
            running_args["client_idx"] = client_idx
            datamodule = DataModule(cfg, client_idx)
            client = client_fn(cfg, param, running_args)
            trainer = hydra.utils.instantiate(cfg.trainer, num_sanity_val_steps=num_sanity_val_steps)
            trainer.fit(client, datamodule=datamodule)
            
            _param = get_parameters(client)
            train_loader = datamodule.train_dataloader()
            num_samples = len(train_loader.dataset)
            fit_res = FitRes(_param, num_samples)
            results.append(fit_res)
            
        
        # ===== Aggregation =====
        param = server.server_conduct(results)
        
        # ===== Test Global =====
        client_agg = client_fn(cfg, param, running_args)
        datamodule = DataModule(cfg, client_idx=1)
        trainer = hydra.utils.instantiate(cfg.trainer)
        trainer.test(client_agg, datamodule=datamodule)
 
    return
