import random
import numpy as np
import torch
import os
import hydra

from core.system import Client, Server
from core.modules import DataModule, CustomCLIP
from core.utils import get_parameters, FitRes, set_random_seed, set_device, select_round_clients


def client_fn(cfg, param):
    """
    Instantiate the Client for each Round
    :param cfg: Config
    :param param: 需要 Client 端进行实例化时初始化的参数
    :return: Client
    """
    custom_clip = CustomCLIP(cfg)
    client = Client(cfg, custom_clip, param)
    
    return client


def train_fl(cfg):
    set_random_seed(seed=cfg.seed)
    set_device(cfg.device)
    
    server = Server(cfg)
    
    # ===== FL Communication Start =====
    for round_i in range(1, cfg.fl.round + 1):
        print(f"===== Round-{round_i} Start =====")
        results = []
        
        if round_i == 1:
            param = get_parameters(CustomCLIP(cfg))
        else:
            # ===== Aggregation =====
            param = server.server_conduct(results)
            
            # ===== Test Global =====
            client_agg = client_fn(cfg, param)
            datamodule = DataModule(cfg, client_idx=1)
            trainer = hydra.utils.instantiate(cfg.trainer)
            trainer.test(client_agg, datamodule=datamodule)
            
        client_list_use = select_round_clients(cfg.fl.client_num, cfg.fl.select_client_num)
        print("----------")
        print(f"Round-{round_i} Selected Clients:")
        print(f"{client_list_use}")
        print("----------")
        
        for client_idx in client_list_use:
            print(f"===== Round-{round_i}|Client-{client_idx} Start =====")
            datamodule = DataModule(cfg, client_idx)
            client = client_fn(cfg, param)
            trainer = hydra.utils.instantiate(cfg.trainer)
            trainer.fit(client, datamodule=datamodule)
            
            _param = get_parameters(client)
            train_loader = datamodule.train_dataloader()
            num_samples = len(train_loader.dataset)
            fit_res = FitRes(_param, num_samples)
            results.append(fit_res)
            print(f"===== Round-{round_i}|Client-{client_idx} End =====")
 
    return
