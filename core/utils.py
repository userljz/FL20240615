import torch
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict
import random
import os
import numpy as np
from core.logger_utils import get_logger


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


# def set_parameters(net, parameters):
#     # 将参数和 keys 对应起来
#     state_dict = net.state_dict()
#     for name, val in parameters:
#         state_dict[name] = torch.tensor(val)
    
#     # 加载新的 state_dict，strict=False 允许我们只更新部分参数
#     net.load_state_dict(state_dict, strict=False)
#     return net


def get_parameters(net):
    # # 0620test
    # for k, v in net.state_dict().items():
    #     print(f"{k}: {v.shape}")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# def get_parameters(net):
#     return [(name, val.detach().cpu().numpy()) for name, val in net.named_parameters() if val.requires_grad]


def print_cfg(cfg):
    mylogger = get_logger(f"{cfg.output_dir}/{cfg.logger.project}_{cfg.logger.name}.log")
    mylogger.info("------ Config values: ------")
    for key, value in cfg.items():
        mylogger.info(f"{key}: {value}")
    mylogger.info("----------------------------")

def is_main_process():
    # 检查是否在多 GPU 环境中
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK']) == 0
    # 如果不在多 GPU 环境中，默认为主进程
    return True


@dataclass
class FitRes:
    """Fit return for a client."""
    parameters: Any
    num_examples: Any


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device(device_config):
    if device_config.cuda == "cpu":
        print("Using CPU")
        return
    else:
        # set the global cuda device
        torch.backends.cudnn.enabled = True
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
        # torch.cuda.set_device(device_config.cuda)
        torch.set_float32_matmul_precision(device_config.float32_matmul_precision)
        # warnings.filterwarnings("always")
    
    
def select_round_clients(client_num, select_client_num):
    client_list = list(range(client_num))
    if select_client_num == 0:
        client_list_use = client_list
    else:
        selected_client_numbers = random.sample(client_list, select_client_num)
        selected_client_numbers.sort()
        client_list_use = selected_client_numbers
    return client_list_use


dtype_mapping = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.float64": torch.float64,
}
 