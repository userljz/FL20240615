import torch
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict
import random
import os
import numpy as np


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


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
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
        torch.cuda.set_device(device_config.cuda)
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

