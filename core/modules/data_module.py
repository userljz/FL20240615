import torch
import pytorch_lightning as pl
from core.data_utils import load_dataloader_from_generate


def count_labels(data_loader):
    """
    In a non-iid setting, count the data distribution in each client's dataloader
    :param data_loader: [DataLoader1, DataLoader2, ...]
    :return: a Dict
    """
    label_counts = {}
    for client_indice, loader_i in enumerate(data_loader):
        _label_counts = {}
        _data_loader = loader_i
        for _, labels in _data_loader:
            # If labels are not already on CPU, move them
            labels = labels.cpu()
            for label in labels:
                # If label is tensor, convert to python number
                if isinstance(label, torch.Tensor):
                    label = label.item()
                # Increment the count for this label
                _label_counts[label] = _label_counts.get(label, 0) + 1
        _label_counts = dict(sorted(_label_counts.items()))
        label_counts[f'client{client_indice}'] = _label_counts

    return label_counts

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, client_idx):
        super().__init__()
        self.cfg = cfg
        self.client_idx = client_idx
        self.train_loaders, self.val_loaders, self.test_loader = [], [], None
        
    def setup(self, stage=None):
        dataset_name = self.cfg.dataset.dataset_name
        model_name = self.cfg.clip.backbone
        batch_size = self.cfg.dataset.batch_size
        client_num = self.cfg.fl.client_num
        dataset_root = self.cfg.dataset.dataset_root
        dirichlet_alpha = self.cfg.fl.dirichlet_alpha
        
        self.train_loaders, self.val_loaders, self.test_loader = load_dataloader_from_generate(
            dataset_name, model_name, batch_size, dirichlet_alpha, dataloader_num=client_num, dataset_root=dataset_root)
        
        label_counts = count_labels(self.train_loaders)
        # for key, value in label_counts.items():
        #     self.trainer.logger.log_metrics(f"[{key}]: {value}")
            
    def train_dataloader(self):
        return self.train_loaders[self.client_idx]

    def val_dataloader(self):
        return self.val_loaders[self.client_idx]

    def test_dataloader(self):
        return self.test_loader
    