import torch.nn.functional as F
import math
import torch

def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)


def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    logits = math.exp(t) * visual_features @ transpose(class_prototypes)
    if labels is not None:
        loss = F.cross_entropy(logits, labels)
        # print(f"{loss.shape = }")
        # print(f"{logits.shape = }")
        return {"loss": loss, "logits": logits}
    else:
        return {"loss": None, "logits": logits}


def text2text_loss(text_feat, ref_feat):
    # 计算 text_feat 和 ref_feat 之间的差异
    diff = ref_feat - text_feat  # 将 text_feat 的维度扩展以匹配 ref_feat
    
    # 计算每个差异的平方
    squared_diff = diff ** 2
    
    # 计算每个差异的均方误差
    mse_loss = squared_diff.mean(dim=1)
    
    # 累加所有均方误差
    l2loss = mse_loss.sum()
    
    return l2loss
    
    
def SupConLoss(cfg, visual_features, class_prototypes, labels):
    """Supervised Contrastive Learning:.
    It also supports the unsupervised contrastive loss in SimCLR"""
    
    """
    Args:
        features: hidden vector of shape [bsz, image_feats].
        labels: ground truth of shape [bsz].

    Returns:
        A loss scalar.
    """
    # --- generate one-hot target ---
    # visual_features, class_prototypes = visual_features.squeeze(), class_prototypes.squeeze()
    num_classes = len(cfg.dataset.classnames)
    target = F.one_hot(labels, num_classes).to(cfg.device.cuda)

    logits_per_image = 100 * visual_features @ transpose(class_prototypes)  # [bsz, num_class]
    logits = logits_per_image

    logits_per_image = logits_per_image / logits_per_image.norm(dim=-1, keepdim=True)
    logits_per_image = torch.where(target == 1, 1 - logits_per_image, logits_per_image - cfg.clip.loss_margin)
    logits_per_image = torch.clamp(logits_per_image, min=0)
    loss = torch.sum(logits_per_image)  # [1,]

    # print(f"{loss.shape = }")
    # print(f"{logits.shape = }")
    return {"loss": loss, "logits": logits}