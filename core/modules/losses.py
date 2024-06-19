import torch.nn.functional as F
import math

def transpose(x):
    return x.t() if x.dim() == 2 else x.permute(0, 2, 1)


def contrastive_loss(visual_features, class_prototypes, labels=None, t=0.07):
    logits = math.exp(t) * visual_features @ transpose(class_prototypes)
    if labels is not None:
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
    else:
        return {"loss": None, "logits": logits}


def text2text_loss(text_feat, ref_feat):
    # 计算 text_feat 和 ref_feat 之间的差异
    diff = ref_feat - text_feat.unsqueeze(0)  # 将 text_feat 的维度扩展以匹配 ref_feat
    
    # 计算每个差异的平方
    squared_diff = diff ** 2
    
    # 计算每个差异的均方误差
    mse_loss = squared_diff.mean(dim=1)
    
    # 累加所有均方误差
    l2loss = mse_loss.sum()
    
    return l2loss
    