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
    