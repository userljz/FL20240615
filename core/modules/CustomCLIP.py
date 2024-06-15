import torch
import torch.nn as nn
from clip import clip
from core.modules.losses import contrastive_loss
from core.modules.PromptLearner import PromptLearner


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class ClipModel_from_generated(nn.Module):
    def __init__(self, cfg):
        super(ClipModel_from_generated, self).__init__()
        
        if cfg.clip.backbone == 'RN50':
            img_emb_length = 1024
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif cfg.clip.backbone == 'ViT-B/32':
            img_emb_length = 512
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif cfg.clip.backbone == 'ViT-B32-timm':
            img_emb_length = 1000
            fc_input_dim = img_emb_length
            fc_output_dim = 512
        
        elif cfg.clip.backbone == 'BLIP-base' or cfg.clip.backbone == 'ALBEF-base':
            img_emb_length = 256
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        elif cfg.clip.backbone == 'BLIP-base-noproj' or cfg.clip.backbone == 'ALBEF-base-noproj':
            img_emb_length = 768
            fc_input_dim = img_emb_length
            fc_output_dim = img_emb_length
        else:
            print('Please specify the img_emb_length')
        
        mlp_hiddenlayer_num = cfg.clip.mlp_hiddenlayer_num
        
        self.mlp = nn.Sequential(
            nn.Linear(fc_input_dim, mlp_hiddenlayer_num),
            nn.ReLU(),
            nn.Linear(mlp_hiddenlayer_num, fc_output_dim)
        ).to(cfg.device.cuda)
    
    def forward(self, img_emb):
        image_features = self.mlp(img_emb)
        return image_features


class CustomCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        origin_clip, _ = clip.load(cfg.clip.backbone, device=cfg.device.cuda)
        classnames = cfg.dataset.classnames
        
        self.image_encoder = ClipModel_from_generated(cfg)
        self.text_encoder = TextEncoder(origin_clip)
        self.prompt_learner = PromptLearner(cfg, classnames, origin_clip, self.text_encoder)
        self.logit_scale = origin_clip.logit_scale
        self.dtype = origin_clip.dtype
        self.cfg = cfg
        self.loss = contrastive_loss

    def forward_text_to_text(self):
        with torch.no_grad():
            # 获得 Frozen text Embedding 作为 Anchor(Constraint)
            class_text_features = self.prompt_learner.class_text_features
            class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

        if torch.rand(1).item() < 0.5:
            noise = 0.05 * torch.randn_like(class_text_features)
            class_text_features.add_(noise)
        
        # 待训练的 Text Embeddings 称为 prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.prompt_learner.tokenized_prompts_all_c_)

        if self.cfg.clip.text_correction:
            w = self.prompt_learner.w
            text_features = text_features + w

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.unsqueeze(0)

        label = torch.arange(self.prompt_learner.n_cls_all, device=class_text_features.device, dtype=torch.long).unsqueeze(0).expand(class_text_features.size(0), -1)
        
        ret_dict = self.loss(text_features, class_text_features, label, t=self.logit_scale)
        return ret_dict["loss"]

    def forward(self, image, label=None):
        print(f"===== In CustomCLIP forward =====")
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        if self.cfg.clip.text_correction:
            w = self.prompt_learner.w
            text_features = text_features + w
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        print(f"{label = }")
        ret_dict = self.loss(image_features, text_features, label, t=self.logit_scale)
        loss, logits = ret_dict["loss"], ret_dict["logits"]

        if self.cfg.clip.text_to_text_enable:
            print(f"{loss = }")
            print(f"{self.cfg.clip.text_to_text_weight = }")
            print(f"{self.forward_text_to_text() = }")
            loss = loss + float(self.cfg.clip.text_to_text_weight) * self.forward_text_to_text()

        return {"loss": loss, "logits": logits}
        

