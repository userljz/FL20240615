import torch
import torch.nn as nn
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, text_encoder_model):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = cfg.clip.ctx_init
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.cfg = cfg
        self.device = cfg.device.cuda
        
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(dtype)  # [batch_size, (1+ n_ctx + *), ctx_dim]
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
        prompt_prefix = ctx_init
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.construct_references_lasp(cfg, clip_model, text_encoder_model, classnames, prompt_prefix, dtype, n_ctx)
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.all_classnames = classnames
        
        if cfg.clip.text_correction:
            self.w = nn.Parameter(torch.zeros(1, ctx_dim, device=embedding.device, dtype=dtype),
                                  requires_grad=self.cfg.clip.text_correction)
    
    def construct_references_lasp(self, cfg, clip_model, text_encoder_model, all_classnames, prompt_prefix, dtype,
                                  n_ctx):
        print('Initializing LASP prompts...')
        template_prompts = cfg.clip.templete_prompts
        all_classnames = [name.replace("_", " ") for name in all_classnames]
        print(f'Num classes used for LASP: {len(all_classnames)}')
        
        all_class_text_features = []
        for c_init in template_prompts:
            prompts = [c_init.format(name) for name in all_classnames]
            tokenized_prompts_all_c = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
            text_encoder_model.to(self.device)
            with torch.no_grad():
                embedding_all_cls = clip_model.token_embedding(tokenized_prompts_all_c).to(self.device).type(dtype)
                class_text_features = text_encoder_model(embedding_all_cls, tokenized_prompts_all_c).type(dtype)
                all_class_text_features.append(class_text_features)
            
            self.register_buffer("class_text_features", torch.stack(all_class_text_features, dim=0))
        
        prompts = [prompt_prefix + " " + name + "." for name in all_classnames]
        tokenized_prompts_all_c_ = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts_all_c_).type(dtype)
        
        self.register_buffer("token_prefix_all", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_all", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.tokenized_prompts_all = tokenized_prompts_all_c
        self.tokenized_prompts_all_c_ = tokenized_prompts_all_c_
        self.n_cls_all = len(prompts)
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )
        
        return prompts
    
    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        n_cls = self.n_cls
        ctx = self.ctx  # (n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        ctx = ctx.unsqueeze(0).expand(n_cls, -1, -1)
        prompts = self.construct_prompts(ctx, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
        
        return prompts

