import os.path as osp
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from medmm.modeling.ops import SNN_Block, Attn_Net_Gated


from clip import clip




class OmicEncoder(nn.Module):
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
    

class CLIPOMIC(Backbone):
    def __init__(
        self,
        clip_model,
        num_mol: int=5,
        len_mol: int=200,
        model_size_omic: str='small',
        dropout: float=0.25,
        **kwargs
    ):
        super().__init__()

        self.num_mol = num_mol
        # self.len_mol = len_mol
        self.omic_encoder = OmicEncoder(clip_model)
        dtype = clip_model.dtype
        
        # prompt_prefix = " ".join(["X"] * len_mol)
        prompts = ["X" +  "." for i in range(num_mol)]

        # import pdb;pdb.set_trace()
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 2 :, :])  # CLS, EOS    
        self.tokenized_prompts = tokenized_prompts
        
        self.omic_fc  = SNN_Block(len_mol, embedding.shape[-1])
        
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.size_dict_OMIC = {"small": [ctx_dim, 512, 256], "big": [ctx_dim, 512, 384]}

        
        ### Deep Sets Architecture Construction
        size = self.size_dict_OMIC[model_size_omic]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
            
        self._out_features = size[2]
                        
    def forward(self, x_omic):
        x_omic = x_omic.reshape(self.num_mol, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.omic_fc(x_omic).unsqueeze(1) 
        # import pdb;pdb.set_trace()
        
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        h_omic = self.omic_encoder(prompts, self.tokenized_prompts)
        
        
        A, h_omic = self.attention_net(h_omic)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_omic = torch.mm(A, h_omic)
        # import pdb;pdb.set_trace()
        h_omic = self.rho(h_omic)
        
        return h_omic



@BACKBONE_REGISTRY.register()
def clipomic(**kwargs):
    return CLIPOMIC(**kwargs)







    


        
 