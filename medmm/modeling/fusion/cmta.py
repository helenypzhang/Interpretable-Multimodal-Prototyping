import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from medmm.modeling.ops import (SNN_Block, MultiheadAttention, BilinearFusion, 
                                Transformer_P, Transformer_G, initialize_weights)

from .build import FUSION_REGISTRY
from .basefusion import BaseFusion

class CMTA(BaseFusion):
    def __init__(self, 
                 fusion: str='concat', 
                 path_input_dim=512, 
                 omic_sizes=[200, 200, 200, 200, 200],   # [100, 200, 300, 400, 500, 372],
                 model_size="small",
                 dropout: float=0.25):
        super(CMTA, self).__init__()
        
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.size_dict = {
            "pathomics": {"small": [path_input_dim, 256, 256], "large": [path_input_dim, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(dropout))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])

        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
            
        self._out_features = hidden[-1]

        self.apply(initialize_weights)

    def forward(self, x_path, x_omic):
        omic_sizes = self.omic_sizes
        x_omics =  [x_omic[:, sum(omic_sizes[:i]):sum(omic_sizes[:i+1])] for i in range(len(omic_sizes))]
        
        # import pdb;pdb.set_trace()
        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omics)]
        genomics_features = torch.stack(genomics_features).transpose(0,1)  # [1, 6, 256]
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path) #[1, N, 256]

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # cross-omics attention
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        # pathomics decoder
        cls_token_pathomics_decoder, _ = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        # genomics decoder
        cls_token_genomics_decoder, _ = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        # fusion
        if self.fusion == "concat":
            fusion = self.mm(
                torch.concat(
                    (
                        (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                        (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
                    ),
                    dim=1,
                )
            )  # take cls token to make prediction
        elif self.fusion == "bilinear":
            fusion = self.mm(
                (cls_token_pathomics_encoder + cls_token_pathomics_decoder) / 2,
                (cls_token_genomics_encoder + cls_token_genomics_decoder) / 2,
            )  # take cls token to make prediction
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))
        

        
        cls_tokens = {'cls_token_pathomics_encoder': cls_token_pathomics_encoder,
                      'cls_token_pathomics_decoder': cls_token_pathomics_decoder, 
                      'cls_token_genomics_encoder': cls_token_genomics_encoder,
                      'cls_token_genomics_decoder': cls_token_genomics_decoder}
        
        return fusion, cls_tokens


    
@FUSION_REGISTRY.register()
def cmta(**kwargs):
    return CMTA(**kwargs)