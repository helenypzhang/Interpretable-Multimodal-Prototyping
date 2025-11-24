import torch
import torch.nn as nn
import torch.nn.functional as F

from medmm.modeling.ops import SNN_Block, MLP_Block, Attn_Net_Gated, LRBilinearFusion, BilinearFusion

from .build import FUSION_REGISTRY
from .basefusion import BaseFusion

class Porpoise(BaseFusion):
    def __init__(self, 
                 fusion: str='bilinear',
                 omic_input_dim:int = 50,
                 path_input_dim=512, 
                 model_size_wsi: str='small', 
                 model_size_omic: str='small',
                 scale_dim1=8, 
                 scale_dim2=8, 
                 gate_path=1, 
                 gate_omic=1, 
                 skip=True, 
                 dropinput=0.10,
                 use_mlp=False, 
                 dropout: float=0.1):
        super(Porpoise, self).__init__()
        
        
        self.omic_input_dim = omic_input_dim
        self.fusion = fusion
        self.size_dict_WSI = {"small": [path_input_dim, 512, 256], "big": [path_input_dim, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Deep Sets Architecture Construction
        size = self.size_dict_WSI[model_size_wsi]
        if dropinput:
            fc_h = [nn.Dropout(dropinput), nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        else:
            fc_h = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net_h = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc_h.append(attention_net_h)
        self.attention_net_h = nn.Sequential(*fc_h)
        self.rho_h = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        if self.fusion is not None:
            if use_mlp:
                Block = MLP_Block
            else:
                Block = SNN_Block
     

        fc_omic = [Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        
        fc_o = [nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net_o = Attn_Net_Gated(L=hidden[0], D=hidden[1], dropout=dropout, n_classes=1)
        fc_o.append(attention_net_o)
        self.attention_net_o = nn.Sequential(*fc_o)
        self.rho_o = nn.Sequential(*[nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout)])
        
        
    
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic, skip=skip, mmhid=256)
        elif self.fusion == 'lrb':
            self.mm = LRBilinearFusion(dim1=256, dim2=256, scale_dim1=scale_dim1, gate1=gate_path, scale_dim2=scale_dim2, gate2=gate_omic)
        else:
            self.mm = None
            
        self._out_features = hidden[-1]


    def forward(self, x_path, x_omic):
        if len(x_path.shape) == 3:
            x_path = x_path.squeeze()
            
        A, h_path = self.attention_net_h(x_path)  
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho_h(h_path)
        
        x_omic =  x_omic.reshape(-1, self.omic_input_dim)
        h_omic = self.fc_omic(x_omic)
        A, h_omic = self.attention_net_o(h_omic)  
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_omic = torch.mm(A, h_omic)
        h_omic = self.rho_o(h_omic)        
        
        
        if self.fusion == 'bilinear':
            h_mm = self.mm(h_path, h_omic)
        elif self.fusion == 'concat':
            h_mm = self.mm(torch.cat([h_path, h_omic], axis=1))
        elif self.fusion == 'lrb':
            h_mm  = self.mm(h_path, h_omic) # logits needs to be a [1 x 4] vector 
            return h_mm
        elif self.fusion == 'add':
            h_mm = h_path + h_omic

        return h_mm

    # def captum(self, h, X):
    #     A, h = self.attention_net(h)  
    #     A = A.squeeze(dim=2)

    #     A = F.softmax(A, dim=1) 
    #     M = torch.bmm(A.unsqueeze(dim=1), h).squeeze(dim=1) #M = torch.mm(A, h)
    #     M = self.rho(M)
    #     O = self.fc_omic(X)

    #     if self.fusion == 'bilinear':
    #         MM = self.mm(M, O)
    #     elif self.fusion == 'concat':
    #         MM = self.mm(torch.cat([M, O], axis=1))
            
    #     logits  = self.classifier(MM)
    #     hazards = torch.sigmoid(logits)
    #     S = torch.cumprod(1 - hazards, dim=1)

    #     risk = -torch.sum(S, dim=1)
    #     return risk



    
    
@FUSION_REGISTRY.register()
def porpoise(**kwargs):
    return Porpoise(**kwargs)