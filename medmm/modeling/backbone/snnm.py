import torch
import torch.nn as nn
from torch.nn import functional as F


from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from medmm.modeling.ops import SNN_Block, init_max_weights, Attn_Net_Gated


class SNN_M(Backbone):
    def __init__(
        self,
        num_mol: int=50,
        dropout: int=0.1,
        model_size_omic: str='small',
        **kwargs
    ):
        super().__init__()
        self.num_mol = num_mol
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=num_mol, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        
        self.fc_omic = nn.Sequential(*fc_omic)
        
    
        fc = [nn.Linear(hidden[1], hidden[2]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=hidden[2], D=hidden[3], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(hidden[2], hidden[3]), nn.ReLU(), nn.Dropout(dropout)])
        
        
        
        self._out_features = hidden[-1]
        init_max_weights(self)
                        
    def forward(self, x_omic):
        x_omic =  x_omic.reshape(-1, self.num_mol)
        # import pdb;pdb.set_trace()
        h_omic = self.fc_omic(x_omic)
        A, h_omic = self.attention_net(h_omic)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_omic = torch.mm(A, h_omic)
        # import pdb;pdb.set_trace()
        h_omic = self.rho(h_omic)
        return h_omic
    
@BACKBONE_REGISTRY.register()
def snnm(**kwargs):
    return SNN_M(**kwargs)
