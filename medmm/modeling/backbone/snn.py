import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from medmm.modeling.ops import SNN_Block, init_max_weights


class SNN(Backbone):
    def __init__(
        self,
        num_mol: int=1000,
        dropout: int=0.1,
        model_size_omic: str='small',
        **kwargs
    ):
        super().__init__()
        self.num_mol = num_mol
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=num_mol, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        
        self.fc_omic = nn.Sequential(*fc_omic)
        
        self._out_features = hidden[-1]
        init_max_weights(self)
                        
    def forward(self, x):
        out = self.fc_omic(x)
        return out
    
@BACKBONE_REGISTRY.register()
def snn(**kwargs):
    return SNN(**kwargs)
