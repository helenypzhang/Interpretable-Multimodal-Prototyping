import torch
import torch.nn as nn
import torch.nn.functional as F

from medmm.modeling.ops import SNN_Block, MLP_Block, Attn_Net_Gated, LRBilinearFusion, BilinearFusion


from medmm.modeling.fusion.basefusion import BaseFusion
from .build import MIL_REGISTRY

class ABMIL_MODULE(BaseFusion):
    def __init__(self, 
                 path_input_dim=512, 
                 model_size_wsi: str='small', 
                 dropout: float=0.25):
        super(ABMIL_MODULE, self).__init__()
        

        self.size_dict_WSI = {"small": [path_input_dim, 512, 256], "big": [1024, 512, 384]}

        
        ### Deep Sets Architecture Construction
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])
            
        self._out_features = size[2]


    def forward(self, x_path):
        if len(x_path.shape) == 3:
            x_path = x_path.squeeze()
        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        # import pdb;pdb.set_trace()
        h_path = self.rho(h_path)

        return h_path






    
    
@MIL_REGISTRY.register()
def abmil(**kwargs):
    return ABMIL_MODULE(**kwargs)