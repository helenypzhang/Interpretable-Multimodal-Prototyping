import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from medmm.modeling.ops import TransLayer, PPEG


from medmm.modeling.fusion.basefusion import BaseFusion
from .build import MIL_REGISTRY

class TransMIL_MODULE(BaseFusion):
    def __init__(self, 
                 path_input_dim=512,
                 ):
        super(TransMIL_MODULE, self).__init__()
        

        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(path_input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
            
        self._out_features = 512


    def forward(self, x_path):

        # import pdb;pdb.set_trace()    
        h_path = self._fc1(x_path) #[B, n, 512]
        
        #---->pad
        H = h_path.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h_path = torch.cat([h_path, h_path[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h_path.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h_path = torch.cat((cls_tokens, h_path), dim=1)

        #---->Translayer x1
        h_path = self.layer1(h_path) #[B, N, 512]

        #---->PPEG
        h_path = self.pos_layer(h_path, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h_path = self.layer2(h_path) #[B, N, 512]

        #---->cls_token
        h_path = self.norm(h_path)[:,0]

        return h_path






    
    
@MIL_REGISTRY.register()
def transmil(**kwargs):
    return TransMIL_MODULE(**kwargs)