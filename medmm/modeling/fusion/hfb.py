import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from medmm.modeling.ops import SNN_Block, MLP_Block, Attn_Net_Gated, LRBilinearFusion, BilinearFusion

from .build import FUSION_REGISTRY
from .basefusion import BaseFusion

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(SubNet, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(in_size, hidden_size),nn.Tanh())
        encoder2 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.encoder = nn.Sequential(encoder1, encoder2)
    def forward(self, x):
        y = self.encoder(x)
        return y


class HFB(BaseFusion):
    def __init__(self, 
                 omic_input_dim:int = 250,
                 path_input_dim=512, 
                 hidden_dims=[128, 128, 128, 256], 
                 output_dims=[128, 128, 1], 
                 dropouts=[0.1, 0.1, 0.1, 0.2], 
                 rank: int=20, 
                 fac_drop: float=0.10,
                 ):
        super(HFB, self).__init__()
        
        self.omic_in = omic_input_dim
        self.path_in = path_input_dim


        self.gene_in = omic_input_dim
        self.path_in = path_input_dim
        self.cona_in = omic_input_dim

        self.gene_hidden = hidden_dims[0]
        self.path_hidden = hidden_dims[1]
        self.cona_hidden = hidden_dims[2]
        self.cox_hidden = hidden_dims[3]

        self.output_intra = output_dims[0]
        self.output_inter = output_dims[1]
        self.label_dim = output_dims[2]
        self.rank = rank
        self.factor_drop = fac_drop

        self.gene_prob = dropouts[0]
        self.path_prob = dropouts[1]
        self.cona_prob = dropouts[2]
        self.cox_prob = dropouts[3]

        self.joint_output_intra = self.rank * self.output_intra
        self.joint_output_inter = self.rank * self.output_inter
        self.in_size = self.gene_hidden + self.output_intra + self.output_inter
        self.hid_size = self.gene_hidden


        self.norm = nn.BatchNorm1d(self.in_size)
        self.factor_drop = nn.Dropout(self.factor_drop)
        self.attention = nn.Sequential(nn.Linear((self.hid_size + self.output_intra), 1), nn.Sigmoid())

        self.encoder_gene = SubNet(self.gene_in, self.gene_hidden)
        self.encoder_path = SubNet(self.path_in, self.path_hidden)
        self.encoder_cona = SubNet(self.cona_in, self.cona_hidden)
        
        
        self.attention_net1 = Attn_Net_Gated(L=self.path_hidden, D=self.path_hidden, dropout=0.25, n_classes=1)
        self.attention_net2 = Attn_Net_Gated(L=self.path_hidden, D=self.path_hidden, dropout=0.25, n_classes=1)

        self.Linear_gene = nn.Linear(self.gene_hidden, self.joint_output_intra)
        self.Linear_path = nn.Linear(self.path_hidden, self.joint_output_intra)
        self.Linear_cona = nn.Linear(self.cona_hidden, self.joint_output_intra)

        self.Linear_gene_a = nn.Linear(self.gene_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_path_a = nn.Linear(self.path_hidden + self.output_intra, self.joint_output_inter)
        self.Linear_cona_a = nn.Linear(self.cona_hidden + self.output_intra, self.joint_output_inter)
        
        self.Linear_encoder = nn.Linear(self.in_size*2, self.in_size)

        self._out_features = self.in_size


    def mfb(self, x1, x2, output_dim):

        self.output_dim =  output_dim
        fusion = torch.mul(x1, x2)
        fusion = self.factor_drop(fusion)
        fusion = fusion.view(-1, 1, self.output_dim, self.rank)
        fusion = torch.squeeze(torch.sum(fusion, 3))
        fusion = torch.sqrt(F.relu(fusion)) - torch.sqrt(F.relu(-fusion))
        fusion = F.normalize(fusion)
        return fusion

    def forward(self, x_path, x_omic):
        if len(x_path.shape) == 3:
            x_path = x_path.squeeze()
            
        x_omic_1, x_omic_2 = x_omic[:, :500], x_omic[:, 500:]
        x_omic_1 = x_omic_1.reshape(-1, 250)
        x_omic_2 = x_omic_2.reshape(-1, 250)
        
        gene_feature = self.encoder_gene(x_omic_1)
        path_feature = self.encoder_path(x_path)
        cona_feature = self.encoder_cona(x_omic_2)
        
        
        A, h_path1 = self.attention_net1(path_feature)  
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_path1 = torch.mm(A, h_path1)
        
        A, h_path2 = self.attention_net2(path_feature.clone())  
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1) 
        h_path2 = torch.mm(A, h_path2)  
        
        path_feature = torch.concat([h_path1, h_path2], dim=0)
        

        gene_h = self.Linear_gene(gene_feature)
        path_h = self.Linear_path(path_feature)
        cona_h = self.Linear_cona(cona_feature)

        ######################### modelity-specific###############################
        #intra_interaction#
        intra_gene = self.mfb(gene_h, gene_h, self.output_intra)
        intra_path = self.mfb(path_h, path_h, self.output_intra)
        intra_cona = self.mfb(cona_h, cona_h, self.output_intra)

        gene_x = torch.cat((gene_feature, intra_gene), 1)
        path_x = torch.cat((path_feature, intra_path), 1)
        cona_x = torch.cat((cona_feature, intra_cona), 1)

        sg = self.attention(gene_x)
        sp = self.attention(path_x)
        sc = self.attention(cona_x)

        sg_a = (sg.expand(gene_feature.size(0), (self.gene_hidden + self.output_intra)))
        sp_a = (sp.expand(path_feature.size(0), (self.path_hidden + self.output_intra)))
        sc_a = (sc.expand(cona_feature.size(0), (self.cona_hidden + self.output_intra)))

        gene_x_a = sg_a * gene_x
        path_x_a = sp_a * path_x
        cona_x_a = sc_a * gene_x

        unimodal = gene_x_a + path_x_a + cona_x_a

        ######################### cross-modelity######################################
        g = F.softmax(gene_x_a, 1)
        p = F.softmax(path_x_a, 1)
        c = F.softmax(cona_x_a, 1)

        sg = sg.squeeze()
        sp = sp.squeeze()
        sc = sc.squeeze()

        sgp = (1 / (torch.matmul(g.unsqueeze(1), p.unsqueeze(2)).squeeze() + 0.5) * (sg + sp))
        sgc = (1 / (torch.matmul(g.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sg + sc))
        spc = (1 / (torch.matmul(p.unsqueeze(1), c.unsqueeze(2)).squeeze() + 0.5) * (sp + sc))
        normalize = torch.cat((sgp.unsqueeze(1), sgc.unsqueeze(1), spc.unsqueeze(1)), 1)
        normalize = F.softmax(normalize, 1)
        sgp_a = normalize[:, 0].unsqueeze(1).expand(gene_feature.size(0), self.output_inter)
        sgc_a = normalize[:, 1].unsqueeze(1).expand(path_feature.size(0), self.output_inter)
        spc_a = normalize[:, 2].unsqueeze(1).expand(cona_feature.size(0), self.output_inter)


        # inter_interaction#
        gene_l = self.Linear_gene_a(gene_x_a)
        path_l = self.Linear_gene_a(path_x_a)
        cona_l = self.Linear_gene_a(cona_x_a)

        inter_gene_path = self.mfb(gene_l, path_l, self.output_inter)
        inter_gene_cona = self.mfb(gene_l, cona_l, self.output_inter)
        inter_path_cona = self.mfb(path_l, cona_l, self.output_inter)

        bimodal = sgp_a * inter_gene_path + sgc_a * inter_gene_cona + spc_a * inter_path_cona
        ############################################### fusion layer ###################################################
        # import pdb;pdb.set_trace()
        fusion = torch.cat((unimodal, bimodal), 1)
        fusion = self.norm(fusion)
        fusion = torch.cat((fusion[:1, :], fusion[1:, :]), 1)
        fusion = self.Linear_encoder(fusion)

        return fusion


    
@FUSION_REGISTRY.register()
def hfb(**kwargs):
    return HFB(**kwargs)