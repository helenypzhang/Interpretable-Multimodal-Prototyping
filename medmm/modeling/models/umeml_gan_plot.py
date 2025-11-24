import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import torch.utils.checkpoint as checkpoint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from extract_prototype_with_plip_train import get_path_prorotypes

from medmm.modeling.ops import (SNN_Block, MultiheadAttention, TransLayer, BilinearFusion, 
                                compute_modularity)

from .build import MODEL_REGISTRY
from .base import Base

import pandas as pd

def reset(x, n_c): x.data.uniform_(-1.0 / n_c, 1.0 / n_c)

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.total_input_features = input_dim[0] * input_dim[1]
        self.total_output_features = output_dim[0] * output_dim[1]

        self.net = nn.Sequential(nn.Linear(self.total_input_features, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, self.total_output_features),
                                nn.Softplus())

    def forward(self, x):
        if x.shape[-2:] != self.input_dim:
            raise ValueError(f"Expected input shape (-2 dimensions): {self.input_dim}, but got: {x.shape[-2:]}")
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.net(x)
        x = x.view(batch_size, self.output_dim[0], self.output_dim[1])
        return x


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        input_features = input_shape[0] * input_shape[1]

        self.layers = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class PathProtoGenerator(nn.Module):
    def __init__(
            self,
            dim: int,
            drop_path: float = 0.,
    ) -> None:
        super().__init__()
        self.cross_attn = MultiheadAttention(embed_dim=dim, num_heads=1)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        _c, attn = self.cross_attn(c.transpose(1, 0), x.transpose(1, 0), x.transpose(1, 0),)  # ([5, 1, 256])
        _c = _c.transpose(1, 0)
        c = c + self.drop_path1(self.norm1(_c))
        return c





class Block(nn.Module):
    def __init__(
            self,
            dim: int,
    ) -> None:
        super().__init__()
        self.attn = TransLayer(dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        return x


    
class BottleneckAttentionBlock(nn.Module):
    def __init__(
            self,
            dim: int=256,
            n_reg: int=2,
    ) -> None:
        super().__init__() 
        self.bottle_tokens = nn.Parameter(torch.FloatTensor(1, n_reg, dim), requires_grad=True)
        nn.init.uniform_(self.bottle_tokens)
        self.encoders =  nn.ModuleList([
           Block(dim=dim)
            for i in range(2)])
        
        
    def forward(self, x_path: torch.Tensor, x_omic: torch.Tensor, patient_id=None):
        # import pdb;pdb.set_trace()
        path_len,  omic_len = x_path.size()[1],  x_omic.size()[1]
        
        token_len = self.bottle_tokens.size()[1]

        K = 3
        x = []
            
        for index in range(x_path.shape[0]):
            x_path_this = x_path[index]
            x_omic_this = x_omic[index]
            sim = np.zeros(shape=(x_path.shape[1], x_omic.shape[1]))
            for i_p in range(x_path.shape[1]):
                for i_o in range(x_omic.shape[1]):
                    sim[i_p, i_o] = F.cosine_similarity(x_path[index, i_p], x_omic[index, i_o], dim=0).item()
            
            if patient_id is not None:
                save_path = f"plots/sim_{patient_id[index]}.png"
            else:
                save_path = f"plots/sim_{index}.png"

            sim_np = sim
            sim_np = (sim_np - sim_np.min()) / (sim_np.max() - sim_np.min() + 1e-6)
            sim_np = sim_np * 0.5 + 0.5 

            # 设置绘图参数
            gap = 5         # 透明间隔的像素宽度
            block_size = 20 # 每个方块大小
            fig_size = (block_size + gap) * sim_np.shape[1] / 100

            fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=100)

            # 绘制每个方块
            for i in range(sim_np.shape[0]):
                for j in range(sim_np.shape[1]):
                    color_intensity = sim_np[i, j]
                    rect = patches.Rectangle(
                        (j * (block_size + gap), i * (block_size + gap)),
                        block_size, block_size,
                        linewidth=0,
                        edgecolor=None,
                        facecolor=plt.cm.Blues(color_intensity),
                    )
                    ax.add_patch(rect)

            # 设置边界
            ax.set_xlim(0, sim_np.shape[1] * (block_size + gap))
            ax.set_ylim(0, sim_np.shape[0] * (block_size + gap))
            ax.invert_yaxis()  # 上下反转，使(0,0)在左上角
            ax.axis('off')     # 不显示坐标轴
            fig.patch.set_alpha(0.0)
            ax.set_facecolor((0, 0, 0, 0))

            # 保存透明背景图
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
            plt.close(fig)

            # 获取按相似度排序的索引
            sorted_indices = np.argsort(sim, axis=None)[::-1]
            selected_A = set()
            selected_B = set()
            top_pairs = []
            for idx in sorted_indices:
                i_p, i_o = np.unravel_index(idx, sim.shape)
                if i_p not in selected_A and i_o not in selected_B:
                    top_pairs.append((i_p, i_o))
                    selected_A.add(i_p)
                    selected_B.add(i_o)

                if len(top_pairs) == 3:
                    break

            ks = []
            i_p_list = []
            i_o_list = []
            for i, (i_p, i_o) in enumerate(top_pairs):
                i_p_list.append(i_p)
                i_o_list.append(i_o)
                it = self.linear_p(torch.unsqueeze(x_path[index, i_p], 0)) + self.linear_o(torch.unsqueeze(x_omic[index, i_o], 0))
                it = torch.unsqueeze(it, 1)
                ks.append(it)
            i_p_list = sorted(i_p_list)
            i_o_list = sorted(i_o_list)
            ks = torch.cat(ks, dim=1)
            x_path_remain = []
            for y in range(len(i_p_list)):
                if y == 0:
                    x_path_remain += x_path_this[: i_p_list[y]]
                else:
                    x_path_remain += x_path_this[i_p_list[y - 1] + 1: i_p_list[y]]
            x_path_remain += x_path_this[i_p_list[-1] + 1:]
            x_path_remain = torch.stack(x_path_remain)
            remaining_p = torch.unsqueeze(x_path_remain, 0)
            x_omic_remain = []
            for y in range(len(i_o_list)):
                if y == 0:
                    x_omic_remain += x_omic_this[: i_o_list[y]]
                else:
                    x_omic_remain += x_omic_this[i_o_list[y - 1] + 1: i_o_list[y]]
            x_omic_remain += x_omic_this[i_o_list[-1] + 1:]
            x_omic_remain = torch.stack(x_omic_remain)
            remaining_o = torch.unsqueeze(x_omic_remain, 0)
            bottle_this = self.bottle_tokens
            x_this = torch.cat((ks, remaining_p, bottle_this, remaining_o), dim=1)
            x.append(x_this)
        x = torch.cat(x, dim=0)


        # x = torch.concat([x_path, self.bottle_tokens.repeat(x_path.shape[0], 1, 1), x_omic], dim=1)
        for blk in self.encoders:
            x = blk(x)
        t_path, x_path = x[:, :1, :], x[:,  1 :(path_len), :]
        t_omic, x_omic = x[:, (path_len + token_len): (path_len + token_len + 1), : ], x[:, (path_len + token_len + 1): , : ]
        return t_path, x_path, t_omic, x_omic
                 

class UMEML_GAN(Base):
    def __init__(self,
                 cfg, 
                 num_classes,
                 omic_sizes,
                 ):
        super(UMEML_GAN, self).__init__()

        self.cfg = cfg
        
        # GAN在这里定义
        self.gan_generator_p2o = Generator(input_dim=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM), output_dim=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM))
        self.gan_generator_o2p = Generator(input_dim=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM), output_dim=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM))
        self.gan_discriminator_o = Discriminator(input_shape=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM))
        self.gan_discriminator_p = Discriminator(input_shape=(self.cfg.MODEL.UMEML.PROTOTYPES + 1, self.cfg.MODEL.HIDDEN_DIM))
        self.gan_opt_gen = torch.optim.Adam(list(self.gan_generator_p2o.parameters()) + list(self.gan_generator_o2p.parameters()), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
        self.gan_opt_dis_o = torch.optim.Adam(self.gan_discriminator_o.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
        self.gan_opt_dis_p = torch.optim.Adam(self.gan_discriminator_p.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

        self.root = osp.abspath(osp.expanduser(self.cfg.DATASET.ROOT))
        
        dropout = self.cfg.MODEL.DROPOUT
        path_input_dim = self.cfg.DATASET.PATH.DIM
        self.omic_input_dim = omic_input_dim = self.cfg.DATASET.OMIC.DIM
        hidden_dim = self.cfg.MODEL.HIDDEN_DIM  # 256
        projection_dim = self.cfg.MODEL.PROJECT_DIM  # 256
        self.fusion = self.cfg.MODEL.FUSION
        self.size = self.cfg.MODEL.SIZE
        
        self.n_proto = self.cfg.MODEL.UMEML.PROTOTYPES 
        self.n_reg = self.cfg.MODEL.UMEML.REGISTERS 
        
        

        p_fc = [nn.Linear(path_input_dim, hidden_dim), nn.ReLU()]
        p_fc.append(nn.Dropout(dropout))
        self.path_net = nn.Sequential(*p_fc)
        
        # o_fc = [nn.Linear(omic_input_dim, hidden_dim), nn.ReLU()]
        # o_fc.append(nn.Dropout(dropout))
        # self.omic_net = nn.Sequential(*o_fc)
        omic_input_dims = [1, 22, 13, 36, 51, 33]
        self.omic_net = nn.ModuleList()
        for i in range(self.n_proto):
            self.omic_net.append(
                nn.Sequential(
                    nn.Linear(omic_input_dims[i], hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        g_o_fc = [nn.Linear(1000, hidden_dim), nn.ReLU()]
        g_o_fc.append(nn.Dropout(dropout))
        self.g_omic_net = nn.Sequential(*g_o_fc)
        
        self.proto_g_blocks = nn.ModuleList([
            PathProtoGenerator(dim=hidden_dim)
            for i in range(2)])

        omic_encoder = nn.ModuleList([
            Block(dim=hidden_dim)
            for i in range(2)])
        self.omic_encoder = nn.Sequential(*omic_encoder)

        self.layer_norm_p = nn.LayerNorm(hidden_dim)
        self.layer_norm_o = nn.LayerNorm(hidden_dim)

        self.path_decoder = TransLayer(dim=hidden_dim)
        self.omic_decoder = TransLayer(dim=hidden_dim) 
           
        self.bottleattn = BottleneckAttentionBlock(dim=hidden_dim, n_reg=self.n_reg)

        # 用于映射topk的
        self.bottleattn.linear_p = nn.Linear(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)
        self.bottleattn.linear_o = nn.Linear(self.cfg.MODEL.HIDDEN_DIM, self.cfg.MODEL.HIDDEN_DIM)
        
        p_proto = get_path_prorotypes()
        self.p_proto = p_proto
        # self.p_encoder_token = p_encoder_token

        # self.p_proto = nn.Parameter(torch.empty(1, self.n_proto, hidden_dim))
        reset(self.p_proto, self.n_proto)
        
        self.p_encoder_token = nn.Parameter(torch.FloatTensor(1, 1, hidden_dim), requires_grad=True)
        nn.init.uniform_(self.p_encoder_token)
        
        self.o_encoder_token = nn.Parameter(torch.FloatTensor(1, 1, hidden_dim), requires_grad=True)
        nn.init.uniform_(self.o_encoder_token)    

        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=hidden_dim, dim2=hidden_dim, scale_dim1=8, scale_dim2=8, mmhid=hidden_dim)
        else:
            self.mm = None
            
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self.lambda_cyc = 10

        self.l1_loss = nn.L1Loss()
        self.dis_loss = nn.BCELoss()

        self.train_gan = False
        self.replace_ratio = 0

        # 用于可视化每个prototype的重要性分数
        self.explainer_path = nn.Linear(hidden_dim, num_classes, bias=False)
        self.explainer_omic = nn.Linear(hidden_dim, num_classes, bias=False)

        self.plot_set = "train"

        import pandas as pd
        signature_path = "DATASET/tcga_glioma/labels/signatures.csv"
        signature_df = pd.read_csv(signature_path)
        tumor_list = signature_df["Tumor Suppressor Genes"].dropna().tolist()
        oncogenes_list = signature_df["Oncogenes"].dropna().tolist()
        protein_list = signature_df["Protein Kinases"].dropna().tolist()
        cell_list = signature_df["Cell Differentiation Markers"].dropna().tolist()
        transcription_list = signature_df["Transcription Factors"].dropna().tolist()
        cytokines_list = signature_df["Cytokines and Growth Factors"].dropna().tolist()
        gene_categories = {
            "Tumor_Suppressor": tumor_list,
            "Oncogenes": oncogenes_list,
            "Protein_Kinases": protein_list,
            "Cell_Markers": cell_list,
            "Transcription_Factors": transcription_list,
            "Cytokines_Factors": cytokines_list,
        }
        self.gene_categories = [tumor_list, oncogenes_list, protein_list, cell_list, transcription_list, cytokines_list]
        omic_df = pd.read_csv("DATASET/tcga_glioma/molecular/TCGA-02-0047-01A-01-BS1.csv")
        self.gene_group_indexes = []
        for gene_category in self.gene_categories:
            matched_index = omic_df.index[omic_df["gene_name"].isin(gene_category)]
            self.gene_group_indexes.append(matched_index.tolist())

    def adversarial_loss(self, D, fake_images):
        return nn.BCEWithLogitsLoss()(D(fake_images), torch.ones_like(D(fake_images)))

    def forward(self, batch, is_survival=True, T=5.0):


        os.makedirs("plots", exist_ok=True)

        if not 'insample_without_omic' in batch.keys():
            insample_without_omic = torch.zeros(batch['omic'].shape)
        else:
            insample_without_omic = batch['insample_without_omic']

        x_path = batch['img']
        try:
            x_omic = batch['omic']
        except:
            x_omic = None
        
        if x_omic is not None and ('insample_without_omic' in batch.keys() and torch.sum(batch['insample_without_omic']) > 0):
            x_omic = torch.where(insample_without_omic.bool(), self.omic_means.to(x_omic.device).unsqueeze(0).repeat(x_omic.shape[0], 1), x_omic)

        bsz = x_path.shape[0]
        if x_omic is not None:
            _, N = x_omic.size()
            x_omic = x_omic.reshape(bsz, -1, N)
            g_omic = x_omic.detach().clone()
        
            x_omic =  x_omic.reshape(bsz, -1, self.omic_input_dim)
        h_path_bag = []
        for i in range(x_path.shape[0]):
            data = x_path[i]
            indices = torch.nonzero(x_path[i] == -10000)
            if len(indices) > 0:
                # 只取第一个出现的位置
                first_index = indices[0][0].item()  # 获取行索引
                # 截取从 0 到 first_index 的部分
                x_path_this = data[:first_index]
            h_path_bag.append(self.path_net(torch.unsqueeze(x_path_this, 0)))
        # h_path_bag = 
        # h_path_bag = x_path
        if x_omic is not None:
            omic_inputs = []
            for list_ in self.gene_group_indexes:
                x_omic_group = x_omic[:, :, list_]
                omic_inputs.append(x_omic_group)
            h_omic_bag_list = [net(x) for net, x in zip(self.omic_net, omic_inputs)]
            h_omic_bag = torch.cat(h_omic_bag_list, dim=1)
            # h_omic_bag = self.omic_net(x_omic)
            g_omic = self.g_omic_net(g_omic)
            # h_omic_bag = torch.concat([h_omic_bag, g_omic], dim=1)

        # import pdb;pdb.set_trace()
        for i, blk in enumerate(self.proto_g_blocks):
            p_proto_list = []
            if i == 0:
                for item in h_path_bag:
                    p_proto_list.append(blk(item, self.p_proto))
                p_proto = torch.cat(p_proto_list, dim=0)
            else:
                for j in range(len(h_path_bag)):
                    p_proto_list.append(blk(h_path_bag[j], torch.unsqueeze(p_proto[j], 0)))
                p_proto = torch.cat(p_proto_list, dim=0)

        # 计算两个模态的一致性 p_proto, h_omic_bag
        # 需要输出出去 p_proto_before, h_omic_bag_before
        p_proto_before = p_proto
        try:
            h_omic_bag_before = h_omic_bag
        except:
            h_omic_bag_before = None
        if x_omic is not None:
            h_omic = torch.concat([self.o_encoder_token.repeat(p_proto.shape[0], 1, 1), h_omic_bag], dim=1)       
                
            h_omic = self.omic_encoder(h_omic)
      
        h_path = torch.concat([self.p_encoder_token.repeat(p_proto.shape[0], 1, 1), p_proto], dim=1)
        
        h_path = self.path_decoder(h_path)
        if x_omic is not None:
            h_omic = self.omic_decoder(h_omic)
        
        h_path = self.layer_norm_p(h_path)
        if x_omic is not None:
            h_omic = self.layer_norm_o(h_omic)

        if self.cca:
            return h_path, h_omic, p_proto_before, h_omic_bag_before, 'cca'

        if self.training and self.train_gan:
            fake_omic = self.gan_generator_p2o(h_path.clone())
            fake_path = self.gan_generator_o2p(h_omic.clone())
            cycle_path = self.gan_generator_o2p(fake_omic)
            cycle_omic = self.gan_generator_p2o(fake_path)

            gan_p2o_loss = self.adversarial_loss(self.gan_discriminator_o, fake_omic)
            gan_o2p_loss = self.adversarial_loss(self.gan_discriminator_p, fake_path)
            gan_cycle_o_loss = self.l1_loss(cycle_omic, h_omic.clone())
            gan_cycle_p_loss = self.l1_loss(cycle_path, h_path.clone())
            gen_loss = (gan_p2o_loss + gan_o2p_loss) + self.lambda_cyc * (gan_cycle_o_loss + gan_cycle_p_loss)
            self.gan_opt_gen.zero_grad()
            gen_loss.backward(retain_graph=True)
            self.gan_opt_gen.step()

            fake_path = self.gan_generator_o2p(h_omic.clone())
            pred_p = torch.cat((self.gan_discriminator_p(h_path.clone()), self.gan_discriminator_p(fake_path)), dim=0)
            labels_p = torch.cat((torch.ones(size=(pred_p.shape[0] // 2, 1)), torch.zeros(size=(pred_p.shape[0] // 2, 1))), dim=0).to(pred_p.device)
            dis_p_loss = self.dis_loss(pred_p, labels_p)
            self.gan_opt_dis_p.zero_grad()
            dis_p_loss.backward(retain_graph=True)
            self.gan_opt_dis_p.step()

            fake_omic = self.gan_generator_p2o(h_path.clone())
            pred_o = torch.cat((self.gan_discriminator_o(h_omic.clone()), self.gan_discriminator_o(fake_omic)), dim=0)
            labels_o = torch.cat((torch.ones(size=(pred_o.shape[0] // 2, 1)), torch.zeros(size=(pred_o.shape[0] // 2, 1))), dim=0).to(pred_p.device)
            dis_o_loss = self.dis_loss(pred_o, labels_o)
            self.gan_opt_dis_o.zero_grad()
            dis_o_loss.backward(retain_graph=True)
            self.gan_opt_dis_o.step()

        if self.training and self.replace_ratio > 0:
            fake_omic = self.gan_generator_p2o(h_path)
            random_vector = np.random.uniform(0, 1, h_omic.shape[0])
            for i in range(h_omic.shape[0]):
                if random_vector[i] > self.replace_ratio:
                    h_omic[i] = fake_omic[i]

        # if x_omic is None:
        if x_omic is None or ('without_omic' in batch.keys() and torch.sum(batch['without_omic']) > 0) or ('insample_without_omic' in batch.keys() and torch.sum(batch['insample_without_omic']) > 0):
            h_omic_gen = self.gan_generator_p2o(h_path)

        if x_omic is not None and ('without_omic' in batch.keys() and torch.sum(batch['without_omic']) > 0):
            index_bool = (batch["without_omic"] == 1).view(-1, 1, 1)
            h_omic = torch.where(index_bool, h_omic_gen, h_omic)
        elif x_omic is None:
            h_omic = h_omic_gen

        if x_omic is not None and ('insample_without_omic' in batch.keys() and torch.sum(batch['insample_without_omic']) > 0):
            gen_ratio = torch.sum(batch['insample_without_omic']) / batch['insample_without_omic'].numel()
            h_omic = (1 - gen_ratio) * h_omic + gen_ratio * h_omic_gen


        t_path, f_path, t_omic, f_omic = self.bottleattn(h_path, h_omic, batch["patient_id"])

        if self.training:
            modular_1 = []
            modular_2 = []
            for j in range(len(h_path_bag)):  
                modular_1.append(compute_modularity(torch.unsqueeze(p_proto[j], 0), h_path_bag[j], grid=False))
                modular_2.append(compute_modularity(torch.unsqueeze(h_omic[j], 0), h_path_bag[j], grid=False))
            modular_1 = torch.stack(modular_1)
            modular_2 = torch.stack(modular_2)
            modular_1 = torch.mean(modular_1)
            modular_2 = torch.mean(modular_2)
            modular_loss =  (modular_1 +  modular_2)

        else:
            modular_loss = 0
            
            
        if self.fusion == 'bilinear':
            h = []
            for j in range(t_path.shape[0]):
                h.append(self.mm(t_path[j], t_omic[j]))
            h = torch.cat(h, dim=0)
        elif self.fusion == 'concat':
            h = []
            for j in range(t_path.shape[0]):
                h.append(self.mm(torch.cat([t_path[j], t_omic[j]], axis=1)))
            h = torch.cat(h, dim=0)

        logits = self.classifier(h)
        
        # if self.training: 
        #     if self.train_gan:
        #         return logits, modular_loss, gen_loss, dis_p_loss, dis_o_loss
        #     else:
        #         return logits, modular_loss, 0, 0, 0
        # else:
        #     return logits

        # h_path 和 h_omic: [B, N_proto, D]
        B, N_proto, D = h_path.shape

        logits_path_proto = self.explainer_path(h_path)      # [B, N_proto, num_classes]
        logits_omic_proto = self.explainer_omic(h_omic)      # [B, N_proto, num_classes]

        logits_path = logits_path_proto.mean(dim=1)          # [B, num_classes]
        logits_omic = logits_omic_proto.mean(dim=1)          # [B, num_classes]

        # 融合后得到最终解释 logits
        logits_explained = (logits_path + logits_omic) / 2

        # 比如当前预测类别为 c_pred
        pred_class = logits_explained.argmax(dim=1)  # [B]

        # 提取每个样本中，每个 prototype 对预测类别的贡献值
        importance_path = torch.gather(logits_path_proto, dim=2, index=pred_class.view(B, 1, 1).expand(B, N_proto, 1)).squeeze(-1)
        importance_omic = torch.gather(logits_omic_proto, dim=2, index=pred_class.view(B, 1, 1).expand(B, N_proto, 1)).squeeze(-1)

        importance_path_ = transform_importance(importance_path)[:, :importance_path.shape[1] - 1]
        importance_omic_ = transform_importance(importance_omic)[:, :importance_omic.shape[1] - 1]

        # 写入 path importance
        path_file = self.plot_set + "_path.txt"
        with open(path_file, "a") as f:
            for row in importance_path_:
                row_str = " ".join(map(str, row.tolist()))
                f.write(row_str + "\n")

        # 写入 omic importance
        omic_file = self.plot_set + "_omic.txt"
        with open(omic_file, "a") as f:
            for row in importance_omic_:
                row_str = " ".join(map(str, row.tolist()))
                f.write(row_str + "\n")

        with torch.no_grad():
            logits_teacher = logits.detach()

        logits_student = logits_explained

        loss_kd = F.kl_div(
            F.log_softmax(logits_student / T, dim=1),
            F.softmax(logits_teacher / T, dim=1),
            reduction='batchmean'
        ) * (T * T)

        gap = 5
        block_size = 20

        # # 可视化
        B, N = importance_path_.shape

        for idx in range(B):
            for mod_name, importance in zip(["path", "omic"], [importance_path_[idx], importance_omic_[idx]]):
                # 图像尺寸（单位：英寸）
                fig_w = (block_size + gap) / 100
                fig_h = (block_size + gap) * N / 100
                fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)

                for i in range(N):
                    intensity = importance[i]

                    # 根据模态选择 colormap：path=蓝色，omic=橙色
                    if mod_name == "path":
                        color = plt.cm.Blues(intensity.detach().cpu().numpy())
                    elif mod_name == "omic":
                        color = plt.cm.Oranges(intensity.detach().cpu().numpy())
                    else:
                        color = "gray"  # 默认颜色，防止异常情况

                    rect = patches.Rectangle(
                        (0, i * (block_size + gap)),
                        block_size, block_size,
                        linewidth=0,
                        edgecolor=None,
                        facecolor=color,
                    )
                    ax.add_patch(rect)

                ax.set_xlim(0, block_size)
                ax.set_ylim(0, N * (block_size + gap))
                ax.invert_yaxis()
                ax.axis('off')
                fig.patch.set_alpha(0.0)
                ax.set_facecolor((0, 0, 0, 0))

                if batch["patient_id"] is not None:
                    image_name = batch["patient_id"][idx]
                    save_path = f"plots/importance_{image_name}_{mod_name}.png"
                else:
                    save_path = f"plots/importance_{idx}_{mod_name}.png"
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, transparent=True)
                plt.close(fig)

        # 将h_path和h_omic每个样本的importance都detach并分别映射到0.5-1.5，然后作为系数乘进去，然后重复t_path, f_path, t_omic, f_omic = self.bottleattn(h_path, h_omic, batch["patient_id"])开始到得到logits的流程
        
        # 映射 importance 到 [0.5, 1.5]
        importance_path_scaled = transform_importance_to_half_one_point_five(importance_path.detach())  # [B, N]
        importance_omic_scaled = transform_importance_to_half_one_point_five(importance_omic.detach())  # [B, N]

        # 权重扩展形状 [B, N, 1]
        importance_path_scaled = importance_path_scaled.unsqueeze(-1)  # [B, N, 1]
        importance_omic_scaled = importance_omic_scaled.unsqueeze(-1)

        # 对 h_path 和 h_omic 的 prototype 特征加权
        h_path_weighted = h_path * importance_path_scaled
        h_omic_weighted = h_omic * importance_omic_scaled

        # 重新进入注意力 + 分类流程
        t_path2, f_path2, t_omic2, f_omic2 = self.bottleattn(h_path_weighted, h_omic_weighted, batch["patient_id"])

        # 再次融合
        if self.fusion == 'bilinear':
            h2 = []
            for j in range(t_path2.shape[0]):
                h2.append(self.mm(t_path2[j], t_omic2[j]))
            h2 = torch.cat(h2, dim=0)
        elif self.fusion == 'concat':
            h2 = []
            for j in range(t_path2.shape[0]):
                h2.append(self.mm(torch.cat([t_path2[j], t_omic2[j]], axis=1)))
            h2 = torch.cat(h2, dim=0)

        # 再分类
        logits = self.classifier(h2)

        # 如果是训练阶段，继续保持原返回逻辑
        if self.training:
            if self.train_gan:
                return logits, modular_loss, gen_loss, dis_p_loss, dis_o_loss, loss_kd, importance_path_
            else:
                return logits, modular_loss, 0, 0, 0, loss_kd, importance_path_
        else:
            return logits

def transform_importance(x):
    # x: [B, N]
    min_val = x.min(dim=1, keepdim=True)[0]
    max_val = x.max(dim=1, keepdim=True)[0]
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    return 0.5 + 0.5 * x_norm

def transform_importance_to_half_one_point_five(x):
    # 每个样本独立归一化并映射到 [0.5, 1.5]
    min_val = x.min(dim=1, keepdim=True)[0]
    max_val = x.max(dim=1, keepdim=True)[0]
    x_norm = (x - min_val) / (max_val - min_val + 1e-8)
    x_scaled = 0.5 + x_norm
    return x_scaled

@MODEL_REGISTRY.register()
def umeml_gan(**kwargs):
    return UMEML_GAN(**kwargs)