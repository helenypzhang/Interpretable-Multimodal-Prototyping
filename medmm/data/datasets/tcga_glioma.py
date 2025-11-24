import os
import pathlib
import h5py
from typing import Any, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch

from .base import MedMMDataset
from .utils import verify_str_arg


class TCGA_Glioma(MedMMDataset):
    """TCGA_Glioma <> Dataset

    Args:
        MedMMDataset (_type_): _description_
    """
    def __init__(
        self,
        root: str,
        train: bool = True,
        fix_flag: str = "norm",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self._fix_flag = verify_str_arg(fix_flag, "fix_flag", ("norm", "2500_fixdim0_norm"))
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.train = train  # training set or test set
        
        self._data_folder = pathlib.Path(self.root) / type(self).__name__.lower()
        self._meta_folder = self._data_folder / "labels"
        self._instance_feature_folder = self._data_folder / "resnet50_features" / fix_flag      
        self._molecular_folder = self._data_folder / "molecular" / fix_flag       

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can acess "" to download it")
        
        self._instance_feature_files = []
        
        file = pd.read_csv(self._meta_folder / f"labels_{self._partition}.csv")
        if self.train:
            # 按照 0.8 的比例随机采样得到训练集, seed 固定,保证每次抽取都一样
            train_set = file.sample(frac=0.8, random_state=42)
            # 重置训练集的索引
            train_set = train_set.reset_index(drop=True)
            self._set = train_set
        else:
            # 按照 0.8 的比例随机采样得到训练集, seed 固定,保证每次抽取都一样
            train_set = file.sample(frac=0.8, random_state=42)
            # 使用 pandas 的 `drop` 方法从原始 DataFrame 中移除训练集的部分，从而得到测试集
            test_set = file.drop(train_set.index).reset_index(drop=True)
            self._set = test_set

        
    def get_bag_feats(self, csv_file_df):
     
        feature_filename = csv_file_df["features"]
        molecular_filename = feature_filename.replace(".h5", ".csv")
        label = csv_file_df["labels"]
        
        
        bag_path = os.path.join(self._instance_feature_folder, feature_filename)
        molecular_path = os.path.join(self._molecular_folder, molecular_filename)
        
        with h5py.File(bag_path, 'r') as f:
            bag = f['Res_feature'][0]
            
        molecular = pd.read_csv(molecular_path)["fpkm_uq_unstranded"]    
            
        # import pdb;pdb.set_trace()
        label = torch.tensor(np.array(label))
        bag = torch.tensor(np.array(bag)).float()
        return label, bag, molecular
    
    def __len__(self) -> int:
        return len(self._set)    
        
    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any]:
        label, bag, molecular = self.get_bag_feats(self._set.iloc[idx])
        return  label, bag, molecular
    
    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder) and os.path.isdir(self._data_folder)