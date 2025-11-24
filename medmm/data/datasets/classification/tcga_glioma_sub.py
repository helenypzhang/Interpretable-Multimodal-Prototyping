import math
import random
import os
import h5py
import os.path as osp
import pandas as pd
import numpy as np

from typing import Any, Callable, Optional, Tuple

import torch
from medmm.utils import listdir_nohidden, verify_str_arg

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase


@DATASET_REGISTRY.register()
class TCGA_Glioma_Sub(DatasetBase):
    """TCGA_Glioma subtyping.


    """

    dataset_dir = "tcga_glioma"

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self._meta_folder = osp.join(self.dataset_dir, "labels/subtyping") 
        self._bag_folder = osp.join(self.dataset_dir, cfg.DATASET.FEATURE_NAME)  
        self._mol_folder = osp.join(self.dataset_dir, "molecular") 
        # 5 fold 
        self._fold = verify_str_arg(cfg.DATASET.FOLD, "Fold", ("1", "2", "3", "4", "5"))
        classnames_dict = {0:"GBM", 1:"Astro", 2:"Oligo"}
        self.classnames_dict = classnames_dict
        
        
        train, val = self._read_data(
            self._meta_folder, self._bag_folder, self._mol_folder, self._fold
        )
        test = val


        if len(val) == 0:
            val = None

        super().__init__(train=train,  val=val, test=test)

    def _read_data(self, meta_folder, bag_folder, mol_folder, fold=0):
        items_t, items_v = [], []
        
        train_meta_path = osp.join(meta_folder, f'subtyping_train_{fold}.csv')
        val_meta_path = osp.join(meta_folder, f'subtyping_test_{fold}.csv')
        
        train_meta = pd.read_csv(train_meta_path)
        
        for index, row in train_meta.iterrows():
            bag_name = row["features"]
            mol_name = bag_name.replace(".h5", ".csv")
            label = int(row["labels"])
            bag_path = osp.join(bag_folder, bag_name)
            mol_path = osp.join(mol_folder, mol_name)
            item = Datum(impath=bag_path, 
                         molpath=mol_path, 
                         label=label, 
                         classname=self.classnames_dict[label])
            items_t.append(item)
            
            
        val_meta = pd.read_csv(val_meta_path)
        
        for index, row in val_meta.iterrows():
            bag_name = row["features"]
            mol_name = bag_name.replace(".h5", ".csv")
            label = row["labels"]
            bag_path = osp.join(bag_folder, bag_name)
            mol_path = osp.join(mol_folder, mol_name)
            item = Datum(impath=bag_path, 
                         molpath=mol_path, 
                         label=label, 
                         classname=self.classnames_dict[label])
            items_v.append(item)


        return items_t, items_v


    
    
