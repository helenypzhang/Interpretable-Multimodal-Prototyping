import math
import random
import os
import h5py
import os.path as osp
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict
import tarfile
import zipfile

from typing import Any, Callable, Optional, Tuple

import torch
from medmm.utils import listdir_nohidden, verify_str_arg

from medmm.data.datasets.build import DATASET_REGISTRY
from medmm.data.datasets.base_dataset import check_isfile
from medmm.data.samplers import build_sampler
from medmm.data.data_manager import DatasetWrapper_UMEML, DatasetWrapper
from medmm.data.datasets.survival.tcga_glioma_s_umeml import Datum as Datum_Survival
from medmm.data.datasets.classification.tcga_glioma_g_umeml import Datum as Datum_Grading


class DatasetBase:
    """A unified dataset class for
    1) grading
    2) classing
    3) survival
    """

    dataset_dir = ""  # the directory where the dataset is stored


    def __init__(self, train=None, val=None, test=None):
        self._train = train  # labeled training data
        self._val = val  # validation data (optional)
        self._test = test  # test data
        self._num_classes = self.get_num_classes(self._test)
        self._lab2cname, self._classnames = self.get_lab2cname(self._test)

    @property
    def train(self):
        return self._train

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        """Count number of classes.

        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(int(item.label))
        return max(label_set) + 1

    @staticmethod
    def get_lab2cname(data_source):
        """Get a label-to-classname mapping (dict).

        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        # import pdb;pdb.set_trace()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames



    def download_data(self, url, dst, from_gdrive=True):
        if not osp.exists(osp.dirname(dst)):
            os.makedirs(osp.dirname(dst))

        # if from_gdrive:
        #     gdown.download(url, dst, quiet=False)
        # else:
        #     raise NotImplementedError

        print("Extracting file ...")

        if dst.endswith(".zip"):
            zip_ref = zipfile.ZipFile(dst, "r")
            zip_ref.extractall(osp.dirname(dst))
            zip_ref.close()

        elif dst.endswith(".tar"):
            tar = tarfile.open(dst, "r:")
            tar.extractall(osp.dirname(dst))
            tar.close()

        elif dst.endswith(".tar.gz"):
            tar = tarfile.open(dst, "r:gz")
            tar.extractall(osp.dirname(dst))
            tar.close()

        else:
            raise NotImplementedError

        print("File extracted to {}".format(osp.dirname(dst)))

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=False
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a small number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed (default: False).
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f"Creating a {num_shots}-shot dataset")

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

def convert_to_molpath(impath):
    molpath = impath.replace("features_clip_vit_b16", "molecular")
    molpath = os.path.splitext(molpath)[0] + ".csv"
    return molpath

class TCGA_Glioma_S_UMEML_NEW_TEST(DatasetBase):
    """TCGA_Glioma Survival
    """

    dataset_dir = "test"


    def __init__(self):
        root = "DATASET"
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self._meta_folder = osp.join(self.dataset_dir, "labels/survival") 
        self._bag_folder = osp.join(self.dataset_dir, "features_clip_vit_b16")  
        self._mol_folder = osp.join(self.dataset_dir, "molecular")
 
        # import pdb;pdb.set_trace()
        
        self.classnames_dict = {0:"SGrade I", 1:"SGrade II", 2:"SGrade III", 3:"SGrade IIII"}
        
        
        test = self._read_data(
            self._meta_folder, self._bag_folder, self._mol_folder
        )

        super().__init__(train=None,  val=None, test=test)

    def _read_data(self, meta_folder, bag_folder, mol_folder):
        items_t = []

        test_label_path = "DATASET/test/labels/survival/survival_test.csv"
        test_meta_apth = "DATASET/test/multimodal_complete_CPTAC.csv"
        
        label_df = pd.read_csv(test_label_path)
        meta_df = pd.read_csv(test_meta_apth)

        mol_base = "DATASET/test/molecular"
        img_base = "DATASET/test/features_clip_vit_b16"

        label_df["patients"] = label_df["patients"].astype(str).str.strip()
        meta_df["WSI_ID"] = meta_df["WSI_ID"].astype(str).str.strip()

        merged_df = label_df.merge(meta_df, left_on="patients", right_on="WSI_ID", how="left")

        merged_df.dropna(subset=["WSI_ID"], inplace=True)

        merged_df["molpath"] = merged_df.apply(
            lambda row: os.path.join(mol_base, f"{row['File ID']}", f"{row['File Name']}"), axis=1
        )

        merged_df["impath"] = merged_df["patients"].apply(lambda pid: os.path.join(img_base, pid) + ".h5")
        merged_df["molpath"] = merged_df["impath"].apply(convert_to_molpath)
        
        for index, row in merged_df.iterrows():
            label = int(row["labels"])
            survival = {"labels": row["labels"], "survival_months": row["survival_months"], "censorship": row["censorship"]}
            bag_path =  row["impath"]
            patient_id = bag_path.split("/")[-1].split('.')[0]
            mol_path = row["molpath"]
            item = Datum_Survival(patientid=patient_id,
                         impath=bag_path, 
                         molpath=mol_path, 
                         label=label, 
                         classname=self.classnames_dict[label],
                         survival=survival,
                         )
            items_t.append(item)

        return items_t

class TCGA_Glioma_G_UMEML_NEW_TEST(DatasetBase):
    """TCGA_Glioma Survival
    """

    dataset_dir = "test"


    def __init__(self):
        root = "DATASET"
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self._meta_folder = osp.join(self.dataset_dir, "labels/survival") 
        self._bag_folder = osp.join(self.dataset_dir, "features_clip_vit_b16")  
        self._mol_folder = osp.join(self.dataset_dir, "molecular")
 
        # import pdb;pdb.set_trace()
        
        self.classnames_dict = {0:"SGrade I", 1:"SGrade II", 2:"SGrade III", 3:"SGrade IIII"}
        
        
        test = self._read_data(
            self._meta_folder, self._bag_folder, self._mol_folder
        )

        super().__init__(train=None,  val=None, test=test)

    def _read_data(self, meta_folder, bag_folder, mol_folder):
        items_t = []

        test_label_path = "DATASET/test/labels/grading/grading_test.csv"
        test_meta_apth = "DATASET/test/multimodal_complete_CPTAC.csv"
        
        label_df = pd.read_csv(test_label_path)
        meta_df = pd.read_csv(test_meta_apth)

        mol_base = "DATASET/test/molecular"
        img_base = "DATASET/test/features_clip_vit_b16"

        label_df["patients"] = label_df["patients"].astype(str).str.strip()
        meta_df["WSI_ID"] = meta_df["WSI_ID"].astype(str).str.strip()

        merged_df = label_df.merge(meta_df, left_on="patients", right_on="WSI_ID", how="left")

        merged_df.dropna(subset=["WSI_ID"], inplace=True)

        merged_df["molpath"] = merged_df.apply(
            lambda row: os.path.join(mol_base, f"{row['File ID']}", f"{row['File Name']}"), axis=1
        )

        merged_df["impath"] = merged_df["patients"].apply(lambda pid: os.path.join(img_base, pid) + ".h5")
        merged_df["molpath"] = merged_df["impath"].apply(convert_to_molpath)
        
        for index, row in merged_df.iterrows():
            label = int(row["labels"])
            bag_path =  row["impath"]
            patient_id = bag_path.split("/")[-1].split('.')[0]
            mol_path = row["molpath"]
            item = Datum_Grading(patientid=patient_id,
                         impath=bag_path, 
                         molpath=mol_path, 
                         label=label, 
                         classname=self.classnames_dict[label]
                         )
            items_t.append(item)

        return items_t
    
def build_data_loader(
    cfg,
    sampler_type="SequentialSampler",
    data_source=None,
    batch_size=64,
    is_train=True,
    dataset_wrapper=None
):
    # Build sampler
    sampler = build_sampler(
        sampler_type,
        cfg=cfg,
        data_source=data_source,
    )

    if dataset_wrapper is None:
        if 'umeml' in cfg.MODEL.NAME:
            dataset_wrapper = DatasetWrapper_UMEML
        else:
            dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(cfg, data_source),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=is_train and len(data_source) >= batch_size,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
        collate_fn=custom_collate
    )
    assert len(data_loader) > 0

    return data_loader


def build_test_new(cfg):
    if cfg.TASK.NAME.lower() == "survival":
        test_dataset = TCGA_Glioma_S_UMEML_NEW_TEST()
    elif cfg.TASK.NAME.lower() == "grading":
        test_dataset = TCGA_Glioma_G_UMEML_NEW_TEST()

    test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=test_dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            # tfm=tfm_train,
            is_train=False,
            dataset_wrapper=None,
        )
    
    return test_loader


def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if all(v is None for v in values):
            collated[key] = None
        else:
            collated[key] = [v for v in values]
    return collated