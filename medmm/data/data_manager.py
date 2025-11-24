import torch
import h5py
import numpy as np
import pandas as pd

import torchvision.transforms as T
from tabulate import tabulate
from torch.utils.data import Dataset as TorchDataset

from medmm.utils import read_image

from .datasets import build_dataset
from .samplers import build_sampler
from .transforms import INTERPOLATION_MODES, build_transform




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
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA)
    )
    assert len(data_loader) > 0

    return data_loader


class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # # Build transform
        # if custom_tfm_train is None:
        #     tfm_train = build_transform(cfg, is_train=True)
        # else:
        #     print("* Using custom transform for training")
        #     tfm_train = custom_tfm_train

        # if custom_tfm_test is None:
        #     tfm_test = build_transform(cfg, is_train=False)
        # else:
        #     print("* Using custom transform for testing")
        #     tfm_test = custom_tfm_test

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            # tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )


        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                # tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            # tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._classnames = dataset.classnames
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes


    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):        
        return ", ".join(map(str, self._classnames))

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME


        table = []
        table.append(["Dataset", dataset_name])

        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# classnames", f"{self.classnames}"])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper(TorchDataset):

    def __init__(self, cfg, data_source):
        self.cfg = cfg
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        item = self.data_source[idx]

        with h5py.File(item.impath, 'r') as f:
            # import pdb;pdb.set_trace()
            bag = f['clip_vit_b32_feature'][:]  #  Res_feature  , clip_vit_b32_feature
            
        molecular = pd.read_csv(item.molpath)["fpkm_uq_unstranded"]    
            
        # import pdb;pdb.set_trace()
        label = torch.tensor(np.array(item.label))
        # print(item.impath)
        bag = torch.tensor(np.array(bag)).float()
        molecular = torch.tensor(np.array(molecular)).float()

        # print(bag.shape)
        # print(molecular.shape)
        
        if self.cfg.TASK.NAME == "Survival":
            #  {"labels": row["labels"], "survival_months": row["survival_months"], "censorship": row["censorship"]}
            label = torch.tensor(np.array(item.survival["labels"]))
            survival_month =  torch.tensor(np.array(item.survival["survival_months"]))
            censorship =  torch.tensor(np.array(item.survival["censorship"]))
            patient_id = item.patientid
            output = {
                "label": label,
                "survival_month": survival_month,
                "censorship": censorship,
                "img": bag,
                "mol": molecular,
                "index": idx,
                "patient_id": patient_id
            }


        else:
            patient_id = item.patientid
            output = {
                "label": label,
                "img": bag,
                "mol": molecular,
                "index": idx,
                "patient_id": patient_id
            }

        return output
    
    def get_envent_and_cenorship(self):
        event_times, censorships = {}, {}
        for data in self.data_source:
            # event_time, censorship = data.label['event_time'], data.label['censorship']
            event_time, censorship = data.survival['survival_months'], data.survival['censorship']
            patient_id = data.patientid

            if patient_id in event_times:
                event_times[patient_id].append(event_time)
            else:
                event_times[patient_id] = [event_time]

            if patient_id in censorships:
                censorships[patient_id].append(censorship)
            else:
                censorships[patient_id] = [censorship]

        

        event_times = [np.mean(values) for _, values in event_times.items()]
        censorships = [int(np.mean(values))
                       for _, values in censorships.items()]

        return np.array(event_times), np.array(censorships)

class DataManager:

    def __init__(
        self,
        cfg,
        custom_tfm_train=None,
        custom_tfm_test=None,
        dataset_wrapper=None
    ):
        # Load dataset
        dataset = build_dataset(cfg)

        # # Build transform
        # if custom_tfm_train is None:
        #     tfm_train = build_transform(cfg, is_train=True)
        # else:
        #     print("* Using custom transform for training")
        #     tfm_train = custom_tfm_train

        # if custom_tfm_test is None:
        #     tfm_test = build_transform(cfg, is_train=False)
        # else:
        #     print("* Using custom transform for testing")
        #     tfm_test = custom_tfm_test

        # Build train_loader
        train_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN.SAMPLER,
            data_source=dataset.train,
            batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
            # tfm=tfm_train,
            is_train=True,
            dataset_wrapper=dataset_wrapper
        )


        # Build val_loader
        val_loader = None
        if dataset.val:
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                # tfm=tfm_test,
                is_train=False,
                dataset_wrapper=dataset_wrapper
            )

        # Build test_loader
        test_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TEST.SAMPLER,
            data_source=dataset.test,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            # tfm=tfm_test,
            is_train=False,
            dataset_wrapper=dataset_wrapper
        )

        # Attributes
        self._num_classes = dataset.num_classes
        self._classnames = dataset.classnames
        self._lab2cname = dataset.lab2cname

        # Dataset and data-loaders
        self.dataset = dataset
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.test_loader = test_loader

        if cfg.VERBOSE:
            self.show_dataset_summary(cfg)

    @property
    def num_classes(self):
        return self._num_classes


    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):        
        return ", ".join(map(str, self._classnames))

    def show_dataset_summary(self, cfg):
        dataset_name = cfg.DATASET.NAME


        table = []
        table.append(["Dataset", dataset_name])

        table.append(["# classes", f"{self.num_classes:,}"])
        table.append(["# classnames", f"{self.classnames}"])
        table.append(["# train", f"{len(self.dataset.train):,}"])
        if self.dataset.val:
            table.append(["# val", f"{len(self.dataset.val):,}"])
        table.append(["# test", f"{len(self.dataset.test):,}"])

        print(tabulate(table))


class DatasetWrapper_UMEML(TorchDataset):

    def __init__(self, cfg, data_source):
        self.cfg = cfg
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)
    
    def pad_bag(self, bag, target_shape):
        # 计算需要填充的数量，只对第一维（w）进行填充
        padding_w = target_shape - bag.shape[0]
        
        # 确保 padding_w >= 0，因为不能填充负值
        if padding_w > 0:
            padding = [(0, padding_w), (0, 0)]  # (对w的填充, 对h的填充)
            padded_bag = np.pad(bag, padding, mode='constant', constant_values=-10000)
        else:
            padded_bag = bag  # 如果 w 已经大于或等于 target_shape，则不填充
        
        return torch.tensor(padded_bag).float()

    def __getitem__(self, idx):
        item = self.data_source[idx]

        with h5py.File(item.impath, 'r') as f:
            # import pdb;pdb.set_trace()
            bag = f['clip_vit_b32_feature'][:]  #  Res_feature  , clip_vit_b32_feature
            
        molecular = pd.read_csv(item.molpath)["fpkm_uq_unstranded"]    
            
        # import pdb;pdb.set_trace()
        label = torch.tensor(np.array(item.label))
        # print(item.impath)
        bag = torch.tensor(np.array(bag)).float()
        molecular = torch.tensor(np.array(molecular)).float()
        
        # print(bag.shape)
        # print(molecular.shape)

        bag = self.pad_bag(bag, 10000)

        if self.cfg.TASK.NAME == "Survival":
            #  {"labels": row["labels"], "survival_months": row["survival_months"], "censorship": row["censorship"]}
            label = torch.tensor(np.array(item.survival["labels"]))
            survival_month =  torch.tensor(np.array(item.survival["survival_months"]))
            censorship =  torch.tensor(np.array(item.survival["censorship"]))
            patient_id = item.patientid
            output = {
                "label": label,
                "survival_month": survival_month,
                "censorship": censorship,
                "img": bag,
                "mol": molecular,
                "patient_id": patient_id,
                "index": idx
            }


        else:
            patient_id = item.patientid
            output = {
                "label": label,
                "img": bag,
                "mol": molecular,
                "patient_id": patient_id,
                "index": idx
            }

        return output

    def get_envent_and_cenorship(self):
        event_times, censorships = {}, {}
        for data in self.data_source:
            # event_time, censorship = data.label['event_time'], data.label['censorship']
            event_time, censorship = data.survival['survival_months'], data.survival['censorship']
            patient_id = data.patientid

            if patient_id in event_times:
                event_times[patient_id].append(event_time)
            else:
                event_times[patient_id] = [event_time]

            if patient_id in censorships:
                censorships[patient_id].append(censorship)
            else:
                censorships[patient_id] = [censorship]

        

        event_times = [np.mean(values) for _, values in event_times.items()]
        censorships = [int(np.mean(values))
                       for _, values in censorships.items()]

        return np.array(event_times), np.array(censorships)