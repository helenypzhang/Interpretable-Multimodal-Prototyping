import time
import random
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F

from medmm.engine import TRAINER_REGISTRY, Trainer

from medmm.utils import load_pretrained_weights, load_checkpoint, print_trainable_parameters
from medmm.optim import build_optimizer, build_lr_scheduler
from medmm.modeling import build_model
from medmm.loss import build_loss
from medmm.modeling.models.umeml_gan import UMEML_GAN

def split_chunk(data, target_length=512):
    # data = [bs, N, D]
    BS, N, D = data.shape
    num_splits = N //  target_length
    remainder = N % target_length

    if remainder > 0:
        padding_length = target_length - remainder
        padding = torch.zeros(BS, padding_length, D).to(data.device)
        data = torch.cat((data, padding), dim=1)  
        N += padding_length
        num_splits += 1  

    indices = torch.randperm(N, device=data.device)
    split_tensors = []
    
    for i in range(num_splits):
        start_index = i * target_length
        end_index = min((i + 1) * target_length, N)

        indices_in_range = indices[start_index:end_index]
        indices_in_range = torch.sort(indices_in_range).values

        sub_data = data[:, indices_in_range, :]
        split_tensors.append(sub_data)

    return split_tensors




@TRAINER_REGISTRY.register()
class MBTRAIN(Trainer):
    """

    """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        # omic_sizes = self.dm.dataset.omic_sizes
        omic_sizes = 1000
        self.use_bsm = cfg.DATASET.USE_BSM
        self.bs_micro = cfg.DATASET.BS_MICRO

        print("Building Model")
        print("Building model")
        num_classes = len(classnames)
        self.model = build_model(
            cfg.MODEL.NAME,
            verbose=cfg.VERBOSE,
            cfg=cfg,
            num_classes=num_classes,
            omic_sizes=omic_sizes,
        )

        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            self.model.float()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        print_trainable_parameters(self.model)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        try:
            self.loss_fn = build_loss(
                cfg.TASK.LOSS, alpha=cfg.LOSS.ALPHA, reduction=cfg.LOSS.REDUCTION)
        except:
            self.loss_fn = build_loss(
                cfg.TASK.LOSS)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(
                f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        # patient_id, x_path, x_mask, x_omic, label, event_time, censorship = self.parse_batch(
        #     batch)
        patient_id, x_path, x_omic, label, _, censorship = self.parse_batch(batch)
        prec = self.cfg.TRAINER.PREC
        alpha = self.cfg.MODEL.UMEML.ALPHA
        if self.use_bsm:
            loss = 0.0
            cnt = 0
            # import pdb;pdb.set_trace()
            x_path_chunks = split_chunk(x_path, self.bs_micro)
            if prec == "amp":
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic, "patient_id": batch["patient_id"]}  
                    with autocast():
                        if self.cfg.TASK.NAME == "Survival":
                            logits, modular_loss_micro = self.model_inference(input)
                            loss_micro = self.loss_fn(
                                logits=logits, Y=label, c=censorship)
                        else:
                            logits, modular_loss_micro = self.model_inference(input)
                            loss_micro = self.loss_fn(logits, label)
                    loss += loss_micro + alpha * modular_loss_micro 
                    cnt+=1
                loss = loss / cnt    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
            else:
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic, "patient_id": batch["patient_id"]}  
                    if self.cfg.TASK.NAME == "Survival":
                        logits, modular_loss_micro = self.model_inference(input)
                        loss_micro = self.loss_fn(logits=logits,
                                            Y=label, c=censorship)
                    else:
                        logits, modular_loss_micro = self.model_inference(input)
                        loss_micro = self.loss_fn(logits, label)
                    loss += loss_micro + alpha * modular_loss_micro
                    cnt+=1
                loss = loss / cnt    
                self.model_backward_and_update(loss)  
        else:
            input = {"img": x_path, "omic": x_omic, "patient_id": batch["patient_id"]}  
            if prec == "amp":
                with autocast():
                    if self.cfg.TASK.NAME == "Survival":
                        logits = self.model_inference(input)
                        loss = self.loss_fn(
                            logits=logits, Y=label, c=censorship)
                    else:
                        logits = self.model_inference(input)
                        loss = self.loss_fn(logits, label)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()
            else:
                if self.cfg.TASK.NAME == "Survival":
                    logits = self.model_inference(input)
                    if len(logits) == 2:
                        loss = self.loss_fn(logits=logits,
                                            Y=label, c=censorship)
                                            
                        loss = loss + 1 * logits[1]
                    elif len(logits) == 5 and logits[-1] != 'cca':
                        loss = self.loss_fn(logits=logits,
                                            Y=label, c=censorship)
                                            
                        loss = loss + 1 * logits[1]
                        loss_gen = logits[2]
                        loss_dis_p = logits[3]
                        loss_dis_o = logits[4]
                    elif len(logits) == 7 and logits[-1] != 'cca':
                        loss = self.loss_fn(logits=logits,
                                            Y=label, c=censorship)
                        loss += logits[-2]
                        loss = loss + 1 * logits[1]
                        loss_gen = logits[2]
                        loss_dis_p = logits[3]
                        loss_dis_o = logits[4]
                    elif len(logits) == 5 and logits[-1] == 'cca':
                        p_proto_before, h_omic_bag_before = logits[2], logits[3]
                        p_proto_before = p_proto_before.view(p_proto_before.shape[0], -1)
                        p_similarity_matrix = torch.mm(p_proto_before, p_proto_before.T)
                        h_omic_bag_before = h_omic_bag_before.view(h_omic_bag_before.shape[0], -1)
                        h_similarity_matrix = torch.mm(h_omic_bag_before, h_omic_bag_before.T)
                        p_similarity_matrix = cosine_similarity_matrix(p_similarity_matrix)
                        h_similarity_matrix = cosine_similarity_matrix(h_similarity_matrix)
                        loss = F.mse_loss(p_similarity_matrix, h_similarity_matrix)
                        return logits[0], logits[1], loss           
                    elif len(logits) == 7 and logits[-1] == 'cca':
                        p_proto_before, h_omic_bag_before = logits[2], logits[3]
                        p_proto_before = p_proto_before.view(p_proto_before.shape[0], -1)
                        p_similarity_matrix = torch.mm(p_proto_before, p_proto_before.T)
                        h_omic_bag_before = h_omic_bag_before.view(h_omic_bag_before.shape[0], -1)
                        h_similarity_matrix = torch.mm(h_omic_bag_before, h_omic_bag_before.T)
                        p_similarity_matrix = cosine_similarity_matrix(p_similarity_matrix)
                        h_similarity_matrix = cosine_similarity_matrix(h_similarity_matrix)
                        loss = F.mse_loss(p_similarity_matrix, h_similarity_matrix)
                        return logits[0], logits[1], loss               
                else:
                    logits = self.model_inference(input)
                    if isinstance(logits, tuple):
                        if logits[-1] != 'cca':
                            loss = self.loss_fn(logits[0], label)
                            if len(logits) == 7:
                                loss += 1 * logits[-2]
                        else:
                            if isinstance(logits, tuple) and len(logits) == 5 and logits[-1] != 'cca':
                                loss_gen = logits[2]
                                loss_dis_p = logits[3]
                                loss_dis_o = logits[4]
                            elif isinstance(logits, tuple) and len(logits) == 7 and logits[-1] != 'cca':
                                loss_gen = logits[2]
                                loss_dis_p = logits[3]
                                loss_dis_o = logits[4]
                            elif isinstance(logits, tuple) and len(logits) == 5 and logits[-1] == 'cca':
                                p_proto_before, h_omic_bag_before = logits[2], logits[3]
                                p_proto_before = p_proto_before.view(p_proto_before.shape[0], -1)
                                p_similarity_matrix = torch.mm(p_proto_before, p_proto_before.T)
                                h_omic_bag_before = h_omic_bag_before.view(h_omic_bag_before.shape[0], -1)
                                h_similarity_matrix = torch.mm(h_omic_bag_before, h_omic_bag_before.T)
                                p_similarity_matrix = cosine_similarity_matrix(p_similarity_matrix)
                                h_similarity_matrix = cosine_similarity_matrix(h_similarity_matrix)
                                loss = F.mse_loss(p_similarity_matrix, h_similarity_matrix) 
                                return logits[0], logits[1], loss
                            elif isinstance(logits, tuple) and len(logits) == 7 and logits[-1] == 'cca':
                                p_proto_before, h_omic_bag_before = logits[2], logits[3]
                                p_proto_before = p_proto_before.view(p_proto_before.shape[0], -1)
                                p_similarity_matrix = torch.mm(p_proto_before, p_proto_before.T)
                                h_omic_bag_before = h_omic_bag_before.view(h_omic_bag_before.shape[0], -1)
                                h_similarity_matrix = torch.mm(h_omic_bag_before, h_omic_bag_before.T)
                                p_similarity_matrix = cosine_similarity_matrix(p_similarity_matrix)
                                h_similarity_matrix = cosine_similarity_matrix(h_similarity_matrix)
                                loss = F.mse_loss(p_similarity_matrix, h_similarity_matrix) 
                                return logits[0], logits[1], loss
                            # return logits[0], logits[1]
                    else:
                        loss = self.loss_fn(logits, label)
                    loss = loss + 1 * logits[1]
                             
                self.model_backward_and_update(loss)
                    
                

        loss_summary = {
            "loss": loss.item(),
        }

        if 'loss_gen' in vars():
            loss_summary['loss_gen'] = loss_gen
            loss_summary['loss_dis_p'] = loss_dis_p
            loss_summary['loss_dis_o'] = loss_dis_o

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def get_omic_delete_mask(self, batch_size, num_features, insample_without_omic_ratio, seed=None):
        rng = np.random.RandomState(seed)
        mask = np.zeros((batch_size, num_features), dtype=np.int32)
        num_selected = int(num_features * insample_without_omic_ratio)

        for i in range(batch_size):
            selected_indices = rng.choice(num_features, num_selected, replace=False)
            mask[i, selected_indices] = 1  # 设置选中的索引为 1

        return torch.tensor(mask, dtype=torch.int32)


    @torch.no_grad()
    def test(self, split=None, umeml_gan_test_without_omic_ratio=0, umeml_gan_test_insample_without_omic_ratio=0, omic_means=None):
        self.model.omic_means = omic_means
        self.set_model_mode("eval")
        self.evaluator.reset()

        if not hasattr(self.model, 'cca'):
            self.model.cca = False

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        # 随机选取特定比例的删除omic
        without_omic_ratio = umeml_gan_test_without_omic_ratio
        insample_without_omic_ratio = umeml_gan_test_insample_without_omic_ratio
        num_samples = len(data_loader.dataset)
        num_selected = int(num_samples * without_omic_ratio)

        def get_random_indices(seed=None):
            rng = np.random.RandomState(seed)
            return rng.choice(num_samples, num_selected, replace=False)

        indexes_without_omic = get_random_indices(seed=42)

        batch_start = 0

        print(f"Evaluate on the *{split}* set")
        for batch_index, batch in enumerate(tqdm(data_loader)):
            # patient_id, x_path, x_mask, x_omic, label, event_time, censorship = self.parse_batch(
            #     batch)

            batch_length = batch['img'].shape[0]
            
            batch_without_omic = []
            for index_in_batch in range(batch_length):
                index_this_sample = batch_start + index_in_batch
                batch_without_omic.append(1 if np.isin(index_this_sample, indexes_without_omic) else 0)

            batch_start += batch_length

            without_omic = torch.tensor(np.array(batch_without_omic))

            patient_id, x_path, x_omic, label, event_time, censorship = self.parse_batch(batch)
            
            if self.use_bsm:
                all_logits = 0.
                all_S = 0.
                cnt = 0
                x_path_chunks = split_chunk(x_path, self.bs_micro)
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic}
                    if isinstance(self.model, UMEML_GAN):
                        del input['omic']
                    logits_micro = self.model_inference(input)
                    all_logits = all_logits + logits_micro
                    cnt+=1
                logits = all_logits / cnt
                
                if self.cfg.TASK.NAME == "Survival":

                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:

                    self.evaluator.process(patient_id, logits, label)
                        
            else:
                # 根据mmdl代码，测试阶段的mask是全1
                x_mask = torch.ones([x_path.shape[0], 1])
                sample_seed = 0
                if split == "val":
                    seed = 10000 + batch_index
                elif split == "test":
                    seed = 20000 + batch_index
                insample_without_mask = self.get_omic_delete_mask(x_omic.shape[0], x_omic.shape[1], insample_without_omic_ratio, seed=seed).to(x_path.device)
                input = {"img": x_path, "mask": x_mask, "omic": x_omic, "without_omic": without_omic.to(x_path.device), "insample_without_omic": insample_without_mask, "patient_id": patient_id}
                # if isinstance(self.model, UMEML_GAN):
                #     del input['omic']
                logits = self.model_inference(input)
                if self.cfg.TASK.NAME == "Survival":
                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:
                    self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    @torch.no_grad()
    def test_new(self, cfg, split=None, umeml_gan_test_without_omic_ratio=0, umeml_gan_test_insample_without_omic_ratio=0, omic_means=None):
        self.model.omic_means = omic_means
        self.set_model_mode("eval")
        self.evaluator.reset()

        if not hasattr(self.model, 'cca'):
            self.model.cca = False
        
        from dataset_new import build_test_new
        data_loader = build_test_new(cfg)

        # 随机选取特定比例的删除omic
        without_omic_ratio = umeml_gan_test_without_omic_ratio
        insample_without_omic_ratio = umeml_gan_test_insample_without_omic_ratio
        num_samples = len(data_loader.dataset)
        num_selected = int(num_samples * without_omic_ratio)

        def get_random_indices(seed=None):
            rng = np.random.RandomState(seed)
            return rng.choice(num_samples, num_selected, replace=False)

        indexes_without_omic = get_random_indices(seed=42)

        batch_start = 0

        print(f"Evaluate on the *{split}* set")
        for batch_index, batch in enumerate(tqdm(data_loader)):
            # patient_id, x_path, x_mask, x_omic, label, event_time, censorship = self.parse_batch(
            #     batch)

            batch_length = np.asarray(batch['img']).shape[0]
            
            batch_without_omic = []
            for index_in_batch in range(batch_length):
                index_this_sample = batch_start + index_in_batch
                batch_without_omic.append(1 if np.isin(index_this_sample, indexes_without_omic) else 0)

            batch_start += batch_length

            without_omic = torch.tensor(np.array(batch_without_omic))

            merged = {}
            for key, values in batch.items():
                if values is None:
                    merged[key] = None
                elif all(v is None for v in values):
                    merged[key] = None
                elif isinstance(values[0], torch.Tensor):
                    valid_tensors = [v for v in values if v is not None]
                    merged[key] = torch.stack(valid_tensors, dim=0)
                else:
                    # 字符串、数字等保持为 list
                    merged[key] = values

            batch = merged

            patient_id, x_path, x_omic, label, event_time, censorship = self.parse_batch(batch)

            if x_omic is None:
                without_omic[0] = 1
            
            if self.use_bsm:
                all_logits = 0.
                all_S = 0.
                cnt = 0
                x_path_chunks = split_chunk(x_path, self.bs_micro)
                for x_path_mb in x_path_chunks:
                    input = {"path": x_path_mb, "omic": x_omic}
                    if isinstance(self.model, UMEML_GAN):
                        del input['omic']
                    logits_micro = self.model_inference(input)
                    all_logits = all_logits + logits_micro
                    cnt+=1
                logits = all_logits / cnt
                
                if self.cfg.TASK.NAME == "Survival":

                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:

                    self.evaluator.process(patient_id, logits, label)
                        
            else:
                # 根据mmdl代码，测试阶段的mask是全1
                x_mask = torch.ones([x_path.shape[0], 1])
                sample_seed = 0
                if split == "val":
                    seed = 10000 + batch_index
                elif split == "test":
                    seed = 20000 + batch_index
                if x_omic is not None:
                    insample_without_mask = self.get_omic_delete_mask(x_omic.shape[0], x_omic.shape[1], insample_without_omic_ratio, seed=seed).to(x_path.device)
                else:
                    insample_without_mask = torch.ones((1, 1000), device=x_path.device)
                input = {"img": x_path, "mask": x_mask, "omic": x_omic, "without_omic": without_omic.to(x_path.device), "insample_without_omic": insample_without_mask, "patient_id": patient_id}
                # if isinstance(self.model, UMEML_GAN):
                #     del input['omic']
                logits = self.model_inference(input)
                if self.cfg.TASK.NAME == "Survival":
                    self.evaluator.process(patient_id, logits, censorship, event_time)
                else:
                    self.evaluator.process(logits, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
def cosine_similarity_matrix(similarity_matrix):
    norm = similarity_matrix.norm(p=2, dim=1, keepdim=True)  # L2 norm
    return similarity_matrix / norm
