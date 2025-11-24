import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from medmm.engine import TRAINER_REGISTRY, Trainer
from medmm.metrics import compute_accuracy
from medmm.utils import load_pretrained_weights, load_checkpoint
from medmm.optim import build_optimizer, build_lr_scheduler
from medmm.modeling import build_head,build_mil
from medmm.loss import build_loss




class TransMIL_NET(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """
    def __init__(self, cfg, classnames, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.mil = build_mil(
            cfg.MODEL.MIL.NAME,
            verbose=cfg.VERBOSE,
            **kwargs,
        )
        fdim = self.mil.out_features
        num_classes = len(classnames)
        # import pdb;pdb.set_trace()

        # 1. Grading  2. Classification 3. Subtyping
        if cfg.TASK.NAME == "Grading" or cfg.TASK.NAME == "Classification" \
            or cfg.TASK.NAME == "Subtyping"  : 
            self.classifier = None
            if num_classes > 0:
                self.classifier = nn.Linear(fdim, num_classes)
        elif cfg.TASK.NAME == "Survival" :
            
            num_classes = 4 
            self.classifier = nn.Linear(fdim, num_classes)     

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.mil(x)

        if self.classifier is None:
            return f

        logits = self.classifier(f)
        # logits = F.softmax(logits, dim = 1)

        if return_feature:
            return logits, f
        
        if self.cfg.TASK.NAME == "Survival":
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, logits

        return logits



@TRAINER_REGISTRY.register()
class TransMIL(Trainer):
    """ABMIL.

    
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        
        print("Building Model")
        self.model = TransMIL_NET(cfg, classnames)
        
        if cfg.TRAINER.PREC == "fp32" or cfg.TRAINER.PREC == "amp":
            self.model.float()

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        self.loss_fn = build_loss(cfg.TASK.LOSS)

        self.scaler = GradScaler() if cfg.TRAINER.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
    
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # import pdb;pdb.set_trace()
        if self.cfg.TASK.NAME == "Survival":
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                image, omic, label, survival_month, censorship = self.parse_batch(batch)
                hazards, S, logits = self.model_inference(image)
                
                self.evaluator.process(S, censorship, survival_month)
            results = self.evaluator.evaluate()
            
        else:
            for batch_idx, batch in enumerate(tqdm(data_loader)):
                image, omic, label, survival_month, censorship = self.parse_batch(batch)
                
                output = self.model_inference(image)
                self.evaluator.process(output, label)
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def forward_backward(self, batch):
        image, omic, label, survival_month, censorship = self.parse_batch(batch)
        # import pdb;pdb.set_trace()
        prec = self.cfg.TRAINER.PREC
        if prec == "amp":
            with autocast():
                if self.cfg.TASK.NAME == "Survival":
                    hazards, S, logits = self.model(image)
                    loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
                else: 
                    output = self.model(image)
                    loss = self.loss_fn(output, label)

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if self.cfg.TASK.NAME == "Survival":
                hazards, S, logits = self.model(image)
                loss = self.loss_fn(hazards=hazards, S=S, Y=label, c=censorship)
            else: 
                output = self.model(image)
                loss = self.loss_fn(output, label)
            self.model_backward_and_update(loss)
            
        if self.cfg.TASK.NAME == "Survival":
            loss_summary = {
                "loss": loss.item(),
            }
        else:
            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }            

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
