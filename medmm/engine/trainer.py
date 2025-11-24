import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from medmm.data import DataManager
from medmm.optim import build_optimizer, build_lr_scheduler
from medmm.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from medmm.modeling import build_head, build_backbone
from medmm.evaluation import build_evaluator

from sksurv.util import Surv

from medmm.modeling.models.umeml_gan import UMEML_GAN
import matplotlib.pyplot as plt


def plot_importance_matrix(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    data = np.array([[float(x) for x in line.strip().split()] for line in lines])  # shape: (N, 6)

    height, width = data.shape  # height = N, width = 6
    dpi = 100
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, cmap='Blues', aspect='auto', interpolation='nearest')

    ax.axis('off')

    png_path = txt_path.replace('txt', 'png')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def cca_loss(X, Y, epsilon=1e-8):

    X_centered = X - X.mean(0)
    Y_centered = Y - Y.mean(0)

    X_std = X_centered / X_centered.std(0, unbiased=False)
    Y_std = Y_centered / Y_centered.std(0, unbiased=False)

    C_xy = X_std.T @ Y_std / (X.size(0) - 1)

    u, s, v = torch.svd(C_xy)
    
    return 1 / (s.sum() / s.numel() + epsilon)

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )
        fdim = self.backbone.out_features

        self.head = None
        if model_cfg.HEAD.NAME and model_cfg.HEAD.HIDDEN_LAYERS:
            self.head = build_head(
                model_cfg.HEAD.NAME,
                verbose=cfg.VERBOSE,
                in_features=fdim,
                hidden_layers=model_cfg.HEAD.HIDDEN_LAYERS,
                activation=model_cfg.HEAD.ACTIVATION,
                bn=model_cfg.HEAD.BN,
                dropout=model_cfg.HEAD.DROPOUT,
                **kwargs,
            )
            fdim = self.head.out_features

        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(fdim, num_classes)

        self._fdim = fdim

    @property
    def fdim(self):
        return self._fdim

    def forward(self, x, return_feature=False):
        f = self.backbone(x)
        if self.head is not None:
            f = self.head(f)

        if self.classifier is None:
            return f

        y = self.classifier(f)

        if return_feature:
            return y, f

        return y


class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch, umeml_gan_test_without_omic_ratio=-1, umeml_gan_test_insample_without_omic_ratio=0):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        # calculate omic mean in train dataset
        omic_means = []
        for self.batch_idx, batch in enumerate(self.train_loader):
            omic_means.append(batch["mol"])
        omic_means = torch.concat(omic_means, dim=0)
        omic_means = torch.mean(omic_means, dim=0)
        self.omic_means = omic_means

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            if self.epoch < 3:
                self.run_epoch(train_gan=False, replace_ratio=0)
            elif self.epoch < 5:
                self.run_epoch(train_gan=True, replace_ratio=0)
            else:
                self.run_epoch(train_gan=True, replace_ratio=(self.epoch + 1 - 5) / (self.max_epoch + 1 - 5) / 2)
            if self.epoch < self.max_epoch - 1:
                self.run_epoch_cca(train_gan=False, replace_ratio=0)
            self.after_epoch(umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio, omic_means=self.omic_means)
        self.after_train(umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio, omic_means=self.omic_means)

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def run_epoch_cca(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch(self, batch):
        raise NotImplementedError


    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg):
        super().__init__()
        self.check_cfg(cfg)

        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

        if self.cfg.TASK.NAME == "Survival":

            train_survival_month, train_censorship = self.train_loader.dataset.get_envent_and_cenorship()
            val_survival_month, val_censorship = self.val_loader.dataset.get_envent_and_cenorship()

            all_censorships = np.concatenate(
                [train_censorship, val_censorship], axis=0)
            all_event_times = np.concatenate(
                [train_survival_month, val_survival_month], axis=0)

            self.all_survival = Surv.from_arrays(
                event=(1-all_censorships).astype(bool), time=all_event_times)
            
            # import pdb;pdb.set_trace()
            self.evaluator = build_evaluator(
                cfg, all_survival=self.all_survival, bins=None)

        else:
            self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)

        # if self.cfg.TASK.NAME == "Survival":

        #     train_survival_month, train_censorship = self.train_loader.dataset.get_envent_and_cenorship()
        #     val_survival_month, val_censorship = self.val_loader.dataset.get_envent_and_cenorship()

        #     all_censorships = np.concatenate(
        #         [train_censorship, val_censorship], axis=0)
        #     all_event_times = np.concatenate(
        #         [train_survival_month, val_survival_month], axis=0)

        #     self.all_survival = Surv.from_arrays(
        #         event=(1-all_censorships).astype(bool), time=all_event_times)
            
        #     # import pdb;pdb.set_trace()
        #     self.evaluator = build_evaluator(
        #         cfg, all_survival=self.all_survival, bins=self.dm.bins)

        # else:
        #     self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        
        
        # if self.cfg.TASK.NAME == "Survival":
        #     self.train_sdata = []
        #     self.test_sdata = []
        #     for batch_idx, batch in enumerate(self.train_loader):
        #         image, omic, label, survival_month, censorship = self.parse_batch(batch)
        #         self.train_sdata.append(survival_month.cpu().numpy().tolist() + (1-censorship.cpu().numpy()).tolist())
        #     for batch_idx, batch in enumerate(self.test_loader):
        #         image, omic, label, survival_month, censorship = self.parse_batch(batch)
        #         self.test_sdata.append(survival_month.cpu().numpy().tolist() + (1-censorship.cpu().numpy()).tolist())

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader = dm.train_loader
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = SimpleNet(cfg, cfg.MODEL, self.num_classes)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        print(f"# params: {count_num_param(self.model):,}")
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Detected {device_count} GPUs (use nn.DataParallel)")
            self.model = nn.DataParallel(self.model)

    def train(self, umeml_gan_test_without_omic_ratio=-1, umeml_gan_test_insample_without_omic_ratio=0):
        super().train(self.start_epoch, self.max_epoch, umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self, umeml_gan_test_without_omic_ratio=-1, umeml_gan_test_insample_without_omic_ratio=0, omic_means=None):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test(umeml_gan_test_without_omic_ratio=umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio=umeml_gan_test_insample_without_omic_ratio, omic_means=omic_means)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self, umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio=0, omic_means=None):
        last_epoch = (self.epoch + 1) == self.max_epoch
        # import pdb;pdb.set_trace()
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )
        
        if do_test:
            curr_result = self.test(split="val", umeml_gan_test_without_omic_ratio=umeml_gan_test_without_omic_ratio, umeml_gan_test_insample_without_omic_ratio=umeml_gan_test_insample_without_omic_ratio, omic_means=omic_means)
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                is_best = curr_result > self.best_result
                if is_best:
                    self.best_result = curr_result
                    self.save_model(
                        self.epoch,
                        self.output_dir,
                        val_result=curr_result,
                        model_name="model-best.pth.tar"
                    )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None, umeml_gan_test_without_omic_ratio=0, umeml_gan_test_insample_without_omic_ratio=0, omic_means=None):
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

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch(batch)
            
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch(self, batch):
        patient_id = batch["patient_id"]
        img = batch["img"]
        mol = batch["mol"]
        label = batch["label"]
        if self.cfg.TASK.NAME == "Survival":
            survival_months = batch["survival_month"]
            censorship = batch["censorship"]
        else:
            survival_months = torch.empty(1)
            censorship = torch.empty(1)
        
        img = img.to(self.device)
        mol = mol.to(self.device)
        label = label.to(self.device)
        survival_months = survival_months.to(self.device)
        censorship = censorship.to(self.device)
        return patient_id, img, mol, label, survival_months, censorship

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]



class Trainer(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self, train_gan=False, replace_ratio=0):

        open("train_path.txt", "w").close()
        open("train_omic.txt", "w").close()

        if isinstance(self.model, UMEML_GAN):
            self.model.plot_set = "train"

        self.model.cca = False

        self.model.train_gan = train_gan
        self.model.replace_ratio = replace_ratio
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        # plot importances
        plot_importance_matrix("train_path.txt")
        plot_importance_matrix("train_omic.txt")

    def run_epoch_cca(self, train_gan=False, replace_ratio=0):
        self.model.train_gan = train_gan
        self.model.replace_ratio = replace_ratio
        self.model.cca = True
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader)

        cca_optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        end = time.time()
        h_path_list = []
        h_omic_list = []
        batch_loss_list = []
        list_length_threshold = 64
        for self.batch_idx, batch in enumerate(self.train_loader):

            h_path, h_omic, batch_loss = self.forward_backward(batch)
            h_path = h_path.view(h_path.shape[0], -1)
            h_omic = h_omic.view(h_omic.shape[0], -1)
            # h_omic = h_omic.repeat(1, 2)
            h_path_list.append(h_path)
            h_omic_list.append(h_omic)
            batch_loss_list.append(batch_loss)
            
            if self.batch_idx == self.num_batches - 1 or len(h_path_list) == list_length_threshold:
                data_time.update(time.time() - end)
                cca_optimizer.zero_grad()
                h_path_list = torch.cat(h_path_list, dim=0)
                h_omic_list = torch.cat(h_omic_list, dim=0)
                batch_loss_list = torch.stack(batch_loss_list)
                batch_loss = torch.mean(batch_loss_list)
                cca_loss_val = cca_loss(h_path_list, h_omic_list)
                loss = cca_loss_val + batch_loss
                loss.backward()
                cca_optimizer.step()
                batch_time.update(time.time() - end)
                loss_summary = {
                    "cca_loss": cca_loss_val.item(),
                    "batch_loss": batch_loss.item()
                }
                losses.update(loss_summary)

                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

                del h_path_list, h_omic_list
                h_path_list, h_omic_list = [], []
                batch_loss_list = []

        self.model.cca = False
