from .build import TRAINER_REGISTRY, build_trainer  # isort:skip
from .trainer import Trainer,  TrainerBase, SimpleTrainer, SimpleNet  # isort:skip
from .abmil import ABMIL
from .transmil import TransMIL
from .snn import SNN
from .mcat import MCAT
from .cmta import CMTA
from .porpoise import Porpoise
from .hfb import HFB
from .concat import ConCAT
from .add import ADD
from .clipomic import CLIPOMIC
from .snnm import SNNM
from .mbtrain import MBTRAIN
# from .mlmf import MLMF