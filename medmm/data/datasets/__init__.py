from .build import DATASET_REGISTRY, build_dataset  # isort:skip
from .base_dataset import Datum, DatasetBase  # isort:skip

from .classification import TCGA_Glioma_C, TCGA_Glioma_G, TCGA_Glioma_Sub

from .survival import TCGA_Glioma_S