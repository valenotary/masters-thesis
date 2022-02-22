from pl_bolts.datamodules import MNISTDataModule
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn,tensor
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class BasicImageRBM(LightningModule):
    def __init__(self, input_width: int, input_height:int, hidden_layer_length: int):
        pass