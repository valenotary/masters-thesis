# credits go to https://github.com/ml-jku/hopfield-layers/blob/f56f929c95b77a070ae675ea4f56b6d54d36e730/examples/bit_pattern/bit_pattern_demo.ipynb
# Importing Hopfield-specific modules.
from hflayers import Hopfield, HopfieldLayer, HopfieldPooling
from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import (distributions, exp, flatten, nn, ones_like, randn_like,
                   tensor, zeros_like)
from torch.nn import functional as F
from torch.optim import AdamW


class BasicHopfieldNet(LightningModule):
    def __init__(self, 
                input_output_width: int, 
                input_output_height: int, 
                hidden_layer_length: int):
        super().__init__()
        self._hopnet = nn.Sequential(Hopfield(input_size = input_output_width * input_output_height, 
                                                hiden_size = hidden_layer_length),
                                    nn.Flatten()) # may not need this 


    def forward(self, x: tensor) -> list[tensor]: 
        return self._hopnet(x)

    def _get_reconstruction_loss(self, batch, batch_idx):
        x, _ = batch 
        x_hat = self(x) 
        print(f"x.shape: {x.shape}; x_hat.shape: {x_hat.shape}")
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        return recon_loss 

    def configure_optimizers(self):
        return AdamW(self.parametrers())
    
    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch, batch_idx) 
        self.log("train_loss", loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch, batch_idx) 
        self.log("val_loss", loss)
        return loss 


if __name__ == '__main__':
    dataset = MNISTDataModule(num_workers=16)
    net = BasicHopfieldNet(input_width=28, input_height=28, hidden_layer_length=256)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3)], logger=TensorBoardLogger("tb_logs", name="BasicImageHopfield"))
    trainer.fit(net, datamodule=dataset)
