# https://pytorch-lightning.readthedocs.io/en/latest/notebooks/course_UvA-DL/08-deep-autoencoders.html


# TODO: make it more general for n channels, rather than assuming just one 

# TODO: make models infer input/output dimensions based on dataloader parameters ?

from pl_bolts.datamodules import MNISTDataModule
from pytorch_lightning import LightningDataModule, Trainer
import pytorch_lightning
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn, tensor
import torch
from torch.nn import functional as F
from torch.optim import Adam
import torchvision


class _BasicImageEncoder(nn.module):
    def __init__(self, 
                input_width: int, 
                input_height: int, 
                hidden_layer_length: int, 
                latent_dim_length: int):
        super().__init__()
        self.input_layer = nn.Linear(input_width*input_height, input_width*input_height)
        self.hidden_layer_1 = nn.Linear(input_width*input_height, hidden_layer_length) 
        self.latent_layer = nn.Linear(hidden_layer_length, latent_dim_length)

    def forward(self, x: tensor):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1) # (batch_size, c*h*w)
        x = self.input_layer(x) # apply no function
        x = self.hidden_layer_1(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        return x

class _BasicImageDecoder(nn.module):
    def __init__(self, 
            input_latent_length: int,
            hidden_layer_length: int, 
            output_width: int, 
            output_height: int):
        super().__init__()
        self.latent_layer = nn.Linear(input_latent_length, input_latent_length)
        self.hidden_layer_1 = nn.Linear(input_latent_length, hidden_layer_length) 
        self.output_layer = nn.Linear(hidden_layer_length, output_width*output_height)
    
    def forward(self, x: tensor):
        x = self.latent_layer
        x = self.hidden_layer_1(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        return x

class BasicImageAE(LightningModule):

    def __init__(self, 
                    input_width: int,
                    input_height: int,
                    hidden_layer_lengths: int, # for both 
                    latent_dim_length: int):
        
        self.encoder = _BasicImageEncoder(input_width=input_width,
                                            input_height=input_height,
                                            hidden_layer_length=hidden_layer_lengths,
                                            latent_dim_length=latent_dim_length)
        self.decoder = _BasicImageDecoder(input_latent_length=latent_dim_length,
                                            hidden_layer_length=hidden_layer_lengths,
                                            output_width=input_width,output_height=input_height)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(x)
        return x_hat 

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, _ = batch  # We do not need the labels
        x_hat = self(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum().mean()
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

def get_train_images(dataset: LightningDataModule, num: int):
    return torch.stack([dataset[i][0] for i in range(num)], dim=0)

class GenerateCallback(pytorch_lightning.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0, 1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1, 1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


if __name__ == '__main__':
    dataset = MNISTDataModule(num_workers=16)
    net = BasicImageAE(input_width=28, input_height=28, hidden_layer_lengths=256, latent_dim_length=8)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3), GenerateCallback(input_imgs=get_train_images(dataset=dataset,num=8),every_n_epochs=1)], logger=TensorBoardLogger("tb_logs", name="BasicImageAE"))
    trainer.fit(net, datamodule=dataset)

# TODO: do I cut the AE in half and feed that into a classifier, or do I feed the entire network in ?