# all credits for this implementation go to https://github.com/AntixK/PyTorch-VAE
# and https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
from pl_bolts.datamodules import MNISTDataModule
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn, tensor, flatten, exp, randn_like, distributions, zeros_like, ones_like
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class BasicImageVAETorch(LightningModule):
    def __init__(self,
                input_width: int, 
                input_height: int, 
                hidden_layer_length: int, 
                latent_dim_length: int,
                kl_coeff: float = 0.1,
                lr: float = 1e-4):
        super().__init__()
        self.lr = lr 
        self.kl_coeff = kl_coeff
        self.encoder = nn.Sequential(
            nn.Linear(input_width*input_height, input_width*input_height),
            nn.Linear(input_width*input_height, hidden_layer_length)
        )
        self.fc_mu = nn.Linear(hidden_layer_length,latent_dim_length)
        self.fc_var = nn.Linear(hidden_layer_length,latent_dim_length)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_length, hidden_layer_length),
            nn.Linear(hidden_layer_length, input_width*input_height)
        )

    def forward(self, input: tensor) ->list[tensor]:
        input = self.encoder(input)
        mu = self.fc_mu(input) 
        log_var = self.fc_var(input) 
        p, q, z = self.sample(mu, log_var) 
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = exp(log_var/2)
        p = distributions.Normal(zeros_like(mu), ones_like(std))
        q = distributions.Normal(mu, std)
        z = q.rsample() # da trick 
        return p, q, z

    def _get_reconstruction_loss(self, batch, batch_idx):
        x, _ = batch 
        z, x_hat, p, q = self(x) 
        
        recon_loss= F.mse_loss(x_hat, x, reduction="mean")
        kl = distributions.kl_divergence(q, p).mean()
        kl *= self.kl_coeff

        loss = kl + recon_loss 
        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        loss, logs = self._get_reconstruction_loss(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss 

    def validation_step(self, batch, batch_idx):
        loss, logs = self._get_reconstruction_loss(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss 

    def test_step(self, batch, batch_idx):
        loss, logs = self._get_reconstruction_loss(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss 
    
if __name__ == '__main__':
    dataset = MNISTDataModule(num_workers=16)
    net = BasicImageVAETorch(input_width=28, input_height=28, hidden_layer_length=256, latent_dim_length=8)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3)], logger=TensorBoardLogger("tb_logs", name="BasicImageVAETorch"))
    trainer.fit(net, datamodule=dataset)
        
                