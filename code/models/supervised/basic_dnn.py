from pl_bolts.datamodules import MNISTDataModule
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn,tensor
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class BasicDNNClassifier(LightningModule):
    def __init__(self, input_length: int, output_length: int):
        super().__init__()
        self.input_layer = nn.Linear(input_length, input_length)
        self.hidden_layer_1 = nn.Linear(input_length, 256)
        self.output_layer = nn.Linear(256, output_length)

    def forward(self, x: tensor):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1) # (batch_size, c*h*w)
        x = self.input_layer(x) # apply no function
        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x) 
        loss = F.nll_loss(logits, y)
        self.loss("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.loss("test_loss", loss) 
        return loss 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    net = BasicDNNClassifier(28*28, 10)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3)], logger=TensorBoardLogger("tb_logs", name="BasicDNNClassifer"))
    trainer.fit(net, datamodule=MNISTDataModule(num_workers=16))

