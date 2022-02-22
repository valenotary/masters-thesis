from pl_bolts.datamodules import MNISTDataModule
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn,tensor
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

class BasicImageDNNClassifier(LightningModule):
    def __init__(self, input_width: int, input_height: int, hidden_layer_length: int, output_num_classes: int):
        super().__init__()
        self.input_layer = nn.Linear(input_width*input_height, input_width*input_height)
        self.hidden_layer_1 = nn.Linear(input_width*input_height, hidden_layer_length) # 256 neuron hidden layer
        self.output_layer = nn.Linear(hidden_layer_length, output_num_classes)

    def forward(self, x: tensor):
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1) # (batch_size, c*h*w)
        x = self.input_layer(x) # apply no function
        x = self.hidden_layer_1(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=1)
        return x
    
    def training_step(self, batch: tensor, batch_idx: int):
        x, y = batch 
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch 
        logits = self(x) 
        loss = F.nll_loss(logits, y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x) # self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("test_loss", loss) 
        return loss 

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    net = BasicImageDNNClassifier(input_width=28, input_height=28, hidden_layer_length=256, output_num_classes=10)
    trainer = Trainer(gpus=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min",patience=3)], logger=TensorBoardLogger("tb_logs", name="BasicDNNClassifer"))
    trainer.fit(net, datamodule=MNISTDataModule(num_workers=16))

