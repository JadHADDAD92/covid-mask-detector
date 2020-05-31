""" Training module
"""
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.init as init
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.nn import (Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU,
                      Sequential)
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .dataset import MaskDataset


class MaskDetector(pl.LightningModule):
    """ MaskDetector PyTorch Lightning class
    """
    def __init__(self, maskDFPath: Path=None):
        super(MaskDetector, self).__init__()
        self.maskDFPath = maskDFPath
        
        self.maskDF = None
        self.trainDF = None
        self.validateDF = None
        self.crossEntropyLoss = None
        self.learningRate = 0.00001
        
        self.convLayer1 = convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer2 = convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.convLayer3 = convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3,3)),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )
        
        self.linearLayers = linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            ReLU(),
            Linear(in_features=1024, out_features=2),
        )
        
        # Initialize layers' weights
        for sequential in [convLayer1, convLayer2, convLayer3, linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)
    
    def forward(self, x: Tensor): # pylint: disable=arguments-differ
        """ forward pass
        """
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
        out = self.linearLayers(out)
        return out
    
    def prepare_data(self) -> None:
        self.maskDF = maskDF = pd.read_pickle(self.maskDFPath)
        train, validate = train_test_split(maskDF, test_size=0.3, random_state=0,
                                           stratify=maskDF['mask'])
        self.trainDF = MaskDataset(train)
        self.validateDF = MaskDataset(validate)
        
        # Create weight vector for CrossEntropyLoss
        maskNum = maskDF[maskDF['mask']==1].shape[0]
        nonMaskNum = maskDF[maskDF['mask']==0].shape[0]
        nSamples = [nonMaskNum, maskNum]
        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        self.crossEntropyLoss = CrossEntropyLoss(weight=torch.tensor(normedWeights))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.trainDF, batch_size=32, shuffle=True, num_workers=4)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.validateDF, batch_size=32, num_workers=4)
    
    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learningRate)
    
    def training_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]: # pylint: disable=arguments-differ
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)
        
        tensorboardLogs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboardLogs}
    
    def validation_step(self, batch: dict, _batch_idx: int) -> Dict[str, Tensor]: # pylint: disable=arguments-differ
        inputs, labels = batch['image'], batch['mask']
        labels = labels.flatten()
        outputs = self.forward(inputs)
        loss = self.crossEntropyLoss(outputs, labels)
        
        _, outputs = torch.max(outputs, dim=1)
        valAcc = accuracy_score(outputs.cpu(), labels.cpu())
        valAcc = torch.tensor(valAcc)
        
        return {'val_loss': loss, 'val_acc':valAcc}
    
    def validation_epoch_end(self, outputs: List[Dict[str, Tensor]]) \
            -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        avgLoss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avgAcc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboardLogs = {'val_loss': avgLoss, 'val_acc':avgAcc}
        return {'val_loss': avgLoss, 'log': tensorboardLogs}

if __name__ == '__main__':
    model = MaskDetector(Path('covid-mask-detector/data/mask_df.pickle'))
    
    checkpoint_callback = ModelCheckpoint(
        filepath='covid-mask-detector/checkpoints/weights.ckpt',
        save_weights_only=True,
        verbose=True,
        monitor='val_acc',
        mode='max'
    )
    trainer = Trainer(gpus=1 if torch.cuda.is_available() else 0,
                      max_epochs=10,
                      checkpoint_callback=checkpoint_callback,
                      profiler=True)
    trainer.fit(model)
