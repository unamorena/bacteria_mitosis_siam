import pandas as pd
from bm.training.trainer import Trainer
from bm.training.utils import set_seed
from bm.config import Config, ProjectPaths
from bm.data.dataset import BacteriaCombinationsDataset, BacteriaCombinationsDataset2
from bm.model.net import SiameseNet
from torchmetrics import MetricCollection, AUROC, Precision, Recall, Accuracy
from torch.utils.data import DataLoader
import torch
import albumentations as A

set_seed()
torch.cuda.empty_cache()

train_dataset = BacteriaCombinationsDataset2(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/train.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize()]))
val_dataset = BacteriaCombinationsDataset2(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/val.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize()]))
train_dataset.combinations_list = train_dataset.combinations_list[::10]
val_dataset.combinations_list = val_dataset.combinations_list[::]

train_loader, val_loader = DataLoader(train_dataset, batch_size=4, shuffle=True), DataLoader(val_dataset, batch_size=4)
print('train_dataset', len(train_dataset))
print('val_dataset', len(val_dataset))

model = SiameseNet()

model.decoder.requires_grad_(True)
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=0.00005)
n_epochs = 20
metrics = MetricCollection({'auroc': AUROC(1),
                            'precision': Precision(1),
                            'recall': Recall(1),
                            'accuracy': Accuracy()})

model_save_path = ProjectPaths.checkpoints_dir.joinpath('test_model')

trainer = Trainer(model,
                  optimizer,
                  torch.nn.BCELoss(),
                  n_epochs,
                  train_loader,
                  val_loader,
                  metrics,
                  None,
                  model_save_path,
                 torch.device("cuda"))

if __name__ == '__main__':
    trainer.run_training()
