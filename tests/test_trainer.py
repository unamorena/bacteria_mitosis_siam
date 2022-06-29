import clearml
import pandas as pd

from bm.training.trainer import Trainer
from bm.training.utils import set_seed
from bm.config import Config, ProjectPaths
from bm.data.dataset import BacteriaCombinationsDataset
from bm.model.net import SiameseNet
from torchmetrics import MetricCollection, AUROC, Precision
from torch.utils.data import DataLoader
import torch
import albumentations as A

set_seed()

dataset = BacteriaCombinationsDataset(pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
                                      transforms=A.Compose([A.Resize(512, 512)]))
dataset.combinations_list = dataset.combinations_list[:10]
train_loader, val_loader = DataLoader(dataset, batch_size=10), DataLoader(dataset, batch_size=10)

model = SiameseNet(512, 512, 3)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCELoss()
n_epochs = 10
metrics = MetricCollection({'auroc': AUROC(1)})
task = clearml.Task.init(Config.project_name, 'test', tags=['test'])
logger = task.get_logger()
model_save_path = ProjectPaths.checkpoints_dir.joinpath('test_model')

trainer = Trainer(model, optimizer, criterion, n_epochs, train_loader,
                  val_loader, metrics, logger, model_save_path, 'cpu')


class TestTrainer:
    def test_train_one_epoch(self):
        trainer.train_one_epoch(0)

    def test_validate_one_epoch(self):
        trainer.validate_one_epoch(0)
