import pathlib
import torch
# import clearml
import torchmetrics
from tqdm import tqdm
from bm.config import Config


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 n_epochs: int,
                 train_loader,
                 val_loader,
                 metrics: torchmetrics.MetricCollection = None,
                 logger = None,
#                  logger: clearml.Logger = None,
                 model_save_path: pathlib.Path = None,
                 device=Config.device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metrics = metrics if metrics is not None else torchmetrics.MetricCollection([])
        self.metrics.to(device)
        self.logger = logger
        self.model_save_path = model_save_path
        self.device = device

    def _prepare_data_from_dataloader(self, data):
        images_1, images_2, y = data
        images_1 = torch.permute(images_1.to(self.device).float(), [0, 3, 1, 2])
        images_2 = torch.permute(images_2.to(self.device).float(), [0, 3, 1, 2])
        y = y.to(self.device)
        return (images_1, images_2, y), len(y)

    def _get_prediction(self, data):
        prediction = self.model(data[0], data[1])
        prediction = torch.sigmoid(prediction)
        return prediction

    def _compute_loss(self, prediction, data):
        return self.criterion(prediction, data[2])

    def _update_metrics(self, prediction, data):
        self.metrics.update(prediction.detach(), data[2].int())

    def _log_metrics(self, series, epoch_idx):
        if self.logger is None:
            return
        metric_values = self.metrics.compute()
        for metric_name, value in metric_values.items():
            self.logger.report_scalar(metric_name, series, value, epoch_idx)

    def _save_model(self, epoch_idx: int):
        if self.model_save_path is not None:
            self.model_save_path.mkdir(parents=True, exist_ok=True)
            torch.save(self.model, self.model_save_path.joinpath(f'epoch_{epoch_idx}.pkl'))

    def train_one_epoch(self, epoch_idx: int):
        self.model.train()
        self.metrics.reset()
        running_loss = 0
        for data in self.train_loader:
            data, batch_size = self._prepare_data_from_dataloader(data)
            self.optimizer.zero_grad()
            prediction = self._get_prediction(data)
            loss = self._compute_loss(prediction, data)
            running_loss += loss.item() * batch_size
            loss.backward()
            self.optimizer.step()
            self._update_metrics(prediction, data)
        self._log_metrics('train', epoch_idx)
        print('')
        print(running_loss / len(self.train_loader.dataset), epoch_idx)
        if self.logger is not None:
            self.logger.report_scalar('loss', 'train', running_loss / len(self.train_loader.dataset), epoch_idx)

    def validate_one_epoch(self, epoch_idx: int):
        self.model.eval()
        self.metrics.reset()
        running_loss = 0
        for data in self.val_loader:
            data, batch_size = self._prepare_data_from_dataloader(data)
            with torch.no_grad():
                prediction = self._get_prediction(data)
                loss = self._compute_loss(prediction, data)
            running_loss += loss.item() * batch_size
            self._update_metrics(prediction, data)
        self._log_metrics('val', epoch_idx)
        if self.logger is not None:
            self.logger.report_scalar('loss', 'val', running_loss / len(self.train_loader.dataset), epoch_idx)
        print(running_loss / len(self.val_loader.dataset), epoch_idx)
        self._save_model(epoch_idx)

    def run_training(self):
        for epoch_idx in tqdm(range(self.n_epochs)):
            self.train_one_epoch(epoch_idx)
            self.validate_one_epoch(epoch_idx)
