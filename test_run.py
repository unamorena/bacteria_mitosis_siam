import torch
import pandas as pd
from torch.utils.data import DataLoader
from bm.model.net import SiameseNet
from bm.data.dataset2 import BacteriaCombinationsDataset
from bm.config import Config, ProjectPaths
import albumentations as A
import matplotlib.pyplot as plt

test_dataset = BacteriaCombinationsDataset(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize()]))
test_dataset.combinations_list = test_dataset.combinations_list[::]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
device='cpu'
model = torch.load('./model_checkpoints/test_model/epoch_19.pkl', map_location = device)

ys = []
preds, preds0 = [], []
for data in test_loader:
    images_1, images_2, y = data
    images_1 = torch.permute(images_1.to(device).float(), [0, 3, 1, 2])
    images_2 = torch.permute(images_2.to(device).float(), [0, 3, 1, 2])
    y = y.to(device)
    
    data, batch_size = (images_1, images_2, y), len(y)
    with torch.no_grad():
        prediction = model(data[0], data[1])
        prediction = torch.sigmoid(prediction)
#         prediction = test_loader._get_prediction(data)
        ys.append(int(y[0]))
        #print(prediction[0][0])
        preds.append(prediction[0][0])
        if ys[-1] == 0:
            preds0.append(preds[-1])
#                 loss = self._compute_loss(prediction, data)
#             running_loss += loss.item() * batch_size
#             self._update_metrics(prediction, data)
plt.scatter(ys, preds)
plt.scatter([0]*len(preds0), preds0, c = 'red')
plt.ylim(sorted(preds)[5] - 0.000001, sorted(preds)[-5] + 0.000001)
#plt.ylim(0.15 - 0.000001, -0.15 + 0.000001)
plt.show()
