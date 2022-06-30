import torch
import pandas as pd
from torch.utils.data import DataLoader
from bm.model.net import SiameseNet
from bm.data.dataset2 import BacteriaCombinationsDataset, BacteriaCombinationsDataset_test
from bm.config import Config, ProjectPaths
import albumentations as A
import matplotlib.pyplot as plt

test_dataset = BacteriaCombinationsDataset_test(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize()]))
test_dataset.combinations_list = test_dataset.combinations_list[::]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
device='cpu'
model = torch.load('./model_checkpoints/test_model/epoch_16.pkl', map_location = device)

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
        ys.append(int(y[0]))
        preds.append(prediction[0][0])
        if ys[-1] == 0:
            preds0.append(preds[-1])
plt.scatter(ys, preds)
plt.scatter([0]*len(preds0), preds0, c = 'red')
plt.ylim(sorted(preds)[5] - 0.000001, sorted(preds)[-5] + 0.000001)
plt.savefig('pure_test_sigm.jpg', dpi = 600)
plt.clf()





test_dataset = BacteriaCombinationsDataset_test(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize(), A.transforms.Flip(always_apply = True, p = 0.5)]))
test_dataset.combinations_list = test_dataset.combinations_list[::]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
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
        ys.append(int(y[0]))
        preds.append(prediction[0][0])
        if ys[-1] == 0:
            preds0.append(preds[-1])
plt.scatter(ys, preds)
plt.scatter([0]*len(preds0), preds0, c = 'red')
plt.ylim(sorted(preds)[5] - 0.000001, sorted(preds)[-5] + 0.000001)
plt.savefig('flip_test_sigm.jpg', dpi = 600)
plt.close()


test_dataset = BacteriaCombinationsDataset_test(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize(), A.transforms.Flip(always_apply = True, p = 0.5), A.RandomRotate90(always_apply = True, p = 0.5)]))
test_dataset.combinations_list = test_dataset.combinations_list[::]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
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
        ys.append(int(y[0]))
        preds.append(prediction[0][0])
        if ys[-1] == 0:
            preds0.append(preds[-1])
plt.scatter(ys, preds)
plt.scatter([0]*len(preds0), preds0, c = 'red')
plt.ylim(sorted(preds)[5] - 0.000001, sorted(preds)[-5] + 0.000001)
plt.savefig('flip_rotate_test_sigm.jpg', dpi = 600)
plt.close()


test_dataset = BacteriaCombinationsDataset_test(
    pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv')),
    transforms=A.Compose([A.Resize(256, 256), A.Normalize(), A.RandomRotate90(always_apply = True, p = 0.5)]))
test_dataset.combinations_list = test_dataset.combinations_list[::]
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
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
        ys.append(int(y[0]))
        preds.append(prediction[0][0])
        if ys[-1] == 0:
            preds0.append(preds[-1])
plt.scatter(ys, preds)
plt.scatter([0]*len(preds0), preds0, c = 'red')
plt.ylim(sorted(preds)[5] - 0.000001, sorted(preds)[-5] + 0.000001)
plt.savefig('rotate_test_sigm.jpg', dpi = 600)
plt.close()
