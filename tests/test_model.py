import pytest
import torch
from bm.model.net import SiameseNet


class TestSiameseNet:
    @pytest.mark.parametrize('image_height', [512, 256])
    @pytest.mark.parametrize('image_width', [512, 256])
    @pytest.mark.parametrize('image_channels', [1, 3])
    def test_prediction_shape(self, image_height: int, image_width: int, image_channels: int):
        model = SiameseNet(image_height, image_width, image_channels)
        model.eval()
        image = torch.randn((image_channels, image_height, image_width))[None, ...]
        out = model(image, image)
        assert out.shape == torch.Size([1, 1])

    def test_is_trainable(self):
        image = torch.randn((1, 3, 512, 512))
        model = SiameseNet(512, 512, 3)
        model.train()
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        out1 = torch.sigmoid(model(image, image))
        loss = criterion(out1, torch.Tensor([[1]]))
        loss.backward()
        optimizer.step()
        out2 = torch.sigmoid(model(image, image))
        assert out1 != out2
