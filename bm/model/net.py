import torch
from torchvision.models import vgg19, vgg11, vgg16

encoder = vgg19(pretrained=True).eval().features


class SiameseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding = 2, padding_mode = 'reflect'),
            torch.nn.Conv2d(32, 32, 3, padding = 1, padding_mode = 'reflect'),
            torch.nn.Conv2d(32, 3, 5, padding = 2, padding_mode = 'reflect'),
            encoder,
            torch.nn.AvgPool2d(kernel_size=(4, 4), stride=(2, 2), padding = 1, count_include_pad=True),
            torch.nn.Flatten()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8192, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1),
        )

    def encode_image(self, image):
        return self.image_encoder(image)

    def predict(self, encoding_1, encoding_2):
        X = encoding_1 - encoding_2
        return self.decoder(X)

    def forward(self, image_1, image_2):
        encoding_1 = self.encode_image(image_1)
        encoding_2 = self.encode_image(image_2)
        return self.predict(encoding_1, encoding_2)

