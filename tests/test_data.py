import pytest
from bm.data.dataset import BacteriaCombinationsDataset
import pandas as pd
from bm.config import ProjectPaths
import albumentations as A

dataframe = pd.read_csv(ProjectPaths.data_dir.joinpath('processed/test.csv'))


class TestCombinationDataset:
    def test_size(self):
        dataset = BacteriaCombinationsDataset(dataframe)
        assert len(dataset) > 0

    @pytest.mark.parametrize('side_len', [512, 218])
    def test_transforms(self, side_len):
        transforms = A.Compose([
            A.Resize(side_len, side_len, p=1),
        ])
        dataset = BacteriaCombinationsDataset(dataframe, transforms=transforms)
        image1, image2, _ = dataset[0]
        assert image1.shape == (side_len, side_len, 3)
        assert image2.shape == (side_len, side_len, 3)
