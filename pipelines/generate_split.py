"""
Split file has the following structure:

| sequence_id |    image_fp   | frame_idx |
|-------------|---------------|-----------|
|     int     | relative path |    int    |
"""
import pathlib
import pandas as pd
from bm.config import ProjectPaths, Config
from sklearn.model_selection import train_test_split
import re


def generate_osf_dataframe(dataset_dir: pathlib.Path = ProjectPaths.data_dir.joinpath('raw', 'osf_dataset'),
                           default_data_dir: pathlib.Path = ProjectPaths.data_dir):
    """
    Returns dataframe for osf dataset
    """
    data = []

    for fp in dataset_dir.iterdir():
        sequence_id, frame_idx = re.findall(r'\d+_book-(\w)-(\d+)\.tif', fp.name)[0]
        image_fp = fp.as_posix().replace(default_data_dir.as_posix(), '')[1:]
        data.append({
            'sequence_id': sequence_id,
            'image_fp': image_fp,
            'frame_idx': int(frame_idx)
        })
    return pd.DataFrame(data)


def split_dataframe(df: pd.DataFrame,
                    train_size=Config.train_size,
                    val_size=Config.val_size,
                    random_state=Config.random_seed):
    """
    Splits dataframe to train, validation and test sets
    """
    df_train, df_val_test = train_test_split(df, stratify=df['sequence_id'],
                                             train_size=train_size, random_state=random_state)
    df_val, df_test = train_test_split(df_val_test, stratify=df_val_test['sequence_id'],
                                       train_size=val_size / (1 - train_size),
                                       random_state=random_state)
    return df_train, df_val, df_test


def main():
    """
    generates all dataframes, splits and saves them.
    :return:
    """
    df = generate_osf_dataframe()
    df_train, df_val, df_test = split_dataframe(df)
    df_train.to_csv(ProjectPaths.data_dir.joinpath('processed/train.csv'), index=False)
    df_val.to_csv(ProjectPaths.data_dir.joinpath('processed/val.csv'), index=False)
    df_test.to_csv(ProjectPaths.data_dir.joinpath('processed/test.csv'), index=False)


if __name__ == '__main__':
    main()
