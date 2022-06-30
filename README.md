# bacteria_mitosis_siam

Siamese network to predict difference between osfdataset https://osf.io/befnx/ pictures of reproducing bacteria

Pipeline:

1. Split train/test dataset with pipelines/generate_split.py
2. Run to train model bm/model/train.py
3. Run model and visualize result test_run.py

Result example: x-axis pic_1_numerical_label - pic_0_numerical_label, y-axis float value predict

![flip_test (2)](https://user-images.githubusercontent.com/92823229/176651943-2373b24c-fb04-4c4b-813b-3dabc77945b2.jpg)
