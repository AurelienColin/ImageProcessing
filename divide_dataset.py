import glob
from tqdm import tqdm
import os
import fire

from Rignak_Misc.path import create_path

# python divide_dataset.py E:\datasets\waifu2latent

def divide_dataset(folder, train_to_val_ratio=10):
    labels = [os.path.join(folder, subfolder) for subfolder in os.listdir(folder) if os.path.isdir(os.path.join(folder, subfolder))]
    for label in tqdm(labels):
        train_folder = os.path.join(folder, 'train', os.path.split(label)[-1])
        val_folder = os.path.join(folder, 'val', os.path.split(label)[-1])
        create_path(train_folder)
        create_path(val_folder)
        for i, filename in tqdm(enumerate(glob.glob(os.path.join(label, '*.png')))):
            new_filename = os.path.join(train_folder, os.path.split(filename)[-1]) if i % train_to_val_ratio else os.path.join(val_folder, os.path.split(filename)[-1])
            os.rename(filename, new_filename)


if __name__ == '__main__':
    fire.Fire(divide_dataset)
