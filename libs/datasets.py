import os
from fastai.vision import *
from fastai.callbacks.hooks import *
from pathlib import PosixPath

import numpy as np
import libs.images2chips
import sys
import os

URLS = {
    'dataset-sample' : 'https://dl.dropboxusercontent.com/s/h8a8kev0rktf4kq/dataset-sample.tar.gz?dl=0',
    'dataset-medium' : 'https://dl.dropboxusercontent.com/s/r0dj9mhyv4bgbme/dataset-medium.tar.gz?dl=0',
}

def download_dataset(dataset):
    """ Download a dataset, extract it and create the tiles """

    if dataset not in URLS:
        print(f"Unknown dataset {dataset}")
        sys.exit(0)

    filename = f'{dataset}.tar.gz'
    url = URLS[dataset]

    if not os.path.exists(filename):
        print(f'Downloading dataset "{dataset}"')
        os.system(f'curl "{url}" -o {filename}')

    if not os.path.exists(dataset):
        print(f'extracting "{filename}"')
        os.system(f'tar -xvf {filename}')
    else:
        print(f'Folder "{dataset}" already exists.')

    print("Creating chips")
    libs.images2chips.run(dataset)

def load_dataset(dataset, training_chip_size, bs):
    """ Load a dataset, create batches and augmentation """

    path = PosixPath(dataset)
    label_path = path/'label-chips'
    image_path = path/'image-chips'
    image_files = get_image_files(image_path)
    label_files = get_image_files(label_path)
    get_y_fn = lambda x: label_path/f'{x.stem}{x.suffix}'
    codes = np.array(['BUILDING', 'CLUTTER', 'VEGETATION', 'WATER', 'GROUND', 'CAR'])
    src = SegmentationItemList.from_folder(image_path).split_by_fname_file('../valid.txt').label_from_func(get_y_fn, classes=codes)
    # some data augmentation here
    # data = src.transform(get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=180.), size=training_chip_size, tfm_y=True).databunch(bs=bs)
    data = src.transform(get_transforms(flip_vert=True), size=training_chip_size, tfm_y=True).databunch(bs=bs)
    return data
