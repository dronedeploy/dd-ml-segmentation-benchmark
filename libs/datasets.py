import os
from fastai.vision import *
from fastai.callbacks.hooks import *
from pathlib import PosixPath

import numpy as np
from libs.config import LABELS
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
        print(f"unknown dataset {dataset}")
        sys.exit(0)

    filename = f'{dataset}.tar.gz'
    url = URLS[dataset]

    if not os.path.exists(filename):
        print(f'downloading dataset "{dataset}"')
        os.system(f'curl "{url}" -o {filename}')
    else:
        print(f'zipfile "{filename}" already exists, remove it if you want to re-download.'')

    if not os.path.exists(dataset):
        print(f'extracting "{filename}"')
        os.system(f'tar -xvf {filename}')
    else:
        print(f'folder "{dataset}" already exists, remove it if you want to re-create.')

    image_chips = f'{dataset}/image-chips'
    label_chips = f'{dataset}/label-chips'
    if not os.path.exists(image_chips) and not os.path.exists(label_chips):
        print("creating chips")
        libs.images2chips.run(dataset)
    else:
        print(f'chip folders "{image_chips}" and "{label_chips}" already exist, remove them to recreate chips.')

def load_dataset(dataset, training_chip_size, bs):
    """ Load a dataset, create batches and augmentation """

    path = PosixPath(dataset)
    label_path = path/'label-chips'
    image_path = path/'image-chips'
    image_files = get_image_files(image_path)
    label_files = get_image_files(label_path)
    get_y_fn = lambda x: label_path/f'{x.stem}{x.suffix}'
    codes = np.array(LABELS)
    src = SegmentationItemList.from_folder(image_path).split_by_fname_file('../valid.txt').label_from_func(get_y_fn, classes=codes)
    # some data augmentation here
    # data = src.transform(get_transforms(flip_vert=True, max_warp=0., max_zoom=0., max_rotate=180.), size=training_chip_size, tfm_y=True).databunch(bs=bs)
    data = src.transform(get_transforms(flip_vert=True), size=training_chip_size, tfm_y=True).databunch(bs=bs)
    return data
