import os
from fastai.vision import *
from fastai.callbacks.hooks import *
from pathlib import PosixPath

from libs.util import MySaveModelCallback, ExportCallback, MyCSVLogger, Precision, Recall, FBeta

import numpy as np
import libs.images2chips
import sys
import os

def download_dataset(dataset):
    if dataset == 'dataset-full':
        print("Full dataset has not been released yet :) ")
        sys.exit(0)

    if os.path.exists(dataset):
        print(f'Folder "{dataset}" already exists.')
    else:
        print(f'Downloading dataset "{dataset}"')
        os.system(f'curl "https://dl.dropboxusercontent.com/s/u3zrd1pgqxi0jvt/dataset-sample.tar.gz?dl=0" -o {dataset}.tar.gz')
        os.system(f'tar -xvf {dataset}.tar.gz')
        libs.images2chips.run(dataset)

def load_dataset(dataset, training_chip_size, bs):
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
