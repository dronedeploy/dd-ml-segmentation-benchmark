from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence, to_categorical
from PIL import Image

import numpy as np
import random

def load_dataset(dataset, bs, aug={'horizontal_flip': True, 'vertical_flip': True, 'rotation_range': 180}):
    train_files = [f'{dataset}/image-chips/{fname}' for fname in load_lines(f'{dataset}/train.txt')]
    valid_files = [f'{dataset}/image-chips/{fname}' for fname in load_lines(f'{dataset}/valid.txt')]
    
    train_seq = SegmentationSequence(
        dataset,
        train_files, 
        ImageDataGenerator(**aug),
        bs
    )
    
    valid_seq = SegmentationSequence(
        dataset,
        valid_files,
        ImageDataGenerator(), # don't augment validation set
        bs
    )
    
    return train_seq, valid_seq

def load_lines(fname):
    with open(fname, 'r') as f:
        return [l.strip() for l in f.readlines()]

def load_img(fname):
    return np.array(Image.open(fname))

def mask_to_classes(mask):
    return to_categorical(mask[:,:,0], 6)

class SegmentationSequence(Sequence):
    def __init__(self, dataset, image_files, datagen, bs):
        self.label_path = f'{dataset}/label-chips'
        self.image_path = f'{dataset}/image-chips'
        self.image_files = image_files
        random.shuffle(self.image_files)

        self.datagen = datagen
        self.bs = bs

    def __len__(self):
        return int(np.ceil(len(self.image_files) / float(self.bs)))

    def __getitem__(self, idx):
        image_files = self.image_files[idx*self.bs:(idx+1)*self.bs]
        label_files = [fname.replace(self.image_path, self.label_path) for fname in image_files]

        images = [load_img(fname) for fname in image_files]
        labels = [mask_to_classes(load_img(fname)) for fname in label_files]

        ts = [self.datagen.get_random_transform(im.shape) for im in images]
        images = [self.datagen.apply_transform(im, ts) for im, ts in zip(images, ts)]
        labels = [self.datagen.apply_transform(im, ts) for im, ts in zip(labels, ts)]

        return np.array(images), np.array(labels)

    def on_epoch_end(self):
        random.shuffle(self.image_files)
