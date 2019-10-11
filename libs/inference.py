"""
    inference.py - Sample implementation of inference with a Dynamic Unet using FastAI
    2019 - Nicholas Pilkington, DroneDeploy
"""

import os
import cv2
import sys
import torch
from libs.scoring import score_masks
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils import *
from libs.config import train_ids, test_ids, val_ids, LABELMAP

def category2mask(img):
    """ Convert a category image to color mask """
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in LABELMAP.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask


def chip_iterator(image, size=256):
    """ Generator that yields chips of size `size`x`size from `image` """

    img = cv2.imread(image)
    shape = img.shape

    chip_count = math.ceil(shape[1] / size) * math.ceil(shape[0] / size)

    for xi, x in enumerate(range(0, shape[1], size)):
        for yi, y in enumerate(range(0, shape[0], size)):
            chip = img[y:y+size, x:x+size, :]
            # Padding right and bottom out to `size` with black pixels
            chip = cv2.copyMakeBorder(chip, top=0, bottom=size - chip.shape[0], left=0, right=size - chip.shape[1], borderType= cv2.BORDER_CONSTANT, value=[0, 0, 0] )
            yield (chip, xi, yi, chip_count)

def image_size(filename):
    img = cv2.imread(filename)
    return img.shape

def tensor2numpy(tensor):
    """ Convert a pytorch tensor image presentation to numpy OpenCV representation """

    ret = tensor.px.numpy()
    ret = ret * 255.
    ret = ret.astype('uint8')
    ret = np.transpose(ret, (1, 2, 0))
    return ret

def numpy2tensor(chip):
    tensorchip = np.transpose(chip, (2, 0, 1))
    tensorchip = tensorchip.astype('float32')
    tensorchip = tensorchip / 255.
    tensorchip = torch.from_numpy(tensorchip)
    tensorchip = Image(tensorchip)
    return tensorchip

class Inference(object):

    def __init__(self, modelpath, modelfile, size=1200):
        print("loading model", modelfile, "...")
        self.learn = load_learner(modelpath, modelfile)
        self.learn.data.single_ds.tfmargs['size'] = size
        self.learn.data.single_ds.tfmargs_y['size'] = size

    def predict(self, imagefile, predsfile, size=1200):

        shape = image_size(imagefile)
        print('loading input image', shape)

        assert(shape[2] == 3)

        prediction = np.zeros(shape[:2], dtype='uint8')

        iter = chip_iterator(imagefile, size=size)

        for counter, (imagechip, x, y, total_chips) in enumerate(iter):

            print(f"running inference on chip {counter} of {total_chips}")

            if imagechip.sum() == 0:
                continue

            tensorchip = numpy2tensor(imagechip)
            preds =  self.learn.predict(tensorchip)[2]
            # add one because we don't predict the ignore class
            category_chip = preds.data.argmax(0).numpy() + 1
            section = prediction[y*size:y*size+size, x*size:x*size+size].shape
            prediction[y*size:y*size+size, x*size:x*size+size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        cv2.imwrite(predsfile, mask)

def run_inference(dataset, model_name='baseline_model', basedir="predictions"):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)

    size = 1200
    modelpath = 'models'

    if not os.path.exists(os.path.join(modelpath, model_name)):
        print(f"model {model_name} not found in {modelpath}")
        sys.exit(0)

    inf = Inference(modelpath, model_name, size=size)

    for scene in train_ids + val_ids + test_ids:
    #for scene in test_ids:

        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        labelfile = f'{dataset}/labels/{scene}-label.png'
        predsfile = f"{basedir}/{scene}-prediction.png"

        if not os.path.exists(imagefile):
            #print(f"image {imagefile} not found, skipping.")
            continue

        print(f"running inference on image {imagefile}.")
        inf.predict(imagefile, predsfile, size=size)


if __name__ == '__main__':
    run_inference()
