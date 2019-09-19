import cv2
import sys
import torch
from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils import *
from fastai.vision.transform import dihedral

CATEGORY2MASK = {
    0:  (255, 0, 255),
    1 : (75, 25, 230),
    2 : (180, 30, 145),
    3 : (75, 180, 60),
    4 : (48, 130, 245),
    5 : (255, 255, 255),
    6 : (200, 130, 0),
}

def category2mask(img):

    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in CATEGORY2MASK.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask


def chip_iterator(image, label=None, size=256):

    img = cv2.imread(image)

    if label is not None:
        print("loading label file too ...")
        label = cv2.imread(label)

    shape = img.shape

    chip_count = 0
    for xi, x in enumerate(range(0, shape[1], size)):
        for yi, y in enumerate(range(0, shape[0], size)):
            chip_count += 1

    print('chips =', chip_count )

    ret = []
    for xi, x in enumerate(range(0, shape[1], size)):
        for yi, y in enumerate(range(0, shape[0], size)):
            chip = img[y:y+size, x:x+size, :]
            chip = cv2.copyMakeBorder(chip, top=0, bottom=size - chip.shape[0], left=0, right=size - chip.shape[1], borderType= cv2.BORDER_CONSTANT, value=[0, 0, 0] )
            labelchip = None
            if label is not None:
                labelchip = label[y:y+size, x:x+size, :]
                labelchip = cv2.copyMakeBorder(labelchip, top=0, bottom=size - labelchip.shape[0], left=0, right=size - labelchip.shape[1], borderType= cv2.BORDER_CONSTANT, value=[250, 0, 250] )

            ret.append((chip, xi, yi, labelchip, chip_count))

    random.shuffle(ret)

    for k in ret:
        yield k

def image_size(filename):
    img = cv2.imread(filename)
    return img.shape

def tensor2numpy(tensor):
    ret = tensor.px.numpy()
    ret = ret * 255.
    ret = ret.astype('uint8')
    ret = np.transpose(ret, (1, 2, 0))
    return ret


class Inference(object):

    def __init__(self, modelfile, predsfile, size=1200):
        print("loading model", modelfile, "...")
        self.learn = load_learner('.', modelfile)
        self.predsfile = predsfile
        self.learn.data.single_ds.tfmargs['size'] = size
        self.learn.data.single_ds.tfmargs_y['size'] = size

    def predict(self, imagefile, labelfile=None, size=1200):

        shape = image_size(imagefile)
        print('loading input image', shape)

        assert(shape[2] == 3)

        prediction = np.zeros(shape[:2], dtype='uint8')

        iter = chip_iterator(imagefile, label=labelfile, size=size)

        for counter, (imagechip, x, y, labelchip, total_chips) in enumerate(iter):

            print(f"running inference on chip {counter} of {total_chips}")

            if imagechip.sum() == 0:
                continue

            mask = category2mask(prediction)
            cv2.imwrite(self.predsfile, mask)

            tensorchip = np.transpose(imagechip, (2, 0, 1))
            tensorchip = tensorchip.astype('float32')
            tensorchip = tensorchip / 255.
            tensorchip = torch.from_numpy(tensorchip)
            tensorchip = Image(tensorchip)
            preds =  self.learn.predict(tensorchip)[2]
            # add one because we don't predict the ignore class
            category_chip = preds.data.argmax(0).numpy() + 1
            section = prediction[y*size:y*size+size, x*size:x*size+size].shape
            prediction[y*size:y*size+size, x*size:x*size+size] = category_chip[:section[0], :section[1]]

        mask = category2mask(prediction)
        cv2.imwrite(self.predsfile, mask)

if __name__ == '__main__':

    model_name = sys.argv[1]
    scene = sys.argv[2]
    size = 1200
    dataset = 'dataset-sample'

    imagefile = f'{dataset}/images/{scene}-ortho.tif'
    labelfile = f'{dataset}/labels/{scene}-label.png'
    predsfile =  "prediction.png"

    inf = Inference(model_name, predsfile, size=size)
    inf.predict(imagefile, labelfile=labelfile, size=size)
