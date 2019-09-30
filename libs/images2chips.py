import cv2
import os
import numpy as np

from libs.config import train_ids, val_ids, test_ids, LABELMAP, INV_LABELMAP

size   = 300
stride = 300

def color2class(orthochip, img):
    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    ret = np.dstack([ret, ret, ret])
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # Skip any chips that would contain magenta (IGNORE) pixels
    seen_colors = set( [tuple(color) for color in colors] )
    IGNORE_COLOR = LABELMAP[0]
    if IGNORE_COLOR in seen_colors:
        return None, None

    for color in colors:
        locs = np.where( (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]) )
        ret[ locs[0], locs[1], : ] = INV_LABELMAP[ tuple(color) ] - 1

    return orthochip, ret

def image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile, windowx=size, windowy=size, stridex=stride, stridey=stride):

    ortho = cv2.imread(orthofile)
    label = cv2.imread(labelfile)

    # Not using elevation in the sample - but useful to incorporate it ;)
    eleva = cv2.imread(elevafile, -1)

    assert(ortho.shape[0] == label.shape[0])
    assert(ortho.shape[1] == label.shape[1])

    shape = ortho.shape

    xsize = shape[1]
    ysize = shape[0]
    print(f"converting {dataset} image {orthofile} {xsize}x{ysize} to chips ...")

    counter = 0

    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):

            orthochip = ortho[yi:yi+windowy, xi:xi+windowx, :]
            labelchip = label[yi:yi+windowy, xi:xi+windowx, :]

            orthochip, classchip = color2class(orthochip, labelchip)

            if classchip is None:
                continue

            orthochip_filename = os.path.join(prefix, 'image-chips', scene + '-' + str(counter).zfill(6) + '.png')
            labelchip_filename = os.path.join(prefix, 'label-chips', scene + '-' + str(counter).zfill(6) + '.png')

            with open(f"{prefix}/{dataset}", mode='a') as fd:
                fd.write(scene + '-' + str(counter).zfill(6) + '.png\n')

            cv2.imwrite(orthochip_filename, orthochip)
            cv2.imwrite(labelchip_filename, classchip)
            counter += 1


def get_split(scene):
    if scene in train_ids:
        return "train.txt"
    if scene in val_ids:
        return 'valid.txt'
    if scene in test_ids:
        return 'test.txt'

def run(prefix):

    open(prefix + '/train.txt', mode='w').close()
    open(prefix + '/valid.txt', mode='w').close()
    open(prefix + '/test.txt', mode='w').close()

    if not os.path.exists( os.path.join(prefix, 'image-chips') ):
        os.mkdir(os.path.join(prefix, 'image-chips'))

    if not os.path.exists( os.path.join(prefix, 'label-chips') ):
        os.mkdir(os.path.join(prefix, 'label-chips'))


    lines = [ line for line in open(f'{prefix}/index.csv') ]
    num_images = len(lines) - 1
    print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")

    for lineno, line in enumerate(lines):

        line = line.strip().split(' ')
        scene = line[1]
        dataset = get_split(scene)

        if dataset == 'test.txt':
            print(f"not converting test image {scene} to chips, it will be used for inference.")
            continue

        orthofile = os.path.join(prefix, 'images',     scene + '-ortho.tif')
        elevafile = os.path.join(prefix, 'elevations', scene + '-elev.tif')
        labelfile = os.path.join(prefix, 'labels',     scene + '-label.png')

        if os.path.exists(orthofile) and os.path.exists(labelfile):
            image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile)
