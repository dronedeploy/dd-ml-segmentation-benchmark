# run with either mask, category or string

import os
import click
import rasterio
import glob
import cv2
import sys
import time
import torch


from libs.config import LABELS, LABELMAP, INV_LABELMAP, train_ids, val_ids, test_ids

import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import numpy as np


def wherecolor(img, color, negate = False):

    k1 = (img[:, :, 0] == color[0])
    k2 = (img[:, :, 1] == color[1])
    k3 = (img[:, :, 2] == color[2])

    if negate:
        return np.where( not (k1 & k2 & k3) )
    else:
        return np.where( k1 & k2 & k3 )

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    labels_used = unique_labels(y_true, y_pred)
    classes = classes[labels_used]

    # Normalization with generate NaN where there are no ground label labels but there are predictions x/0
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.xlim([-0.5, cm.shape[1] - 0.5])
    plt.ylim([-0.5, cm.shape[0]- 0.5])

    fig.tight_layout()
    return fig, cm

def score_masks(labelfile, predictionfile):

    label = cv2.imread(labelfile)
    prediction = cv2.imread(predictionfile)

    shape = label.shape[:2]

    label_class = np.zeros(shape, dtype='uint8')
    pred_class  = np.zeros(shape, dtype='uint8')

    for color, category in INV_LABELMAP.items():
        locs = wherecolor(label, color)
        label_class[locs] = category

    for color, category in INV_LABELMAP.items():
        locs = wherecolor(prediction, color)
        pred_class[locs] = category

    label_class = label_class.reshape((label_class.shape[0] * label_class.shape[1]))
    pred_class = pred_class.reshape((pred_class.shape[0] * pred_class.shape[1]))

    # Remove all predictions where there is a IGNORE (magenta pixel) in the groud label and then shift labels down 1 index
    not_ignore_locs = np.where(label_class != 0)
    label_class = label_class[not_ignore_locs] - 1
    pred_class = pred_class[not_ignore_locs] - 1

    precision = precision_score(label_class, pred_class, average='weighted')
    recall = recall_score(label_class, pred_class, average='weighted')
    f1 = f1_score(label_class, pred_class, average='weighted')
    print(f'precision={precision} recall={recall} f1={f1}')

    fig, cm = plot_confusion_matrix(label_class, pred_class, np.array(LABELS), title=predictionfile)

    score_params = {
        'name' : 'inference_score',
        'predictionfile' : predictionfile,
        'precision' : precision,
        'recall' : recall,
        'f1' : f1,
        # 'cm' : cm #FIXME: This doesnt log successfully to wandb
    }

    return score_params

def score_model(dataset):

    import wandb

    for scene in train_ids + test_ids + val_ids:

        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        labelfile = f'{dataset}/labels/{scene}-label.png'
        predsfile = f"{scene}-prediction.png"

        if not os.path.exists(labelfile):
            continue

        if not os.path.exists(predsfile):
            continue

        score_params = score_masks(labelfile, predsfile)

        wandb.log(score_params)
        wandb.save(predsfile)


if __name__ == '__main__':
    score_model('dataset-sample')
