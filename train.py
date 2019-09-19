from fastai.vision import *
from fastai.callbacks.hooks import *
from util import MySaveModelCallback, ExportCallback, MyCSVLogger, Precision, Recall, FBeta
import datasets


def run(dataset):

    import wandb
    from wandb.fastai import WandbCallback
    wandb.init()

    epochs = 15
    lr = 1e-4
    size = 300
    wd = 1e-2
    bs = 16
    pretrained = True

    metrics = [
        Precision(average='weighted', clas_idx=1),
        Recall(average='weighted', clas_idx=1),
        FBeta(average='weighted', beta=1, clas_idx=1),
    ]

    data = datasets.load_dataset(dataset, size, bs)
    encoder_model = models.resnet18
    learn = unet_learner(data, encoder_model, metrics=metrics, wd=wd, bottle=True, pretrained=pretrained, callback_fns=WandbCallback)

    callbacks = [
        MyCSVLogger(learn, filename='example_model'),
        ExportCallback(learn, "example_model", monitor='f_beta'),
        MySaveModelCallback(learn, every='epoch', monitor='f_beta')
    ]

    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr, callbacks=callbacks)


if __name__ == '__main__':

    # Change this to 'dataset-full' for the full dataset
    dataset = 'dataset-sample'
    datasets.download_dataset(dataset)

    run(dataset)
