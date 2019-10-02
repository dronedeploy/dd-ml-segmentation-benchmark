from typing import Any
from fastai.callbacks import CSVLogger, SaveModelCallback, TrackerCallback
from fastai.callback import Callback
from fastai.metrics import add_metrics
from fastai.torch_core import dataclass, torch, Tensor, Optional, warn
from fastai.basic_train import Learner


class ExportCallback(TrackerCallback):
    """"Exports the model when monitored quantity is best.

    The exported model is the one used for inference.
    """
    def __init__(self, learn:Learner, model_path:str, monitor:str='valid_loss', mode:str='auto'):
        self.model_path = model_path
        super().__init__(learn, monitor=monitor, mode=mode)

    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        current = self.get_monitor_value()

        if (epoch == 0 or (current is not None and self.operator(current, self.best))):
            print(f'Better model found at epoch {epoch} with {self.monitor} value: {current} - exporting {self.model_path}')
            self.best = current
            self.learn.export(self.model_path)

# TODO: does this delete some other path or just overwrite?
class MySaveModelCallback(SaveModelCallback):
    """Saves the model after each epoch to potentially resume training.

    Modified from fastai version to delete the previous model that was saved
    to avoid wasting disk space.
    """
    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        "Compare the value monitored to its best score and maybe save the model."
        current = self.get_monitor_value()
        if current is not None and self.operator(current, self.best):
            self.best = current
            self.learn.save(f'{self.name}')


class MyCSVLogger(CSVLogger):
    """Logs metrics to a CSV file after each epoch.

    Modified from fastai version to:
    - flush after each epoch
    - append to log if already exists
    """
    def __init__(self, learn, filename='history'):
        super().__init__(learn, filename)

    def on_train_begin(self, **kwargs):
        if self.path.exists():
            # TODO: does this open a file named "a"...?
            self.file = self.path.open('a')
        else:
            super().on_train_begin(**kwargs)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        out = super().on_epoch_end(
            epoch, smooth_loss, last_metrics, **kwargs)
        self.file.flush()
        return out

# The following are a set of metric callbacks that have been modified from the
# original version in fastai to support semantic segmentation, which doesn't
# have the class dimension in position -1. It also adds an ignore_idx
# which is used to ignore pixels with class equal to ignore_idx. These
# would be good to contribute back upstream to fastai -- however we should
# wait for their upcoming refactor of the callback architecture.

@dataclass
class ConfusionMatrix(Callback):
    "Computes the confusion matrix."
    # The index of the dimension in the output and target arrays which ranges
    # over the different classes. This is -1 (the last index) for
    # classification, but is 1 for semantic segmentation.
    clas_idx:int=-1

    def on_train_begin(self, **kwargs):
        self.n_classes = 0


    def on_epoch_begin(self, **kwargs):
        self.cm = None

    def on_batch_end(self, last_output:Tensor, last_target:Tensor, **kwargs):


        preds = last_output.argmax(self.clas_idx).view(-1).cpu()
        targs = last_target.view(-1).cpu()
        if self.n_classes == 0:
            self.n_classes = last_output.shape[self.clas_idx]
            self.x = torch.arange(0, self.n_classes)
        cm = ((preds==self.x[:, None]) & (targs==self.x[:, None, None])).sum(dim=2, dtype=torch.float32)
        if self.cm is None: self.cm =  cm
        else:               self.cm += cm

    def on_epoch_end(self, **kwargs):
        self.metric = self.cm

@dataclass
class CMScores(ConfusionMatrix):
    "Base class for metrics which rely on the calculation of the precision and/or recall score."
    average:Optional[str]="binary"      # `binary`, `micro`, `macro`, `weighted` or None
    pos_label:int=1                     # 0 or 1
    eps:float=1e-9
    # If ground truth label is equal to the ignore_idx, it should be ignored
    # for the sake of evaluation.
    ignore_idx:int=None

    def _recall(self):
        rec = torch.diag(self.cm) / self.cm.sum(dim=1)
        rec[rec != rec] = 0  # removing potential "nan"s
        if self.average is None: return rec
        else:
            if self.average == "micro": weights = self._weights(avg="weighted")
            else: weights = self._weights(avg=self.average)
            return (rec * weights).sum()

    def _precision(self):
        prec = torch.diag(self.cm) / self.cm.sum(dim=0)
        prec[prec != prec] = 0  # removing potential "nan"s
        if self.average is None: return prec
        else:
            weights = self._weights(avg=self.average)
            return (prec * weights).sum()

    def _weights(self, avg:str):
        if self.n_classes != 2 and avg == "binary":
            avg = self.average = "macro"
            warn("average=`binary` was selected for a non binary case. Value for average has now been set to `macro` instead.")
        if avg == "binary":
            if self.pos_label not in (0, 1):
                self.pos_label = 1
                warn("Invalid value for pos_label. It has now been set to 1.")
            if self.pos_label == 1: return Tensor([0,1])
            else: return Tensor([1,0])
        else:
            if avg == "micro": weights = self.cm.sum(dim=0) / self.cm.sum()
            if avg == "macro": weights = torch.ones((self.n_classes,)) / self.n_classes
            if avg == "weighted": weights = self.cm.sum(dim=1) / self.cm.sum()
            if self.ignore_idx is not None and avg in ["macro", "weighted"]:
                weights[self.ignore_idx] = 0
                weights /= weights.sum()
            return weights

class Recall(CMScores):
    "Compute the Recall."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._recall())

class Precision(CMScores):
    "Compute the Precision."
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, self._precision())

@dataclass
class FBeta(CMScores):
    "Compute the F`beta` score."
    beta:float=2

    def on_train_begin(self, **kwargs):
        self.n_classes = 0
        self.beta2 = self.beta ** 2
        self.avg = self.average
        if self.average != "micro": self.average = None

    def on_epoch_end(self, last_metrics, **kwargs):
        prec = self._precision()
        rec = self._recall()
        metric = (1 + self.beta2) * prec * rec / (prec * self.beta2 + rec + self.eps)
        metric[metric != metric] = 0  # removing potential "nan"s
        if self.avg: metric = (self._weights(avg=self.avg) * metric).sum()
        return add_metrics(last_metrics, metric)

    def on_train_end(self, **kwargs): self.average = self.avg
