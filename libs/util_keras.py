from keras.metrics import Metric
from keras import backend as K
from keras.utils import metrics_utils
import numpy as np

# adapted from keras.metrics.Precision
class FBeta(Metric):
    def __init__(self,
                 beta=1,
                 name=None,
                 dtype=None):
        super(FBeta, self).__init__(name=name, dtype=dtype)
        self.beta2 = beta*beta
        self.true_positives = self.add_weight(
            'true_positives',
            shape=(1,),
            initializer='zeros')
        self.false_positives = self.add_weight(
            'false_positives',
            shape=(1,),
            initializer='zeros')
        self.false_negatives = self.add_weight(
            'false_negatives',
            shape=(1,),
            initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=[metrics_utils.NEG_INF],
            top_k=1,
            class_id=None,
            sample_weight=sample_weight)

    def _precision(self):
        denom = (self.true_positives + self.false_positives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))
        return result[0]

    def _recall(self):
        denom = (self.true_positives + self.false_negatives)
        result = K.switch(
            K.greater(denom, 0),
            self.true_positives / denom,
            K.zeros_like(self.true_positives))
        return result[0]

    def result(self):
        precision, recall = self._precision(), self._recall()
        denom = self.beta2 * precision + recall
        result = K.switch(
            K.greater(denom, 0),
            (1 + self.beta2) * precision * recall / denom,
            0.)
        return result

    def reset_states(self):
        K.batch_set_value(
            [(v, np.zeros((1,))) for v in self.weights])

    def get_config(self):
        config = {
            'beta2': self.beta2
        }
        base_config = super(FBeta, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))