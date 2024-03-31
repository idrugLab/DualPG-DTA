from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import numpy as np


class ConcordanceIndex(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds).cpu().tolist()
        target = dim_zero_cat(self.target).cpu().tolist()

        z = 0.0
        h = 0.0
        for i in range(1, len(target)):
            for j in range(0, i):
                if target[i] > target[j]:
                    z = z + 1
                    h += 1.0 * (preds[i] > preds[j]) + 0.5 * (preds[i] == preds[j])

        return h / z if z != 0 else 0.0


class R2mIndex(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor):
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds).cpu().numpy()
        target = dim_zero_cat(self.target).cpu().numpy()

        preds_mean = np.array([np.mean(preds) for _ in preds])
        target_mean = np.array([np.mean(target) for _ in target])

        mult = np.sum((preds - preds_mean) * (target - target_mean)) ** 2
        r2 = mult / (
            np.sum((preds - preds_mean) ** 2) * np.sum((target - target_mean) ** 2)
        )

        k = np.sum(preds * target) / np.sum(preds**2)
        r20 = 1 - (
            np.sum((target - (k * preds)) * (target - (k * preds)))
            / np.sum((target - target_mean) * (target - target_mean))
        )

        return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r20 * r20))))
