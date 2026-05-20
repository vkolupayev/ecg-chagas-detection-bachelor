# imports
from copy import deepcopy
from torch import save
from typing import Dict

# TODO Checkpoints save model state, optimizer state, scheduler state, etc.


# TODO Add tolerance
class EarlyStopping:
    """Early Stopping

    Possible metrics: ["Loss", "Recall", "Precision", "F1-Score", "F2-Score"]
    """

    def __init__(self, patience, maximize: bool = True, metric: str = "Recall"):
        self.patience = patience
        self.maximize = maximize
        self.metric = metric
        self.counter = 0
        self.best_epoch = 0
        self.best_metric = float("-inf") if maximize else float("inf")
        self.early_stop = False

    def __call__(self, current_metrics: Dict, current_epoch: int):
        current_metric = current_metrics[self.metric]
        if (self.maximize and current_metric > self.best_metric) or (
            not self.maximize and current_metric < self.best_metric
        ):
            self.best_metric = current_metric
            self.best_epoch = current_epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


class SaveModel:
    """Save Model

    Possible metrics: ["Loss", "Recall", "Precision", "F1-Score", "F2-Score"]
    """

    def __init__(
        self,
        save_path,
        model_name="best_model_weights",
        maximize=True,
        compiled_model=True,
        metric="Recall",
    ):
        self.save_path = save_path
        self.model_name = model_name
        self.maximize = maximize
        self.compiled_model = compiled_model
        self.metric = metric
        self.best_metric = float("-inf") if maximize else float("inf")
        self.best_model = None

    def __call__(self, current_metrics: Dict, model_weights):
        current_metric = current_metrics[self.metric]
        if (self.maximize and current_metric > self.best_metric) or (
            not self.maximize and current_metric < self.best_metric
        ):
            self.best_metric = current_metric
            self.best_model = deepcopy(model_weights)

    def save_model(self):
        if self.compiled_model:
            best_model_cpu = {
                k.replace("_orig_mod.", ""): v.cpu() for k, v in self.best_model.items()
            }
        else:
            best_model_cpu = {k: v.cpu() for k, v in self.best_model.items()}
        self.save_path.mkdir(parents=True, exist_ok=True)
        save(best_model_cpu, self.save_path / f"{self.model_name}.pth")
