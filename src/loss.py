import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    def __init__(
        self,
        pos_weight=None,
        alpha: float = 0.0,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "none",
        device: str = "cuda",
    ):
        """
        Binary focal loss. To do: add label smoothing  param to control whether to smooth out positive, negative or both classes.
        Some inputs maybe soft already (positive class).

        """
        super(BinaryFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                reduction="none", pos_weight=torch.tensor([pos_weight]).to(device)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = self.criterion(inputs, targets)

        p = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
