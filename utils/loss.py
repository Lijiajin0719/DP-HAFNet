import torch
import torch.nn as nn


class CorrelationLoss(nn.Module):
    """Correlation Loss Function"""

    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, predictions, targets):
        """Calculate correlation loss"""
        batch_size = predictions.size(0)
        total_loss = 0.0

        for i in range(batch_size):
            x = predictions[i].flatten()
            y = targets[i].flatten()

            xy = x * y
            mean_xy = torch.mean(xy)
            mean_x = torch.mean(x)
            mean_y = torch.mean(y)
            cov_xy = mean_xy - mean_x * mean_y

            var_x = torch.sum((x - mean_x) ** 2) / x.size(0)
            var_y = torch.sum((y - mean_y) ** 2) / y.size(0)

            eps = 1e-8
            corr_xy = cov_xy / (torch.sqrt(var_x * var_y) + eps)

            total_loss += 1 - corr_xy

        return total_loss / batch_size


class CombinedLoss(nn.Module):
    """Combined Loss Function: MSE Loss + Correlation Loss"""

    def __init__(self, mse_weight=1.0, corr_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.corr_loss = CorrelationLoss()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight

    def forward(self, predictions, targets):
        """Calculate combined loss"""
        mse = self.mse_loss(predictions, targets)
        corr = self.corr_loss(predictions, targets)
        total_loss = self.mse_weight * mse + self.corr_weight * corr

        return total_loss, mse, corr


def get_dual_output_loss(outputs1, outputs2, targets1, targets2):
    """Calculate combined loss for dual outputs"""

    combined_loss = CombinedLoss(mse_weight=1.0, corr_weight=1.0)

    total_loss1, mse_loss1, corr_loss1 = combined_loss(outputs1, targets1)
    total_loss2, mse_loss2, corr_loss2 = combined_loss(outputs2, targets2)

    total_loss = 2 * total_loss1 + total_loss2

    return total_loss, total_loss1, total_loss2


if __name__ == "__main__":
    batch_size = 1
    channels = 1
    height = 609
    width = 387

    preds = torch.randn(batch_size, channels, height, width)
    targets = torch.randn(batch_size, channels, height, width)

    corr_loss_fn = CorrelationLoss()
    corr_loss = corr_loss_fn(preds, targets)
    print(f"Correlation loss: {corr_loss.item():.4f}")

    combined_loss_fn = CombinedLoss()
    total_loss, mse_loss, corr_loss = combined_loss_fn(preds, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"MSE loss: {mse_loss.item():.4f}")
    print(f"Correlation loss: {corr_loss.item():.4f}")

    preds2 = torch.randn(batch_size, channels, height, width)
    targets2 = torch.randn(batch_size, channels, height, width)
    total_loss, loss1, loss2 = get_dual_output_loss(preds, preds2, targets, targets2)
    print(f"Dual output total loss: {total_loss.item():.4f}")
    print(f"Output1 loss: {loss1.item():.4f}")
    print(f"Output2 loss: {loss2.item():.4f}")