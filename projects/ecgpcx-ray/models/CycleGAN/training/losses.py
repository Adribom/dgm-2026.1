import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """
    LSGAN loss used in CycleGAN.
    """

    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, prediction, target_is_real):

        if target_is_real:
            target = torch.ones_like(prediction)
        else:
            target = torch.zeros_like(prediction)

        return self.loss(prediction, target)


class CycleConsistencyLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, reconstructed, real):
        return self.loss(reconstructed, real)


class IdentityLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss = nn.L1Loss()

    def forward(self, same_domain_output, real):
        return self.loss(same_domain_output, real)


class MaskedCycleConsistencyLoss(nn.Module):
    """
    Cycle consistency loss with independent weighting for inside-lung
    (mask=1) and outside-lung (mask=0) regions.
    """

    def __init__(self, lambda_inside=1.0, lambda_outside=1.0):
        super().__init__()
        self.lambda_inside = lambda_inside
        self.lambda_outside = lambda_outside

    def forward(self, reconstructed, real, mask):
        outside = 1.0 - mask
        abs_diff = torch.abs(reconstructed - real)
        loss_inside = (abs_diff * mask).sum() / mask.sum().clamp_min(1.0)
        loss_outside = (abs_diff * outside).sum() / outside.sum().clamp_min(1.0)
        return self.lambda_inside * loss_inside + self.lambda_outside * loss_outside


class MaskedIdentityLoss(nn.Module):
    """
    Identity loss with independent weighting for inside-lung and
    outside-lung regions.
    """

    def __init__(self, lambda_inside=1.0, lambda_outside=1.0):
        super().__init__()
        self.lambda_inside = lambda_inside
        self.lambda_outside = lambda_outside

    def forward(self, same_output, real, mask):
        outside = 1.0 - mask
        abs_diff = torch.abs(same_output - real)
        loss_inside = (abs_diff * mask).sum() / mask.sum().clamp_min(1.0)
        loss_outside = (abs_diff * outside).sum() / outside.sum().clamp_min(1.0)
        return self.lambda_inside * loss_inside + self.lambda_outside * loss_outside