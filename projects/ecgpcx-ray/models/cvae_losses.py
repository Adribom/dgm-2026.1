import torch
import torch.nn as nn
import torch.nn.functional as F


def split_image_and_mask(x):
    x_img = x[:, 0:1, :, :]
    x_mask = x[:, 1:2, :, :].clamp(0, 1) if x.size(1) > 1 else torch.ones_like(x_img)
    return x_img, x_mask


def kl_loss(mu, logvar):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=1).mean()


def masked_l1_loss(x, y, mask):
    return (torch.abs(x - y) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_ssim_loss(x, y, mask, window_size=7):
    padding = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, window_size, stride=1, padding=padding) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, window_size, stride=1, padding=padding) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=padding) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    )
    loss_map = (1.0 - ssim_map).clamp(0, 2)
    return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)


def masked_reconstruction_loss(x, y, mask, ssim_weight=0.5):
    l1 = masked_l1_loss(x, y, mask)
    ssim = masked_ssim_loss(x, y, mask)
    return (1.0 - ssim_weight) * l1 + ssim_weight * ssim


class VGGPerceptualLoss(nn.Module):
    def __init__(self, enabled=True, layer_ids=(3, 8, 15)):
        super().__init__()
        self.enabled = enabled
        self.layer_ids = set(layer_ids)

        if not enabled:
            self.vgg = None
            return

        try:
            from torchvision.models import VGG16_Weights, vgg16

            weights = VGG16_Weights.IMAGENET1K_FEATURES
            self.vgg = vgg16(weights=weights).features.eval()
        except Exception:
            from torchvision.models import vgg16

            self.vgg = vgg16(weights=None).features.eval()

        for param in self.vgg.parameters():
            param.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _prepare(self, x):
        x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    def forward(self, x, y, mask=None):
        if not self.enabled or self.vgg is None:
            return torch.zeros((), device=x.device)

        if mask is not None:
            x = x * mask
            y = y * mask

        x_features = self._prepare(x)
        y_features = self._prepare(y)
        loss = torch.zeros((), device=x.device)

        for idx, layer in enumerate(self.vgg):
            x_features = layer(x_features)
            y_features = layer(y_features)
            if idx in self.layer_ids:
                loss = loss + F.l1_loss(x_features, y_features)
            if idx >= max(self.layer_ids):
                break

        return loss


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))
        self.register_buffer("laplacian", laplacian.view(1, 1, 3, 3))

    def _gradients(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        lap = F.conv2d(x, self.laplacian, padding=1)
        return gx, gy, lap

    def forward(self, x, y):
        x_gx, x_gy, x_lap = self._gradients(x)
        y_gx, y_gy, y_lap = self._gradients(y)
        return (
            F.l1_loss(x_gx, y_gx)
            + F.l1_loss(x_gy, y_gy)
            + 0.5 * F.l1_loss(x_lap, y_lap)
        )


def generator_gan_loss(discriminator, fake):
    pred_fake = discriminator(fake)
    return F.mse_loss(pred_fake, torch.ones_like(pred_fake))


def discriminator_loss(discriminator, real, fake):
    pred_real = discriminator(real)
    pred_fake = discriminator(fake.detach())
    loss_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
    loss_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
    return 0.5 * (loss_real + loss_fake)


def validation_image_score(x, x_hat):
    x_img, x_mask = split_image_and_mask(x)
    outside_mask = 1.0 - x_mask
    inside_ssim = 1.0 - masked_ssim_loss(x_img, x_hat, x_mask)
    outside_ssim = 1.0 - masked_ssim_loss(x_img, x_hat, outside_mask)
    return 0.5 * inside_ssim + 0.5 * outside_ssim


class RobustCVAELoss(nn.Module):
    def __init__(
        self,
        alpha_outside=5.0,
        beta_inside=1.0,
        gamma_perceptual=0.05,
        delta_gan=0.02,
        eta_gradient=0.2,
        beta_kl=0.005,
        use_vgg=True,
    ):
        super().__init__()
        self.alpha_outside = alpha_outside
        self.beta_inside = beta_inside
        self.gamma_perceptual = gamma_perceptual
        self.delta_gan = delta_gan
        self.eta_gradient = eta_gradient
        self.beta_kl = beta_kl
        self.perceptual = VGGPerceptualLoss(enabled=use_vgg)
        self.gradient = GradientLoss()

    def forward(self, x, x_hat, mu, logvar, discriminator=None):
        x_img, x_mask = split_image_and_mask(x)
        outside_mask = 1.0 - x_mask

        rec_outside = masked_reconstruction_loss(x_img, x_hat, outside_mask)
        rec_inside = masked_reconstruction_loss(x_img, x_hat, x_mask)
        perceptual = self.perceptual(x_img, x_hat, outside_mask)
        gan = generator_gan_loss(discriminator, x_hat) if discriminator is not None else torch.zeros((), device=x.device)
        grad = self.gradient(x_img, x_hat)
        kl = kl_loss(mu, logvar)

        total = (
            self.alpha_outside * rec_outside
            + self.beta_inside * rec_inside
            + self.gamma_perceptual * perceptual
            + self.delta_gan * gan
            + self.eta_gradient * grad
            + self.beta_kl * kl
        )

        return {
            "total": total,
            "rec_outside": rec_outside,
            "rec_inside": rec_inside,
            "perceptual": perceptual,
            "gan": gan,
            "gradient": grad,
            "kl": kl,
        }
