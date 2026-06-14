import torch
from tqdm import tqdm

from .losses import (
    GANLoss,
    CycleConsistencyLoss,
    IdentityLoss,
    MaskedCycleConsistencyLoss,
    MaskedIdentityLoss,
)

from .image_buffer import ImageBuffer


class CycleGANTrainer:

    def __init__(
        self,
        model,
        device,
        lambda_cycle=10.0,
        lambda_identity=5.0,
        use_mask=False,
        lambda_mask_inside=1.0,
        lambda_mask_outside=1.0,
    ):

        self.device = device
        self.use_mask = use_mask

        self.G_H2P = model.G_H2P.to(device)
        self.G_P2H = model.G_P2H.to(device)

        self.D_H = model.D_H.to(device)
        self.D_P = model.D_P.to(device)

        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        # Losses
        self.gan_loss = GANLoss()

        if use_mask:
            self.cycle_loss = MaskedCycleConsistencyLoss(lambda_mask_inside, lambda_mask_outside)
            self.identity_loss = MaskedIdentityLoss(lambda_mask_inside, lambda_mask_outside)
        else:
            self.cycle_loss = CycleConsistencyLoss()
            self.identity_loss = IdentityLoss()

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            list(self.G_H2P.parameters()) +
            list(self.G_P2H.parameters()),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        self.optimizer_D_H = torch.optim.Adam(
            self.D_H.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        self.optimizer_D_P = torch.optim.Adam(
            self.D_P.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999)
        )

        # Replay buffers
        self.fake_healthy_buffer = ImageBuffer()
        self.fake_pneumonia_buffer = ImageBuffer()

    def train_epoch(self, dataloader):

        self.G_H2P.train()
        self.G_P2H.train()

        self.D_H.train()
        self.D_P.train()

        epoch_losses = {
            "G": 0.0,
            "D_H": 0.0,
            "D_P": 0.0,
            "cycle": 0.0,
            "identity": 0.0,
            "gan": 0.0
        }

        progress_bar = tqdm(dataloader)

        for batch in progress_bar:

            healthy = batch["healthy"].to(self.device)
            pneumonia = batch["pneumonia"].to(self.device)

            if self.use_mask:
                # Split image (ch 0) and lung mask (ch 1)
                real_H = healthy[:, 0:1, :, :]
                H_mask = healthy[:, 1:2, :, :]
                real_P = pneumonia[:, 0:1, :, :]
                P_mask = pneumonia[:, 1:2, :, :]
                # Concatenated inputs for generators
                H_input = torch.cat([real_H, H_mask], dim=1)
                P_input = torch.cat([real_P, P_mask], dim=1)
            else:
                real_H = healthy
                real_P = pneumonia

            # ==========================================
            # Train Generators
            # ==========================================

            self.optimizer_G.zero_grad()

            if self.use_mask:
                # Identity loss — generator should preserve images already in target domain
                same_healthy = self.G_P2H(H_input)
                loss_identity_H = self.identity_loss(same_healthy, real_H, H_mask)

                same_pneumonia = self.G_H2P(P_input)
                loss_identity_P = self.identity_loss(same_pneumonia, real_P, P_mask)

                # GAN loss — hard mask: background is taken directly from the real image
                fake_pneumonia = self.G_H2P(H_input)
                fake_pneumonia = fake_pneumonia * H_mask + real_H * (1 - H_mask)

                fake_healthy = self.G_P2H(P_input)
                fake_healthy = fake_healthy * P_mask + real_P * (1 - P_mask)

                # Cycle loss — use source mask for the fake image (same spatial structure)
                recovered_healthy = self.G_P2H(
                    torch.cat([fake_pneumonia, H_mask], dim=1)
                )
                recovered_pneumonia = self.G_H2P(
                    torch.cat([fake_healthy, P_mask], dim=1)
                )

                loss_cycle_H = self.cycle_loss(recovered_healthy, real_H, H_mask)
                loss_cycle_P = self.cycle_loss(recovered_pneumonia, real_P, P_mask)
            else:
                # Identity loss
                same_healthy = self.G_P2H(real_H)
                loss_identity_H = self.identity_loss(same_healthy, real_H)

                same_pneumonia = self.G_H2P(real_P)
                loss_identity_P = self.identity_loss(same_pneumonia, real_P)

                # GAN loss
                fake_pneumonia = self.G_H2P(real_H)
                fake_healthy = self.G_P2H(real_P)

                # Cycle loss
                recovered_healthy = self.G_P2H(fake_pneumonia)
                recovered_pneumonia = self.G_H2P(fake_healthy)

                loss_cycle_H = self.cycle_loss(recovered_healthy, real_H)
                loss_cycle_P = self.cycle_loss(recovered_pneumonia, real_P)

            # Discriminators see only the generated image (1 channel), not the mask
            pred_fake_P = self.D_P(fake_pneumonia)
            pred_fake_H = self.D_H(fake_healthy)

            loss_identity = (
                loss_identity_H +
                loss_identity_P
            ) * self.lambda_identity

            loss_GAN_H2P = self.gan_loss(pred_fake_P, True)
            loss_GAN_P2H = self.gan_loss(pred_fake_H, True)

            loss_cycle = (
                loss_cycle_H +
                loss_cycle_P
            ) * self.lambda_cycle

            # Total generator loss
            loss_G = (
                loss_GAN_H2P +
                loss_GAN_P2H +
                loss_cycle +
                loss_identity
            )

            loss_G.backward()

            self.optimizer_G.step()

            # ==========================================
            # Train Discriminator Healthy
            # ==========================================

            self.optimizer_D_H.zero_grad()

            pred_real_H = self.D_H(real_H)

            loss_real_H = self.gan_loss(
                pred_real_H,
                True
            )

            fake_healthy_buffered = (
                self.fake_healthy_buffer.push_and_pop(
                    fake_healthy
                )
            )

            pred_fake_H = self.D_H(
                fake_healthy_buffered.detach()
            )

            loss_fake_H = self.gan_loss(
                pred_fake_H,
                False
            )

            loss_D_H = (
                loss_real_H +
                loss_fake_H
            ) * 0.5

            loss_D_H.backward()

            self.optimizer_D_H.step()

            # ==========================================
            # Train Discriminator Pneumonia
            # ==========================================

            self.optimizer_D_P.zero_grad()

            pred_real_P = self.D_P(real_P)

            loss_real_P = self.gan_loss(
                pred_real_P,
                True
            )

            fake_pneumonia_buffered = (
                self.fake_pneumonia_buffer.push_and_pop(
                    fake_pneumonia
                )
            )

            pred_fake_P = self.D_P(
                fake_pneumonia_buffered.detach()
            )

            loss_fake_P = self.gan_loss(
                pred_fake_P,
                False
            )

            loss_D_P = (
                loss_real_P +
                loss_fake_P
            ) * 0.5

            loss_D_P.backward()

            self.optimizer_D_P.step()

            # ==========================================
            # Logging
            # ==========================================

            epoch_losses["G"] += loss_G.item()
            epoch_losses["D_H"] += loss_D_H.item()
            epoch_losses["D_P"] += loss_D_P.item()

            epoch_losses["cycle"] += loss_cycle.item()
            epoch_losses["identity"] += loss_identity.item()

            epoch_losses["gan"] += (
                loss_GAN_H2P.item() +
                loss_GAN_P2H.item()
            ) / 2

            progress_bar.set_description(
                f"G: {loss_G.item():.4f} | "
                f"D_H: {loss_D_H.item():.4f} | "
                f"D_P: {loss_D_P.item():.4f}"
            )

        n_batches = len(dataloader)

        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        return {
            "losses": epoch_losses,
            "images": {
                "real_healthy": real_H.detach().cpu(),
                "fake_pneumonia": fake_pneumonia.detach().cpu(),
                "recovered_healthy": recovered_healthy.detach().cpu(),

                "real_pneumonia": real_P.detach().cpu(),
                "fake_healthy": fake_healthy.detach().cpu(),
                "recovered_pneumonia": recovered_pneumonia.detach().cpu(),
            }
        }
