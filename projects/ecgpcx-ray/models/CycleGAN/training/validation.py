import torch


@torch.no_grad()
def validate(trainer, dataloader):

    trainer.G_H2P.eval()
    trainer.G_P2H.eval()

    losses = {
        "cycle": 0.0
    }

    for batch in dataloader:

        healthy = batch["healthy"].to(trainer.device)
        pneumonia = batch["pneumonia"].to(trainer.device)

        if trainer.use_mask:
            real_H = healthy[:, 0:1, :, :]
            H_mask = healthy[:, 1:2, :, :]
            real_P = pneumonia[:, 0:1, :, :]
            P_mask = pneumonia[:, 1:2, :, :]

            fake_P = trainer.G_H2P(torch.cat([real_H, H_mask], dim=1))
            fake_P = fake_P * H_mask + real_H * (1 - H_mask)
            rec_H = trainer.G_P2H(torch.cat([fake_P, H_mask], dim=1))

            fake_H = trainer.G_P2H(torch.cat([real_P, P_mask], dim=1))
            fake_H = fake_H * P_mask + real_P * (1 - P_mask)
            rec_P = trainer.G_H2P(torch.cat([fake_H, P_mask], dim=1))

            loss_cycle_H = trainer.cycle_loss(rec_H, real_H, H_mask)
            loss_cycle_P = trainer.cycle_loss(rec_P, real_P, P_mask)
        else:
            real_H = healthy
            real_P = pneumonia

            fake_P = trainer.G_H2P(real_H)
            rec_H = trainer.G_P2H(fake_P)

            fake_H = trainer.G_P2H(real_P)
            rec_P = trainer.G_H2P(fake_H)

            loss_cycle_H = trainer.cycle_loss(rec_H, real_H)
            loss_cycle_P = trainer.cycle_loss(rec_P, real_P)

        losses["cycle"] += (loss_cycle_H + loss_cycle_P).item()

    losses["cycle"] /= len(dataloader)

    return losses
