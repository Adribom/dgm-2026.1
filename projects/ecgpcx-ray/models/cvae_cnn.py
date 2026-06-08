import numpy as np
import torch
import torch.nn as nn


# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed at module level
set_seed(42)

class CVAE(nn.Module):
    def __init__(
        self,
        img_channels=1,
        output_channels=1,
        img_size=128,
        latent_dim=64,
        num_classes=2,
        metadata_dim=2,
        gender_embedding_dim=4,
        base_channels=32,
        dropout=0.05,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.metadata_dim = metadata_dim
        self.gender_embedding_dim = gender_embedding_dim
        self.img_channels = img_channels
        self.img_size = img_size
        self.output_channels = output_channels
        self.feature_size = img_size // 16
        self.encoder_channels = base_channels * 8

        self.cond_dim = num_classes + (metadata_dim - 1) + gender_embedding_dim

        self.encoder_blocks = nn.ModuleList([
            self._encoder_block(img_channels, base_channels, dropout=0.0),
            self._encoder_block(base_channels, base_channels * 2, dropout=dropout),
            self._encoder_block(base_channels * 2, base_channels * 4, dropout=dropout),
            self._encoder_block(base_channels * 4, self.encoder_channels, dropout=dropout),
        ])
        self.flatten = nn.Flatten()

        encoder_dim = self.encoder_channels * self.feature_size * self.feature_size
        self.fc21 = nn.Linear(encoder_dim + self.cond_dim, latent_dim)
        self.fc22 = nn.Linear(encoder_dim + self.cond_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim + self.cond_dim, encoder_dim)

        decoder_channels = [base_channels * 4, base_channels * 2, base_channels]
        self.decoder_blocks = nn.ModuleList([
            self._decoder_block(self.encoder_channels, decoder_channels[0], dropout=dropout),
            self._decoder_block(decoder_channels[0] + base_channels * 4, decoder_channels[1], dropout=dropout),
            self._decoder_block(decoder_channels[1] + base_channels * 2, decoder_channels[2], dropout=dropout),
        ])
        self.film_layers = nn.ModuleList([
            nn.Linear(self.cond_dim, channels * 2) for channels in decoder_channels
        ])
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(base_channels + base_channels, self.output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.gender_embedding = nn.Embedding(2, gender_embedding_dim)

    @staticmethod
    def _norm(channels):
        return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)

    @classmethod
    def _encoder_block(cls, in_channels, out_channels, dropout=0.0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            cls._norm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    @classmethod
    def _decoder_block(cls, in_channels, out_channels, dropout=0.0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            cls._norm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    @staticmethod
    def _apply_film(x, cond, film_layer):
        gamma, beta = film_layer(cond).chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + gamma) + beta

    def _prepare_metadata(self, m):
        if m is None:
            return None
        age = m[:, [0]]
        gender_idx = m[:, 1].long().clamp(0, self.gender_embedding.num_embeddings - 1)
        gender_emb = self.gender_embedding(gender_idx)
        return torch.cat([age, gender_emb], dim=1)

    def _prepare_condition(self, y, m=None):
        cond = [y]
        if m is not None:
            cond.append(self._prepare_metadata(m))
        else:
            missing_dim = self.cond_dim - self.num_classes
            cond.append(y.new_zeros(y.size(0), missing_dim))
        return torch.cat(cond, dim=1)

    @staticmethod
    def _concat_skip(x, skip, skip_scale=1.0):
        if skip is None:
            skip = torch.zeros_like(x)
        else:
            skip = skip * skip_scale
        return torch.cat([x, skip], dim=1)

    def encode(self, x, y, m=None, return_skips=False):
        x = x.view(x.size(0), self.img_channels, self.img_size, self.img_size)
        skips = []
        h = x
        for block in self.encoder_blocks:
            h = block(h)
            skips.append(h)

        h1 = self.flatten(h)
        cond = self._prepare_condition(y, m)
        inputs = torch.cat([h1, cond], dim=1)
        mu = self.fc21(inputs)
        logvar = self.fc22(inputs)
        if return_skips:
            return mu, logvar, skips[:-1]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y, m=None, skips=None, skip_scale=1.0):
        cond = self._prepare_condition(y, m)
        inputs = torch.cat([z, cond], dim=1)

        h = self.decoder_input(inputs)
        h = h.view(h.size(0), self.encoder_channels, self.feature_size, self.feature_size)
        skip1 = skip2 = skip3 = None
        if skips is not None:
            skip1, skip2, skip3 = skips

        h = self.decoder_blocks[0](h)
        h = self._apply_film(h, cond, self.film_layers[0])
        h = self._concat_skip(h, skip3, skip_scale)

        h = self.decoder_blocks[1](h)
        h = self._apply_film(h, cond, self.film_layers[1])
        h = self._concat_skip(h, skip2, skip_scale)

        h = self.decoder_blocks[2](h)
        h = self._apply_film(h, cond, self.film_layers[2])
        h = self._concat_skip(h, skip1, skip_scale)
        return self.output_layer(h)

    def forward(self, x, y, m=None):
        mu, logvar, skips = self.encode(x, y, m, return_skips=True)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y, m, skips=skips)
        return x_hat, mu, logvar
