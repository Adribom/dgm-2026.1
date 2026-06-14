from .generators import Generator
from .discriminators import Discriminator


class CycleGAN:
    """
    Wrapper class for CycleGAN architecture.

    When use_mask=True the generators receive a 2-channel input (image + lung
    mask) and produce a 1-channel image output.  Discriminators always see
    only the image channel, keeping the discriminator architecture unchanged.
    """

    def __init__(self, image_channels=1, use_mask=False):

        # With mask: generator input = image + mask (2 channels), output = image (1 channel)
        gen_in = image_channels + 1 if use_mask else image_channels

        # Generators
        self.G_H2P = Generator(
            input_channels=gen_in,
            output_channels=image_channels
        )

        self.G_P2H = Generator(
            input_channels=gen_in,
            output_channels=image_channels
        )

        # Discriminators — always receive only the image channel
        self.D_H = Discriminator(
            input_channels=image_channels
        )

        self.D_P = Discriminator(
            input_channels=image_channels
        )