import random

import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


def get_train_transforms(image_size=128):

    return transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=5
        ),

        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


def get_val_transforms(image_size=128):

    return transforms.Compose([

        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5],
            std=[0.5]
        )
    ])


class MaskAwareTransform:
    """
    Applies synchronized spatial augmentations to an (image, mask) PIL pair,
    returning a (2, H, W) tensor: channel 0 = image normalized to [-1, 1],
    channel 1 = binary lung mask in {0, 1}.

    The same random spatial parameters are applied to both PIL images so the
    mask stays aligned with the X-ray after augmentation.
    """

    def __init__(self, image_size=128, augment=True):
        self.image_size = image_size
        self.augment = augment

    def __call__(self, image_pil, mask_pil):
        # Resize — NEAREST for mask to keep binary values crisp
        image_pil = TF.resize(image_pil, (self.image_size, self.image_size))
        mask_pil = TF.resize(
            mask_pil,
            (self.image_size, self.image_size),
            interpolation=InterpolationMode.NEAREST,
        )

        if self.augment:
            # Horizontal flip
            if random.random() > 0.5:
                image_pil = TF.hflip(image_pil)
                mask_pil = TF.hflip(mask_pil)

            # Rotation — same angle for both
            angle = random.uniform(-5.0, 5.0)
            image_pil = TF.rotate(image_pil, angle)
            mask_pil = TF.rotate(mask_pil, angle, interpolation=InterpolationMode.NEAREST)

            # Affine: translation + scale (mirrors RandomAffine(translate=(0.02,0.02), scale=(0.98,1.02)))
            dx = random.uniform(-0.02, 0.02) * self.image_size
            dy = random.uniform(-0.02, 0.02) * self.image_size
            scale = random.uniform(0.98, 1.02)
            image_pil = TF.affine(image_pil, angle=0, translate=(dx, dy), scale=scale, shear=0)
            mask_pil = TF.affine(
                mask_pil, angle=0, translate=(dx, dy), scale=scale, shear=0,
                interpolation=InterpolationMode.NEAREST,
            )

        image_tensor = TF.to_tensor(image_pil)                        # (1, H, W) in [0, 1]
        image_tensor = TF.normalize(image_tensor, [0.5], [0.5])       # [-1, 1]

        mask_tensor = TF.to_tensor(mask_pil)                          # (1, H, W) in [0, 1]
        mask_tensor = (mask_tensor > 0.5).float()                     # binarize

        return torch.cat([image_tensor, mask_tensor], dim=0)          # (2, H, W)


def get_train_mask_transforms(image_size=128):
    return MaskAwareTransform(image_size=image_size, augment=True)


def get_val_mask_transforms(image_size=128):
    return MaskAwareTransform(image_size=image_size, augment=False)