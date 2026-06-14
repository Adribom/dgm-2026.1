"""
Point-E Inference Core Class.

This module encapsulates the state and logic required for predicting
3D point clouds from 2D images using OpenAI's Point-E, with support
for loading custom fine-tuned weights.
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image

# Point-E imports
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config

class PotholePointE:
    def __init__(
        self,
        device: torch.device | None = None,
        base_model_name: str = "base40M", 
        upsample_model_name: str = "upsample",
        custom_base_weights_path: str | None = None,
        num_points_base: int = 1024,
        num_points_upsample: int = 4096,
        guidance_scale_base: float = 3.0,
        guidance_scale_upsample: float = 3.0,
        use_upsampler: bool = True,
    ):
        """Initializes the Point-E models (Base + Upsampler) supporting custom weights."""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.use_upsampler = use_upsampler
            
        print(f'Creating base model ({base_model_name})...')
        self.base_model = model_from_config(MODEL_CONFIGS[base_model_name], self.device)
        self.base_model.eval()
        self.base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_model_name])

        # Define explicit cache directory parallel to the src/ folder to avoid os.getcwd() side-effects in Notebooks
        cache_dir = Path(__file__).resolve().parent.parent.parent / "point_e_model_cache"
        cache_dir_str = str(cache_dir)

        if custom_base_weights_path:
            print(f'Loading CUSTOM local base weights from: {custom_base_weights_path}')
            weights = torch.load(custom_base_weights_path, map_location=self.device)
            if isinstance(weights, dict) and "model_state_dict" in weights:
                weights = weights["model_state_dict"]
            self.base_model.load_state_dict(weights)
        else:
            print(f'Downloading/Loading baseline checkpoint from custom cache: {cache_dir_str}')
            self.base_model.load_state_dict(load_checkpoint(base_model_name, self.device, cache_dir=cache_dir_str))

        if self.use_upsampler:
            print(f'Creating upsample model ({upsample_model_name})...')
            self.upsampler_model = model_from_config(MODEL_CONFIGS[upsample_model_name], self.device)
            self.upsampler_model.eval()
            self.upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsample_model_name])

            print('Downloading/Loading upsampler checkpoint from custom cache...')
            self.upsampler_model.load_state_dict(load_checkpoint(upsample_model_name, self.device, cache_dir=cache_dir_str))

            self.sampler = PointCloudSampler(
                device=self.device,
                models=[self.base_model, self.upsampler_model],
                diffusions=[self.base_diffusion, self.upsampler_diffusion],
                num_points=[num_points_base, num_points_upsample - num_points_base],
                aux_channels=['R', 'G', 'B'],
                guidance_scale=[guidance_scale_base, guidance_scale_upsample],
            )
        else:
            self.upsampler_model = None
            self.upsampler_diffusion = None
            self.sampler = PointCloudSampler(
                device=self.device,
                models=[self.base_model],
                diffusions=[self.base_diffusion],
                num_points=[num_points_base],
                aux_channels=['R', 'G', 'B'],
                model_kwargs_key_filter=('*',),
                guidance_scale=[guidance_scale_base],
                use_karras=[True],
                karras_steps=[64],
                sigma_min=[1e-3],
                sigma_max=[120],
                s_churn=[3],
            )

    def predict(self, image: Image.Image, scale: float = 1.0) -> np.ndarray:
        """
        Runs Point-E inference on a single image and returns the un-normalized 3D coordinates.
        Expects a PIL Image.
        """
        samples = None
        for x in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[image])):
            samples = x
        
        # Extract point cloud
        pc = self.sampler.output_to_point_clouds(samples)[0]
        coords = np.array(pc.coords) # Point-E coords: Shape (N, 3)
        
        # De-normalize ALL points back to real-world proportions 
        # The metadata scale is inherently in millimeters. We multiply all axes to fix the aspect ratio 
        # and divide by 1000.0 to convert the whole cloud to meters.
        coords = (coords * scale) / 1000.0
        
        return coords
