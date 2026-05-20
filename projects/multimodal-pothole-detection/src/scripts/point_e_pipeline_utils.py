"""
Point-E Pipeline Utilities
Provides pure functions to handle 2D square cropping and geometry 
adaptations necessary for the Generative 3D model inputs.
"""

import numpy as np
import cv2

def get_square_bbox_from_mask(mask: np.ndarray, margin_px: int = 50) -> tuple:
    """
    Finds a square bounding box around a given mask, including a margin.
    If the bounding box falls outside the image, the coordinates can be negative
    or exceed image dimensions. These will be handled by padding later.

    Args:
        mask (np.ndarray): 2D binary or grayscale mask (H, W).
        margin_px (int): Number of context pixels to add around the largest dimension.

    Returns:
        tuple: (y_min, y_max, x_min, x_max) absolute coordinates. 
               Note: These can be out of image bounds, requiring padding.
    """
    # Find all non-zero pixels
    coords = cv2.findNonZero((mask > 0).astype(np.uint8))
    
    if coords is None:
        return 0, 0, 0, 0

    x_min, y_min, w, h = cv2.boundingRect(coords)
    x_max = x_min + w
    y_max = y_min + h

    # Center of the bounding box
    c_x = (x_min + x_max) // 2
    c_y = (y_min + y_max) // 2

    # Determine the largest dimension to make it square
    max_side = max(w, h)
    
    # Half side of the new square, including the context margin
    half_side = (max_side // 2) + margin_px

    # Calculate new square bounds (can be outside the image)
    new_x_min = c_x - half_side
    new_x_max = c_x + half_side
    new_y_min = c_y - half_side
    new_y_max = c_y + half_side

    return new_y_min, new_y_max, new_x_min, new_x_max


def apply_square_crop(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Crops an image using a square bounding box and applies zero-padding (black edges)
    if the bounding box exceeds the original image dimensions.

    Args:
        image (np.ndarray): The 2D or 3D image array (H, W, C) or (H, W).
        bbox (tuple): (y_min, y_max, x_min, x_max) from `get_square_bbox_from_mask`.

    Returns:
        np.ndarray: The perfectly square cropped (and potentially padded) image.
    """
    img_h, img_w = image.shape[:2]
    y_min, y_max, x_min, x_max = bbox

    # Calculate the size of the target square
    target_h = y_max - y_min
    target_w = x_max - x_min

    # Create an empty template (black/zeros) for the crop
    if len(image.shape) == 3:
        channels = image.shape[2]
        crop = np.zeros((target_h, target_w, channels), dtype=image.dtype)
    else:
        crop = np.zeros((target_h, target_w), dtype=image.dtype)

    # Determine overlapping regions between the bounding box and the real image
    # Source image coordinates
    src_y_min = max(0, y_min)
    src_y_max = min(img_h, y_max)
    src_x_min = max(0, x_min)
    src_x_max = min(img_w, x_max)

    # Destination crop coordinates (where the src goes into the zero-padded template)
    dst_y_min = src_y_min - y_min
    dst_y_max = src_y_max - y_min
    dst_x_min = src_x_min - x_min
    dst_x_max = src_x_max - x_min

    # Copy the valid pixels from the image into the padded template
    if src_y_max > src_y_min and src_x_max > src_x_min:
        crop[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = image[src_y_min:src_y_max, src_x_min:src_x_max]

    return crop

def compute_leveled_point_cloud(
    rgb_img: np.ndarray, 
    depth_img: np.ndarray, 
    mask_img: np.ndarray, 
    square_bbox: tuple,
    fx: float = 460.0, 
    fy: float = 460.0, 
    cx: float = 320.0, 
    cy: float = 240.0
) -> tuple:
    """
    Computes the leveled 3D point cloud for the target square crop region using RANSAC on the full asphalt.

    Args:
        rgb_img, depth_img, mask_img (np.ndarray): Original full (unpadded) image arrays.
        square_bbox (tuple): Target coordinates (y_min, y_max, x_min, x_max).
        fx, fy, cx, cy (float): Camera intrinsics.
    
    Returns:
        points_3d (np.ndarray): N x 3 geometric points perfectly leveled.
        colors (np.ndarray): N x 3 RGB values [0 - 255].
    """
    import open3d as o3d
    
    # 1. RANSAC on the entire road surface for a robust fit
    h_img, w_img = depth_img.shape
    u_grid, v_grid = np.meshgrid(np.arange(w_img), np.arange(h_img))
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (150, 150))
    expanded_mask = cv2.dilate((mask_img > 0).astype(np.uint8), kernel_large, iterations=1)
    road_only_mask = (expanded_mask > 0) & (mask_img == 0) & (depth_img > 0)
    
    road_u = u_grid[road_only_mask]
    road_v = v_grid[road_only_mask]
    road_z = depth_img[road_only_mask].astype(np.float32)
    
    # Check if road valid (unlikely to fail on sensible datasets)
    if len(road_z) < 10:
        return np.array([]), np.array([])
        
    road_X = (road_u - cx) * road_z / fx
    road_Y = (road_v - cy) * road_z / fy
    road_points_3d = np.stack((road_X, road_Y, road_z), axis=-1)
    
    pcd_road = o3d.geometry.PointCloud()
    pcd_road.points = o3d.utility.Vector3dVector(road_points_3d)
    plane_model, inliers = pcd_road.segment_plane(distance_threshold=10.0, ransac_n=3, num_iterations=1000)
    A, B, C, D = plane_model

    # 2. Extract points ONLY within the unpadded valid region of our Square Crop
    y_min, y_max, x_min, x_max = square_bbox
    src_y_min, src_y_max = max(0, y_min), min(h_img, y_max)
    src_x_min, src_x_max = max(0, x_min), min(w_img, x_max)
    
    crop_mask = np.zeros_like(depth_img, dtype=bool)
    if src_y_max > src_y_min and src_x_max > src_x_min:
        crop_mask[src_y_min:src_y_max, src_x_min:src_x_max] = True
        
    # Points inside the crop that also have valid sensor data (depth > 0)
    valid_points_mask = crop_mask & (depth_img > 0)
    
    u_valid = u_grid[valid_points_mask]
    v_valid = v_grid[valid_points_mask]
    z_raw = depth_img[valid_points_mask].astype(np.float32)
    valid_colors = rgb_img[valid_points_mask]
    
    # 3. Unproject Valid Crop Points
    X = (u_valid - cx) * z_raw / fx
    Y = (v_valid - cy) * z_raw / fy
    
    # 4. Math subtraction to level everything against the found Asphalt Plane
    if abs(C) > 1e-6:
        z_expected = -(A * X + B * Y + D) / C
    else:
        z_expected = np.zeros_like(z_raw)
        
    Z_leveled = z_raw - z_expected
    points_3d = np.stack((X, Y, Z_leveled), axis=-1)
    
    # 5. Clean flying noisy pixels using Statistical Outlier Removal
    if len(points_3d) > 20: # ensure enough points exist to filter
        pcd_crop = o3d.geometry.PointCloud()
        pcd_crop.points = o3d.utility.Vector3dVector(points_3d)
        pcd_crop.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)
        pcd_clean, _ = pcd_crop.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return np.asarray(pcd_clean.points), (np.asarray(pcd_clean.colors) * 255.0).astype(np.uint8)
        
    return points_3d, valid_colors

def format_point_e_tensor(points_3d: np.ndarray, colors_rgb: np.ndarray, num_points: int = 1024) -> tuple:
    """
    Formats the raw leveled point cloud into a strict (1024, 6) tensor in the [-1, 1] range.
    Employs Farthest Point Sampling (Open3D) if N > 1024.

    Args:
        points_3d (np.ndarray): N x 3 float array
        colors_rgb (np.ndarray): N x 3 int/float array [0 - 255]
        num_points (int): The target K points to fit Point-E requirements.
        
    Returns:
        np.ndarray: The finalized geometry-color tensor of shape (K, 6) 
        float: The global scale factor used for uniform spatial normalization.
    """
    import open3d as o3d
    
    n_points = points_3d.shape[0]
    
    if n_points == 0:
        return np.zeros((num_points, 6), dtype=np.float32), 1.0

    # 1. Stric Sampling
    if n_points > num_points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)
        
        # Geometrically downsample using FPS strategy
        pcd_sampled = pcd.farthest_point_down_sample(num_samples=num_points)
        sampled_points = np.asarray(pcd_sampled.points)
        sampled_colors = np.asarray(pcd_sampled.colors) * 255.0
    elif n_points < num_points:
        # Uniform Random Upsampling with replacement to fill in gaps
        idx = np.random.choice(n_points, num_points, replace=True)
        sampled_points = points_3d[idx]
        sampled_colors = colors_rgb[idx]
    else:
        sampled_points = points_3d
        sampled_colors = colors_rgb

    # 2. Geometry Center
    # We recenter X and Y so the model sees the object centered, 
    # but DO NOT center Z (Z=0 is the exact leveled boundary of the physical street surface)
    c_x = np.mean(sampled_points[:, 0])
    c_y = np.mean(sampled_points[:, 1])
    sampled_points[:, 0] -= c_x
    sampled_points[:, 1] -= c_y

    # 3. Uniform Global Scaling [-1, 1] 
    # We find the single max coordinate over any axis to preserve depth width/height proportions
    scale_factor = float(np.max(np.abs(sampled_points)))
    if scale_factor < 1e-6:
        scale_factor = 1.0
        
    points_norm = sampled_points / scale_factor

    # 4. Color Scaling [-1, 1]
    # Input is 0-255 -> convert to range around 0
    colors_norm = (sampled_colors / 127.5) - 1.0

    # Create (1024, 6) tensor
    tensor_6d = np.concatenate((points_norm, colors_norm), axis=-1).astype(np.float32)

    return tensor_6d, scale_factor
