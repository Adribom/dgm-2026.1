"""Preprocessing module for NIH Chest X-rays dataset.

This module provides functionality to download, load, clean, and preprocess
the NIH Chest X-rays dataset from Kaggle. It handles data cleaning (duplicate
removal, outlier detection), patient filtering, and image loading and resizing.
"""

import os
import kagglehub
import pandas as pd
import numpy as np
from PIL import Image
from xrv_lung_segmentation import (
    TorchXRayVisionLungSegmenter,
    apply_lung_segmentation,
    create_lung_mask_images,
)
import torch
from torch.utils.data import DataLoader
from dataset import PyTorchDataset

class Preprocessing():
    """Handle preprocessing of NIH Chest X-rays dataset.
    
    This class manages the download, loading, and preprocessing of chest X-ray
    images and associated metadata. It filters patients by condition labels,
    removes outliers and duplicates, and loads and resizes images.
    """
    def __init__(self, label):
        """Initialize Preprocessing instance and download dataset.
        
        Args:
            label (str): Finding label to filter for disease cases (e.g., 'Pneumonia').
        """
        self.label = label
        self.pneumonia = None  # Will store filtered pneumonia patient dataframe
        self.healthy = None  # Will store filtered healthy patient dataframe
        self.download_path = kagglehub.dataset_download("nih-chest-xrays/data")  # Download dataset from Kaggle
        self.metadata = None  # Will store the CSV metadata

    def _load_dataframe(self):
        """Load metadata CSV file from downloaded dataset.
        
        Reads the Data_Entry_2017.csv file containing patient metadata including
        finding labels and demographics. Removes the unnamed column and stores
        the result in self.metadata.
        """
        file_path_in_dataset = "Data_Entry_2017.csv"
        full_csv_path = os.path.join(self.download_path, file_path_in_dataset)
        
        # Load CSV with pandas
        df = pd.read_csv(full_csv_path)
        df.drop('Unnamed: 11', axis=1, inplace=True)  # Remove unnecessary column
        self.metadata = df.copy()
    
    def _outlier_removal(self, df, verbose=True):
        """Remove age outliers using Interquartile Range (IQR) method.
        
        Removes records where Patient Age falls outside 1.5 * IQR bounds
        (values below Q1 - 1.5*IQR or above Q3 + 1.5*IQR).
        
        Args:
            df (pd.DataFrame): Input dataframe to remove outliers from.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        # Calculate quartiles and IQR for age
        q1 = df['Patient Age'].quantile(0.25)
        q3 = df['Patient Age'].quantile(0.75)
        iqr = q3 - q1
        
        # Determine outlier bounds using IQR method
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Filter records within acceptable age range
        initial_rows = df.shape[0]
        df = df[(df['Patient Age'] >= lower_bound) & (df['Patient Age'] <= upper_bound)]
        
        # Report results
        if verbose:
            removed_rows = initial_rows - df.shape[0]
            print(f"Removed {removed_rows} outliers based on Patient Age.")
            print(f"New shape of dataframe: {df.shape}")

        return df

    def _remove_duplicates(self, df, verbose=True):
        """Remove duplicate records from dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe to remove duplicates from.
            
        Returns:
            pd.DataFrame: Dataframe with duplicates removed.
        """
        duplicates = df.duplicated().sum()
        print(f"Number of duplicated rows: {duplicates}")
        
        # Remove duplicates if any exist
        if duplicates > 0:
            df = df.drop_duplicates()
            if verbose:
                print(f"Duplicated rows removed. New dataframe length: {df.shape[0]}")
        return df
    
    def _normalize_age(self, df, method='minmax', verbose=True):
        """Normalize patient age values using specified method.
        
        Applies age normalization to create features suitable for machine learning
        models. Supports Min-Max scaling (0-1 range) and standardization (z-score).
        
        Args:
            df (pd.DataFrame): Input dataframe containing 'Patient Age' column.
            method (str): Normalization method. Options are:
                         'minmax': Min-Max scaling to [0, 1] range (default).
                         'standard': Standardization (z-score normalization).
                         
        Returns:
            pd.DataFrame: Dataframe with new 'Patient Age Normalized' column.
        """
        df_normalized = df.copy()
        
        if method == 'minmax':
            # Min-Max scaling: (x - min) / (max - min)
            min_age = df['Patient Age'].min()
            max_age = df['Patient Age'].max()
            df_normalized['Patient Age Normalized'] = (df['Patient Age'] - min_age) / (max_age - min_age)
            if verbose:
                print(f"Age normalized using Min-Max scaling. Range: [{min_age}, {max_age}] -> [0, 1]")
            
        elif method == 'standard':
            # Standardization (z-score): (x - mean) / std
            mean_age = df['Patient Age'].mean()
            std_age = df['Patient Age'].std()
            df_normalized['Patient Age Normalized'] = (df['Patient Age'] - mean_age) / std_age
            if verbose:
                print(f"Age normalized using standardization. Mean: {mean_age:.2f}, Std: {std_age:.2f}")
            
        else:
            raise ValueError(f"Unknown normalization method: {method}. Choose 'minmax' or 'standard'.")
        
        return df_normalized

    def _encode_gender(self, df, encoding_map=None):
        """Encode categorical gender values to numeric format.
        
        Converts gender values ('M', 'F') to numeric values suitable for
        machine learning models. Default: M=1, F=0.
        
        Args:
            df (pd.DataFrame): Input dataframe containing 'Patient Gender' column.
            encoding_map (dict, optional): Custom mapping for gender values.
                                          Format: {'M': 1, 'F': 0}.
                                          If None, uses default mapping.
                                          
        Returns:
            pd.DataFrame: Dataframe with new 'Patient Gender Encoded' column.
        """
        df_encoded = df.copy()
        
        # Use default mapping if not provided
        if encoding_map is None:
            encoding_map = {'M': 1, 'F': 0}
        
        # Apply encoding
        df_encoded['Patient Gender Encoded'] = df['Patient Gender'].map(encoding_map)
        
        return df_encoded

    def _one_hot_encode_label_column(self, df, label_col='Label', num_classes=2, prefix='Label'):
        """Add one-hot encoded label columns to the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe containing the label column.
            label_col (str): Name of the label column to one-hot encode.
            num_classes (int): Number of classes to encode.
            prefix (str): Prefix for one-hot column names.
            
        Returns:
            pd.DataFrame: Copy of dataframe with new one-hot columns added.
        """
        df_encoded = df.copy()
        labels = df_encoded[label_col].astype(int).values
        
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Labels must be in [0, {num_classes - 1}], got {labels.min()} to {labels.max()}")
        
        one_hot = np.eye(num_classes)[labels]
        for i in range(num_classes):
            df_encoded[f"{prefix}_{i}"] = one_hot[:, i].astype(int)
        
        return df_encoded

    def _one_hot_encode_labels(self, labels, num_classes=2):
        """Convert label values to one-hot encoded torch tensors.
        
        Args:
            labels (array-like): Integer label values.
            num_classes (int): Number of classes.
            
        Returns:
            torch.Tensor: One-hot encoded labels of shape (N, num_classes).
        """
        labels_arr = np.asarray(labels).astype(int)
        if labels_arr.min() < 0 or labels_arr.max() >= num_classes:
            raise ValueError(f"Labels must be in [0, {num_classes - 1}], got {labels_arr.min()} to {labels_arr.max()}")
        return torch.from_numpy(np.eye(num_classes)[labels_arr]).float()
    
    def _get_pneumonia_patients(self, verbose=True, method='minmax'):
        """Filter, clean, and store pneumonia patient records.
        
        Filters metadata for records matching the specified finding label,
        removes duplicates, and removes age outliers.
        """
        # Filter patients with the specified condition
        # pneumonia_only_df = self.metadata[self.metadata['Finding Labels'] == self.label]
        pneumonia_only_df = self.metadata[self.metadata['Finding Labels'].str.contains('Pneumonia')]


        if verbose:
            print('Pneumonia patient records:')
        
        # Apply data cleaning steps
        pneumonia_only_df = self._remove_duplicates(pneumonia_only_df, verbose)
        pneumonia_only_df = self._outlier_removal(pneumonia_only_df, verbose)

        #Normalize Age
        pneumonia_only_df = self._normalize_age(pneumonia_only_df, method=method, verbose=verbose)
        #Encode Gender
        pneumonia_only_df = self._encode_gender(pneumonia_only_df, encoding_map={'M': 1, 'F': 0})

        pneumonia_only_df['Label'] = 1  # Add label column for pneumonia (1)
        pneumonia_only_df = pneumonia_only_df.reset_index(drop=True)  # Reset index after filtering and cleaning
        
        # Store cleaned dataframe
        self.pneumonia = pneumonia_only_df.copy()

    def _get_healthy_patients(self, verbose=True, method='minmax'):
        """Filter, clean, and store healthy patient records.
        
        Filters metadata for records with 'No Finding' label (healthy controls),
        removes duplicates, and removes age outliers.
        """
        # Filter patients with no medical findings (healthy controls)
        healthy_df = self.metadata[self.metadata['Finding Labels'] == 'No Finding']
        
        if verbose:
            print('Healthy patient records:')

        # Apply data cleaning steps
        healthy_df = self._remove_duplicates(healthy_df, verbose)
        healthy_df = self._outlier_removal(healthy_df, verbose)

        # Normalize Age
        healthy_df = self._normalize_age(healthy_df, method=method, verbose=verbose)
        # Encode Gender
        healthy_df = self._encode_gender(healthy_df, encoding_map={'M': 1, 'F': 0})

        healthy_df['Label'] = 0  # Add label column for healthy (0)
        healthy_df = healthy_df.reset_index(drop=True)  # Reset index after filtering and cleaning

        # Store cleaned dataframe
        self.healthy = healthy_df.copy()

    def _load_image(self, image_filename, base_path, size=None):
        """Load and optionally resize a single chest X-ray image.
        
        Images are organized in subdirectories (images_001 through images_012).
        This method searches through all subdirectories to locate the image.
        
        Args:
            image_filename (str): Name of the image file to load.
            base_path (str): Base path to the images directory.
            size (tuple, optional): Target size (width, height) for resizing.
                                   If None, image is not resized. Defaults to None.
                                   
        Returns:
            PIL.Image.Image: Loaded and optionally resized image, or None if not found.
        """
        # Search through image subdirectories (images_001 to images_012)
        for i in range(1, 13):
            folder_name = f'images_{i:03d}'
            image_path = os.path.join(base_path, folder_name, 'images', image_filename)
            
            # Load image if found
            if os.path.exists(image_path):
                img = Image.open(image_path)
                
                # Resize if size parameter provided
                if size:
                    img = img.resize(size)
                return img
        
        # Return None if image not found in any directory
        return None

    def _load_all_images(self, size, df, verbose=True):
        """Load all images for records in a given dataframe.
        
        Args:
            size (tuple): Target size (width, height) for resizing images.
            df (pd.DataFrame): Dataframe containing 'Image Index' column with image filenames.
            verbose (bool): Whether to print loading progress. Default: True.
            
        Returns:
            list: List of loaded PIL Image objects.
        """
        all_images = []
        
        # Load each image from the dataframe
        for img_idx in df['Image Index']:
            img = self._load_image(img_idx, self.download_path, size=size)
            if img:  # Only add successfully loaded images
                all_images.append(img)
        
        if verbose:
            print(f"Loaded {len(all_images)} images.")

        return all_images
    
    def _load_images(
        self,
        size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        verbose=True,
        method='minmax',
        apply_segmentation=False,
        add_mask_channel=False,
        segmentation_device=None,
        segmentation_threshold=0.5,
        segmentation_mask_value=0,
    ):
        """Main pipeline to load and preprocess all images and metadata.
        
        Orchestrates the complete preprocessing workflow:
        1. Loads metadata CSV
        2. Filters and cleans pneumonia patient data
        3. Filters and cleans healthy patient data
        4. Loads all images for both groups
        
        Args:
            size (tuple): Target size (width, height) for resizing all images.
            train_ratio (float): Fraction of data to use for training. Default: 0.7.
            val_ratio (float): Fraction of data to use for validation. Default: 0.15.
            test_ratio (float): Fraction of data to use for testing. Default: 0.15.
            seed (int): Random seed for reproducibility. Default: 42.
            verbose (bool): Whether to print loading progress. Default: True.
            method (str): Normalization method for age. Options: 'minmax' or 'standard'. Default: 'minmax'.
            apply_segmentation (bool): Whether to apply lung segmentation to images. Default: False.
            segmentation_device (str or torch.device): Device for lung segmentation model. Default: None (auto-detect).
            segmentation_threshold (float): Threshold for lung mask binarization. Default: 0.5.
            segmentation_mask_value (int): Pixel value to use for masked-out areas. Default: 0.

        Returns:
            tuple: Two lists (all_pneumonia_images, all_healthy_images) containing
                   PIL Image objects.
        """
        split_results = self.split_by_patient(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,  # Fixed seed for reproducibility,
            method=method
        )

        #Load images for train, test and validation sets
        train_df = split_results['train_df']
        val_df = split_results['val_df']
        test_df = split_results['test_df']

        train_images = self._load_all_images(size, train_df, verbose=verbose)
        val_images = self._load_all_images(size, val_df, verbose=verbose)
        test_images = self._load_all_images(size, test_df, verbose=verbose)

        train_masks = None
        val_masks = None
        test_masks = None

        if apply_segmentation or add_mask_channel:
            if verbose:
                print("Running TorchXRayVision lung segmentation...")
            segmenter = TorchXRayVisionLungSegmenter(
                device=segmentation_device,
                threshold=segmentation_threshold,
                mask_value=segmentation_mask_value,
            )

        if add_mask_channel:
            train_masks = create_lung_mask_images(
                train_images,
                verbose=verbose,
                segmenter=segmenter,
            )
            val_masks = create_lung_mask_images(
                val_images,
                verbose=verbose,
                segmenter=segmenter,
            )
            test_masks = create_lung_mask_images(
                test_images,
                verbose=verbose,
                segmenter=segmenter,
            )

        if apply_segmentation:
            train_images = apply_lung_segmentation(
                train_images,
                verbose=verbose,
                segmenter=segmenter,
            )
            val_images = apply_lung_segmentation(
                val_images,
                verbose=verbose,
                segmenter=segmenter,
            )
            test_images = apply_lung_segmentation(
                test_images,
                verbose=verbose,
                segmenter=segmenter,
            )

        return train_images, val_images, test_images, split_results, train_masks, val_masks, test_masks

    def create_cvae_dataset(
        self,
        img_size=(128, 128),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        verbose=True,
        method='minmax',
        apply_lung_segmentation=False,
        add_lung_mask_channel=False,
        segmentation_device=None,
        segmentation_threshold=0.5,
        segmentation_mask_value=0,
    ):
        """Create a PyTorch Dataset suitable for CVAE model training with DataLoader.
        
        Combines pneumonia and healthy data, normalizes age, encodes gender,
        and returns a PyTorch Dataset with images, labels, and metadata.
        
        Args:
            img_size (tuple): Image size (height, width) for resizing. Default: (128, 128).
            train_ratio (float): Fraction of data to use for training. Default: 0.7.
            val_ratio (float): Fraction of data to use for validation. Default: 0.15.
            test_ratio (float): Fraction of data to use for testing. Default: 0.15.
            seed (int): Random seed for reproducibility. Default: 42.
            verbose (bool): Whether to print loading progress. Default: True.
            method (str): Normalization method for age. Options: 'minmax' or 'standard'. Default: 'minmax'.
            apply_lung_segmentation (bool): Whether to apply lung segmentation to images. Default: False.
            segmentation_device (str or torch.device): Device for lung segmentation model. Default: None (auto-detect).
            segmentation_threshold (float): Threshold for lung mask binarization. Default: 0.5.
            segmentation_mask_value (int): Pixel value to use for masked-out areas. Default: 0.
        Returns:
            tuple: (train_dataset, test_dataset, val_dataset) where:
                   - train_dataset: PyTorchDataset with training samples
                   - test_dataset: PyTorchDataset with test samples
                   - val_dataset: PyTorchDataset with validation samples
        """
        np.random.seed(seed)
        
        if verbose:
            print("Creating CVAE dataset...")
        
        # Load and preprocess data
        if verbose:
            print("Loading images...")
        (
            train_images,
            val_images,
            test_images,
            split_results,
            train_masks,
            val_masks,
            test_masks,
        ) = self._load_images(
            size=img_size,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            verbose=verbose,
            method=method,
            apply_segmentation=apply_lung_segmentation,
            add_mask_channel=add_lung_mask_channel,
            segmentation_device=segmentation_device,
            segmentation_threshold=segmentation_threshold,
            segmentation_mask_value=segmentation_mask_value,
        )
        
        # Get dataframes with preprocessed data
        train_df = split_results['train_df']
        val_df = split_results['val_df']
        test_df = split_results['test_df']
        
        # Filter to only include rows where images were successfully loaded
        train_df = train_df.iloc[:len(train_images)]
        val_df = val_df.iloc[:len(val_images)]
        test_df = test_df.iloc[:len(test_images)]
        
        
        # One-hot encode the label columns for each split
        train_df = self._one_hot_encode_label_column(train_df, label_col='Label', num_classes=2, prefix='Label')
        val_df = self._one_hot_encode_label_column(val_df, label_col='Label', num_classes=2, prefix='Label')
        test_df = self._one_hot_encode_label_column(test_df, label_col='Label', num_classes=2, prefix='Label')

        train_labels = self._one_hot_encode_labels(train_df['Label'].values, num_classes=2)
        val_labels = self._one_hot_encode_labels(val_df['Label'].values, num_classes=2)
        test_labels = self._one_hot_encode_labels(test_df['Label'].values, num_classes=2)

        # Combine age and gender into metadata tensors
        train_metadata = np.column_stack([train_df['Patient Age Normalized'].values, train_df['Patient Gender Encoded'].values])
        train_metadata = torch.from_numpy(train_metadata).float()
        
        test_metadata = np.column_stack([test_df['Patient Age Normalized'].values, test_df['Patient Gender Encoded'].values])
        test_metadata = torch.from_numpy(test_metadata).float()

        val_metadata = np.column_stack([val_df['Patient Age Normalized'].values, val_df['Patient Gender Encoded'].values])
        val_metadata = torch.from_numpy(val_metadata).float()
        
        train_dataset = PyTorchDataset(train_images, train_labels, train_metadata, masks=train_masks)
        test_dataset = PyTorchDataset(test_images, test_labels, test_metadata, masks=test_masks)
        val_dataset = PyTorchDataset(val_images, val_labels, val_metadata, masks=val_masks)
        
        return train_dataset, test_dataset, val_dataset

    def get_pneumonia_dataframe(self):
        """Get a copy of the processed pneumonia patient dataframe.
        
        Returns:
            pd.DataFrame: Cleaned pneumonia patient records.
        """
        return self.pneumonia.copy()
    
    def get_healthy_dataframe(self):
        """Get a copy of the processed healthy patient dataframe.
        
        Returns:
            pd.DataFrame: Cleaned healthy patient records.
        """
        return self.healthy.copy()

    def split_by_patient(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, verbose=True, method='minmax'):
        """Create patient-level train/validation/test splits ensuring reproducibility.
        
        Performs stratified patient-level splitting so that all images from the same
        patient appear in only one split. This is important for avoiding data leakage
        during model training and evaluation. Uses a fixed seed for reproducible splits
        across multiple runs.
        
        Args:
            train_ratio (float): Fraction of patients for training. Default: 0.7.
            val_ratio (float): Fraction of patients for validation. Default: 0.15.
            test_ratio (float): Fraction of patients for testing. Default: 0.15.
            seed (int): Random seed for reproducibility. Default: 42.
            verbose (bool): Whether to print split statistics. Default: True.
            
        Returns:
            dict: Dictionary containing split information with keys:
                  - 'train_df': Training dataframe
                  - 'val_df': Validation dataframe
                  - 'test_df': Test dataframe
                  - 'splits': Dict with detailed split statistics
                  - 'split_mapping': Dict mapping patient IDs to split names
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        # Load and prepare data
        self._load_dataframe()
        self._get_pneumonia_patients(verbose=verbose, method=method)
        self._get_healthy_patients(verbose=verbose)
        
        # Combine pneumonia and healthy dataframes
        combined_df = pd.concat([self.pneumonia, self.healthy], ignore_index=True)
        
        # Get unique patients
        unique_patients = combined_df['Patient ID'].unique()
        num_patients = len(unique_patients)
        
        if verbose:
            print(f"Total unique patients: {num_patients}")
        
        # Set seed for reproducibility
        np.random.seed(seed)
        
        # Shuffle patients in a fixed order
        shuffled_patients = np.random.permutation(unique_patients)
        
        # Calculate split indices
        train_size = int(num_patients * train_ratio)
        val_size = int(num_patients * val_ratio)
        
        train_patients = shuffled_patients[:train_size]
        val_patients = shuffled_patients[train_size:train_size + val_size]
        test_patients = shuffled_patients[train_size + val_size:]
        
        # Create patient to split mapping for reproducibility
        split_mapping = {}
        for patient in train_patients:
            split_mapping[patient] = 'train'
        for patient in val_patients:
            split_mapping[patient] = 'val'
        for patient in test_patients:
            split_mapping[patient] = 'test'
        
        # Create split dataframes
        train_df = combined_df[combined_df['Patient ID'].isin(train_patients)].reset_index(drop=True)
        val_df = combined_df[combined_df['Patient ID'].isin(val_patients)].reset_index(drop=True)
        test_df = combined_df[combined_df['Patient ID'].isin(test_patients)].reset_index(drop=True)
        
        # Calculate statistics
        splits_info = {
            'train': {
                'total_samples': len(train_df),
                'total_patients': len(train_patients),
                'pneumonia': len(train_df[train_df['Finding Labels'] == self.label]),
                'healthy': len(train_df[train_df['Finding Labels'] == 'No Finding']),
            },
            'val': {
                'total_samples': len(val_df),
                'total_patients': len(val_patients),
                'pneumonia': len(val_df[val_df['Finding Labels'] == self.label]),
                'healthy': len(val_df[val_df['Finding Labels'] == 'No Finding']),
            },
            'test': {
                'total_samples': len(test_df),
                'total_patients': len(test_patients),
                'pneumonia': len(test_df[test_df['Finding Labels'] == self.label]),
                'healthy': len(test_df[test_df['Finding Labels'] == 'No Finding']),
            }
        }
        
        # Print split statistics
        if verbose:
            print("\n--- Split Statistics (Patient-Level) ---")
            print(f"Seed: {seed} (for reproducibility)")
            for split_name, stats in splits_info.items():
                print(f"\n{split_name.upper()} set:")
                print(f"  Patients: {stats['total_patients']}")
                print(f"  Total images: {stats['total_samples']}")
                print(f"  Pneumonia: {stats['pneumonia']} images")
                print(f"  Healthy: {stats['healthy']} images")
                print(f"  Ratio: {stats['total_samples']/len(combined_df)*100:.1f}%")
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'splits': splits_info,
            'split_mapping': split_mapping,
            'seed': seed
        }


if __name__ == '__main__':
    """Example usage of the Preprocessing class."""
    
    from torch.utils.data import DataLoader
    
    # Initialize preprocessor for pneumonia classification
    preprocessing = Preprocessing('Pneumonia')
    
    # Example 1: Patient-level train/validation/test split
    print("=" * 60)
    print("EXAMPLE 1: Patient-Level Train/Validation/Test Split")
    print("=" * 60)
    
    split_results = preprocessing.split_by_patient(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,  
        verbose=True
    )
    
    train_df = split_results['train_df']
    val_df = split_results['val_df']
    test_df = split_results['test_df']
    
    print("\nDataframes created:")
    print(f"  Train: {len(train_df)} images")
    print(f"  Validation: {len(val_df)} images")
    print(f"  Test: {len(test_df)} images")
    
    # Verify no patient overlap between splits
    train_patients = set(train_df['Patient ID'].unique())
    val_patients = set(val_df['Patient ID'].unique())
    test_patients = set(test_df['Patient ID'].unique())
    
    overlap_train_val = len(train_patients & val_patients)
    overlap_train_test = len(train_patients & test_patients)
    overlap_val_test = len(val_patients & test_patients)
    
    print("\nPatient overlap verification:")
    print(f"  Train ∩ Val: {overlap_train_val} (should be 0)")
    print(f"  Train ∩ Test: {overlap_train_test} (should be 0)")
    print(f"  Val ∩ Test: {overlap_val_test} (should be 0)")
    
    # Example 2: Create CVAE-compatible dataset with splits
    print("\n" + "=" * 60)
    print("EXAMPLE 2: CVAE-Compatible Dataset (Original)") 
    print("=" * 60)
    
    # Create CVAE-compatible dataset
    train_dataset, test_dataset, val_dataset = preprocessing.create_cvae_dataset(
        img_size=(128, 128),
        verbose=True,
    )
        
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Display sample batch
    print("\n--- Sample Batch from DataLoader ---")
    sample_batch = next(iter(train_loader))
    images, labels, metadata = sample_batch
    
    print(f"Images shape: {images.shape}")  # [batch_size, 1, 128, 128]
    print(f"Labels shape: {labels.shape}")  # [batch_size, 2]
    print(f"Metadata shape: {metadata.shape}")  # [batch_size, 2]
    
    print(f"\nFirst sample:")
    print(f"  Image range: [{images[0].min():.4f}, {images[0].max():.4f}]")
    print(f"  Label (one-hot): {labels[0].tolist()}")
    print(f"  Metadata (age, gender): {metadata[0].tolist()}")
