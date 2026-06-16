# Reconstrução 2D para 3D baseada em aprendizado profundo para análise de buracos em vias públicas

# Deep Learning-Based 2D to 3D Reconstruction for Pothole Analysis

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> Include name, RA, and specialization focus of each group member. Groups must have at most three members.
> |Name | RA | Specialization|
> |--|--|--|
> | Adriel Bombonato | 291654 | Electrical Engineering|
> | Hasnat Hameed | 270284 | Civil Engineering|
> | Iniobong Nicholas Udeme | 298961 | Applied Physics|

[**Presentation Link (D2) Here**](https://docs.google.com/presentation/d/1DMDBgBiviRlcB0tcGtO3RpONQTYD3BqT/edit?usp=sharing&ouid=113310569201771357639&rtpof=true&sd=true)

## Abstract

Road potholes remain one of the most significant challenges affecting transportation infrastructure, vehicle safety, and maintenance planning worldwide. Traditional pothole inspection techniques rely heavily on manual surveys, LiDAR scanners, and specialized sensing equipment, making large-scale deployment expensive and operationally challenging. Although recent advances in computer vision have enabled automatic pothole detection and segmentation, most existing approaches remain limited to two-dimensional analysis and do not provide the geometric information required for engineering decision-making.

This project proposes a hybrid Deep Learning and 3D Computer Vision framework for reconstructing pothole geometry from monocular RGB images and estimating quantitative severity metrics, including depth, surface area, and volume. The proposed pipeline combines semantic segmentation, monocular depth estimation, Point-E diffusion-based point cloud generation, and Open3D geometric processing to transform ordinary road images into engineering measurements suitable for infrastructure monitoring and maintenance prioritization.

To improve robustness under diverse road conditions, the framework leverages synthetic pothole image generation using Stable Diffusion and LoRA fine-tuning. The resulting system provides a low-cost alternative to LiDAR-based inspection while enabling applications in intelligent transportation systems, smart city infrastructure monitoring, digital twins, and autonomous vehicle navigation.


---

## Introduction

Road transportation is the backbone of economic activity and mobility in most countries, particularly in Brazil, where approximately 65% of freight transportation depends on the road network [1]. As a result, pavement condition directly influences transportation efficiency, logistics costs, road safety, and overall economic productivity.

Among various forms of pavement distress, potholes represent one of the most common and hazardous road defects. Potholes develop due to repeated traffic loading, environmental degradation, water infiltration, and inadequate maintenance practices. Their presence not only reduces driving comfort but also increases vehicle operating costs, accelerates tire and suspension damage, contributes to fuel inefficiency, and poses significant safety risks to road users [2].
The severity of the problem is particularly evident in Brazil. According to the CNT Rodovias 2024 Survey, which evaluated more than 111,000 km of paved roads, only 33.5% of the assessed road network was classified as being in good or excellent condition, while approximately 66.5% was rated as regular, poor, or very poor [1]. Furthermore, 26.6% of Brazilian roads were classified as poor or very poor, highlighting the urgent need for more efficient road condition monitoring and maintenance strategies [1].

The economic consequences of deteriorated road infrastructure are substantial. The CNT estimates that approximately R$ 100 billion would be required to restore the evaluated road network to acceptable conditions [1]. In addition, poor pavement conditions resulted in the waste of nearly 1.18 billion liters of diesel annually by trucks and buses operating on Brazilian highways, generating significant economic losses and environmental impacts [1].
Beyond economic losses, pavement deterioration also affects road safety. Road accidents associated with inadequate infrastructure conditions generate annual losses exceeding R$ 14 billion, while thousands of road defects continue to compromise transportation safety across the national road network [3]. Poor pavement conditions have been linked to increased accident risk, vehicle damage, and emergency maintenance requirements, creating additional burdens on transportation agencies and road users [2], [3].

To address these challenges, transportation agencies require scalable and cost-effective technologies capable of continuously monitoring road conditions and prioritizing maintenance interventions based on objective severity measurements. Traditionally, road inspection relies on manual surveys, laser scanning systems, photogrammetric surveys, or LiDAR-based platforms. Although these approaches provide accurate geometric measurements, they are often expensive, labor-intensive, and difficult to deploy at large scales [4].

Recent advances in Artificial Intelligence (AI), Deep Learning, and Computer Vision have enabled automated pothole detection and segmentation from road imagery [5]. However, most existing approaches remain limited to two-dimensional analysis, focusing primarily on classification, detection, or segmentation tasks. While these methods can identify the presence of potholes, they generally fail to provide accurate geometric characterization, which is essential for engineering decision-making [6].

For pavement management systems, geometric attributes such as depth, surface area, and volume are critical indicators of pothole severity because they directly influence maintenance prioritization, repair cost estimation, risk assessment, and infrastructure management strategies [7]. Accurate quantification of these parameters can also support autonomous vehicle navigation systems by enabling more informed path planning and road hazard assessment [8].

Consequently, there is a growing need for intelligent systems capable of reconstructing pothole geometry from low-cost image data and converting visual information into quantitative engineering metrics. In response to this challenge, this research proposes a hybrid Deep Learning and 3D Computer Vision framework that integrates semantic segmentation, monocular depth estimation, Point-E diffusion-based point cloud generation [9], and Open3D geometric processing [10] to reconstruct pothole geometry from RGB images and estimate key severity metrics, including depth, area, and volume.

The proposed framework aims to provide a scalable, low-cost, and practical solution for intelligent infrastructure monitoring, smart city applications, autonomous vehicle systems, and data-driven road maintenance planning.

## Problem Description / Motivation
Road potholes are among the most common forms of pavement deterioration affecting transportation safety, driving comfort, and infrastructure maintenance costs. Traditional pothole inspection methods are often manual, time-consuming, and inefficient for large-scale road monitoring. Although many computer vision approaches can detect potholes from 2D images, they usually provide limited information about pothole geometry such as depth, volume, and severity.
Recent advances in deep learning, monocular depth estimation, and 3D reconstruction have made it possible to generate spatial information from ordinary road images. In particular, point cloud representation techniques enable accurate modeling of road surface structures and pothole geometry. Furthermore, modern generative and diffusion-based 3D reconstruction methods have improved the quality and consistency of point cloud generation from visual data.
This study proposes a framework for converting 2D road images into 3D point cloud representations for pothole detection and severity assessment. The proposed approach integrates monocular depth estimation, point cloud reconstruction, and geometric analysis to estimate pothole characteristics such as depth, area, and volume. The framework aims to provide a cost-effective and scalable solution for intelligent road condition monitoring and automated maintenance planning.

## Project Objective
**Main Objective**
Develop a Deep Learning-based framework capable of generating accurate 3D point cloud representations from monocular 2D pothole images for the estimation of critical geometric parameters, including depth, surface area, and volume.

**Specific Objectives**
1.	Segment pothole regions from RGB road images and extract relevant geometric information through monocular depth estimation.
2.	Generate and refine 3D point cloud representations of potholes using Point-E and Open3D.
3.	Reconstruct pothole geometry and estimate key geometric metrics, including depth, surface area, and volume.
4.	Evaluate the effectiveness of the proposed framework for pothole quantification and infrastructure monitoring applications.

## 2. Contributions

This project presents a novel framework for pothole geometric quantification from monocular RGB images by integrating Deep Learning, generative AI, and 3D computer vision techniques. The main contributions of this work are:

1.	Development of an end-to-end framework for 3D pothole reconstruction from 2D images, eliminating the need for expensive sensing technologies such as LiDAR or laser scanners.
2.	Integration of Point-E and Open3D for pothole geometry reconstruction, enabling the generation, refinement, and processing of 3D point clouds from monocular RGB images.
3.	Estimation of key geometric characteristics of potholes, including depth, surface area, and volume, which are essential for objective severity assessment and maintenance prioritization.
4.	Demonstration of a low-cost and scalable approach for road infrastructure monitoring, with potential applications in pavement management systems, smart cities, and autonomous vehicle road hazard assessment.
5.	Creation and utilization of synthetic pothole data using Stable Diffusion and LoRA fine-tuning, improving dataset diversity and supporting the development of robust AI models for real-world road conditions.

## Methodology
The proposed framework converts road images into quantitative outputs for pothole severity assessment and maintenance planning.
### 1. Hypothesis 
Our central hypothesis is that a generative 3D model (Point-E) can successfully reconstruct the topology of a pothole from a single monocular RGB image, enabling practical severity assessment (depth/volume) without requiring perfect metrological-grade stereo setups during inference. Operationally, we test the following:
- **Latent Space Scaling:** Feeding carefully padded square crops to a Generative Point Cloud diffusion model allows the extraction of 3D geometry whilst maintaining physical proportions intact.
- **Robustness via RANSAC:** Applying geometric leveling over training data guarantees that the generative model learns pure depth (the crater) without being biased by camera pitch or road inclination.

***Scope note:*** The project prioritizes practical utility, relative severity ranking, and successful architectural pipeline adaptation, rather than sub-millimeter full mesh reconstruction.

### 2. Geometric Core and Data Standardization
#### Geometric Leveling (RANSAC)
Unlike early assumptions that treated the road as a flat plane parallel to the camera by calculating simple depth medians, we implemented a robust mathematical leveling algorithm. We project the real road pixels into 3D space and use RANSAC (Random Sample Consensus) to find the exact equation of the asphalt plane. By subtracting this plane from the raw depth, we isolate purely the pothole's cavity ($z=0$ at street level), completely neutralizing camera tilt and road slope.

#### Point-E Constraints and Data Normalization
Generative 3D models like Point-E impose strict dimensional bottlenecks:
1. 2D Constraint: The CLIP image encoder demands perfect squares. We overcome this without distorting/stretching the pothole by implementing Synchronized Square Cropping with zero-padding.
2. 3D Constraint: The model natively expects exactly 1024 points bounded in a $[-1, 1]$ Cartesian cube. We fulfill this via Farthest Point Sampling (FPS) to downsample point clouds elegantly, and we isolate the global scaling factor in a `metadata.json` so the physical units (mm/cm) can be un-normalized for severity calculation post-inference.
#### Calibration Limitations Statement
The dataset metadata (PothRGDB) does not provide exact per-device camera intrinsics.
Implications:
- Absolute geometry (in centimeters) may contain systematic bias.
- Relative ranking remains meaningful.
Current handling strategy: Use physically plausible D415 intrinsics, keep scales explicit, and perform sensitivity tests.


To implement the aforementioned pipelines within the project's timeframe, we will utilize the following frameworks and libraries:
- Python & PyTorch: The core programming language and deep learning framework for model training, fine-tuning, and tensor operations.
- [Open3D](https://github.com/isl-org/Open3D): Python library for 3D point cloud processing and visualization.
- [Point-E](https://github.com/openai/point-e) from OpenAI

### Datasets and Evolution

|Dataset| Source | Descriptive Summary|
|--|--|--|
|**PothRGDB** | [Kaggle](https://www.kaggle.com/datasets/mahyeks/pothrgbd-rgb-and-depth-images-of-potholes) | Provides 1,000 paired RGB and depth (2.5D) images with YOLO annotations captured using an Intel RealSense camera as the primary dataset.|
|**Rui Fan's Stereo Pothole** | [Rui Fan GitHub](https://github.com/ruirangerfan/rethinking_road_reconstruction_pothole_detection) | Contains 79 pothole instances with high-precision 3D ground truth obtained from laser-scanned gypsum molds|

### 3.1 PothRGDB: Primary Training Dataset

**Authorship and Affiliations**

The PothRGDB dataset and its accompanying research were developed by:

• Mustafa YURDAKUL (Kırıkkale University, Computer Engineering Department, Kırıkkale, Turkey)

• Şakir TAŞDEMİR (Selçuk University, Computer Engineering Department, Konya, Turkey)

PothRGDB is a paired RGB and depth dataset of potholes, captured with an Intel RealSense D415 active stereo depth camera. Each sample provides:

- A full-frame RGB image of a road surface with one or more potholes.
- A paired 16-bit depth map aligned with the RGB frame.
- A YOLO-format bounding box annotation identifying the pothole region.

The dataset contains **998 sample entries** in the manifest. After integrity checking, 996 are valid. The batch EDA pipeline processed 992 of those successfully; 4 failed due to empty masks or unstable road-surface estimation.

**Dataset Collection Setup**

A portable system was meticulously designed to collect depth and RGB image data of potholes from various road surfaces. The system's architecture emphasizes mobility and efficiency, comprising key components integrated for seamless data acquisition [1]. 

<p align="center">
  <img src="images/PotRGB_Dataset_Collection.png" width="900">
</p>

<p align="center">
  <b>Fig. 1.</b> Schematic diagram of the pothole data collection system.
</p>

**Key Components:**

*   **LattePanda 3 Delta Computer:** Chosen for its compact structure, powerful processor, and extensive connectivity, providing a suitable platform for portable systems [1].
*   **Intel RealSense D415 Depth Camera:** Selected for its capability to capture high-resolution RGB images and precise depth maps, essential for accurate pothole dimension measurement [1].
*   **Touchscreen:** Offers instant user feedback, simplifying data viewing and management during collection [1].
*   **Powerbank:** Ensures uninterrupted power supply, enabling prolonged mobile data collection sessions [1].

## 5. Dataset Characteristics: RGB, Depth, and Segmentation Masks

PothRGDB is a paired RGB and depth dataset of potholes, captured using the Intel RealSense D415 active stereo depth camera. Each sample within the dataset provides [1]:

*   **Full-frame RGB Images:** High-resolution color images of road surfaces, often containing one or more potholes. These are the visual inputs for detection and analysis.

<p align="center">
  <img src="images/RGB_Pothole_Images.png" width="900">
</p>

<p align="center">
  <b>Fig. 2.</b>  Sample images from the PothRGBD dataset.
</p>

*   **Paired 16-bit Depth Maps:** Precisely aligned with their corresponding RGB frames, these maps provide crucial 3D geometric information, indicating the distance of each pixel from the camera.

*   **YOLO-format Segmentation Annotations:** The dataset includes bounding box annotations in YOLO format, but more importantly, it is **labeled for segmentation**. This means that for each pothole, a precise mask is provided, outlining the exact pixels belonging to the pothole region. This allows for accurate perimeter and depth measurements, going beyond simple bounding box detection.

<p align="center">
  <img src="images/RGB_Segmentation_Data.png" width="900">
</p>

<p align="center">
  <b>Fig. 3.</b>  Sample segmentation results of the proposed model on the PothRGBD test set
</p>


## 6. Data Processing and Quality Control

The dataset underwent a rigorous processing and quality control workflow to ensure data integrity and reliability. This multi-stage process involved initial integrity checks, batch EDA, outlier detection, and a final manual review [1].

```mermaid
--- 
title: PothRGDB Data Processing Workflow
---
graph LR
    A[RGB-D Acquisition] --> B{Integrity Check}
    B --> C[Batch EDA Pipeline]
    C --> D{Outlier Detection}
    D --> E[Manual Review]
    E --> F[Final Training Set]
    
    subgraph Data Cleaning
    B
    C
    D
    E
    end
```

**Workflow Details:**

*   **Integrity Check:** Initial validation of the 1000 collected samples, resulting in 996 valid entries.
*   **Batch EDA Pipeline:** Processed 992 valid samples, with 4 failures attributed to empty masks or unstable road-surface estimation.
*   **Outlier Detection:** Performed on log-transformed values (IQR on log-volume and log-depth) to identify physically implausible samples. This step prevented the incorrect discarding of genuinely large potholes [1]. A total of 29 samples were flagged due to sensor failures, such as depths exceeding 5,000 mm or volumes over 1,000,000 cm³ [1].
*   **Manual Review:** A final review of prepared 3D point clouds ensured the quality and accuracy of the dataset.

## 7. Key EDA Metrics

Exploratory Data Analysis revealed critical insights into the characteristics of the potholes within the dataset. The volume and depth distributions exhibited a heavy-tailed, approximately log-normal shape [1].

| Metric                    | Median  | 95th Percentile |
| :------------------------ | :------ | :-------------- |
| Volume (cm³)              | 4,464   | 93,571          |
| Max depth (mm)            | 72      | 522             |
| Mask fraction             | 0.209   | 0.501           |
| Missing depth fraction    | -       | 1.67%           |

## 8. Final Dataset Size

After comprehensive filtering and manual review, the **final training set of PothRGDB contains 975 samples**. This refined dataset provides a robust foundation for training and evaluating models for pothole detection and 3D measurement.

#### Calibration Limitations

Converting a depth map into a 3D point cloud requires **camera intrinsic parameters**: the focal lengths (how much the lens magnifies the scene) and the principal point (the pixel coordinates of the optical center). Together, these four numbers define the mapping from a pixel location and its measured depth to a real-world 3D coordinate; without them, the reconstructed geometry can be systematically scaled or skewed.

PothRGDB does not provide per-device intrinsics. Because the dataset was collected with Intel RealSense D415 cameras, we use the typical factory intrinsics published by Intel for that model as a reasonable approximation. However, every physical camera differs slightly from the factory nominal values due to manufacturing variation and mounting geometry, and field measurements such as this dataset can deviate further. This means that the absolute metric values we compute (depth in cm, volume in cm³) may carry a systematic scale error that is the same across all samples but cannot be corrected without knowing the true per-device calibration.

In practice this means: relative comparisons between samples are meaningful, but the absolute numbers should be interpreted as estimates, not ground-truth measurements. This limitation is explicitly acknowledged throughout the evaluation, and all severity thresholds were chosen to be well-separated enough that small calibration errors do not move samples across bin boundaries.

### 3.2 Rui Fan Stereo Pothole Dataset: Held-Out Evaluation Benchmark

The Rui Fan dataset provides high-precision 3D ground truth for pothole evaluation. Three synthetic gypsum molds (model1, model2, model3) were scanned with a laser profilometer, achieving RMSE of 2.23 mm. Each model folder contains:

- Left-camera PNG images from multiple viewpoints.
- Ground-truth PLY files, one per section of the cast.

The benchmark subset used for evaluation (`rethinking_road_reconstruction_pothole_detection-main/dataset/`) contains 54 PNG images and 13 PLY files across 3 model folders, yielding 27 prepared evaluation samples after the benchmark preparation pipeline is applied.

This dataset was kept strictly separate from training data throughout the entire project. It acts as a held-out test contract: it is never used to select hyperparameters, tune augmentation probabilities, or choose checkpoints.

### Workflow

```mermaid
graph LR
        Input["Full RGB Image<br>and Camera Intrinsics"]
        Crop["Padded Square Crop"]
        NormCloud["Normalized 3D Cloud<br>[-1, 1]"]
        Output["Scaled 3D Topology"]

        subgraph Stage1 [2D Segmentation]
            Seg["Generic 2D Segmenter"]
        end

        subgraph Stage2 [3D Generation]
            PointE["Point-E Diffusion"]
        end

        %% Image processing flow
        Input --> Seg
        Seg --> Crop
        Crop --> PointE
        PointE --> NormCloud
        
        %% Scale bypass flow (The solution to Scale Ambiguity)
        Input -.->|"Calibration Data"| ScaleNode{"Apply Camera<br>Intrinsics"}
        NormCloud --> ScaleNode
        ScaleNode --> Output

        classDef dataNode fill:#ffffff,stroke:#000000,stroke-width:1px,color:#000000,shape:rect
        classDef processNode fill:#fcfcfc,stroke:#000000,stroke-width:1px,color:#000000,shape:rect
        classDef logicNode fill:#ffffff,stroke:#000000,stroke-width:1px,stroke-dasharray: 2 2,color:#000000
        
        class Input,Crop,NormCloud,Output dataNode
        class Seg,PointE processNode
        class ScaleNode logicNode

        style Stage1 fill:none,stroke:#333333,stroke-width:1px,stroke-dasharray: 4 4
        style Stage2 fill:none,stroke:#333333,stroke-width:1px,stroke-dasharray: 4 4
```

## 4. Exploratory Data Analysis

### 4.1 Dataset Inventory and Integrity

The EDA pipeline began with a structural inventory: loading the manifest, counting paired files, checking mask completeness, and identifying duplicate sample IDs.

Two samples were flagged as duplicates, each having two sets of images, depths, and labels. These were noted for review but retained in the general statistics, since the root cause was a dataset collection artifact.

### 4.2 Geometric Metric Extraction

For each valid sample the pipeline:

1. Loaded the depth map and RGB image.
2. Applied the YOLO bounding box to isolate the masked pothole region.
3. Estimated the road surface depth as the median of a ring of pixels surrounding the mask.
4. Projected the pothole pixels into 3D using approximate D415 intrinsics.
5. Computed volume (integration of depth delta over surface area), maximum depth delta, and mask fraction.

The road-surface estimation uses a flat-plane approximation (median of surrounding ring), not a full RANSAC plane fit. This is a deliberate simplification acceptable for the EDA phase; the actual training preprocessing uses RANSAC-based geometric leveling (see Section 5).

### 4.3 Outlier Handling

The initial version of the EDA applied IQR-based outlier detection on raw linear values. This was updated to apply IQR on log-transformed values (`np.log1p`) after observing that the distribution is log-normal. The updated approach correctly preserves genuinely large potholes while flagging the implausible sensor artifacts.

The 29 flagged samples are excluded from training via the quality gate in the preprocessing pipeline.

### 4.4 Missing Depth Analysis

The missing depth fraction, the proportion of masked pothole pixels with zero depth reading, was added as an explicit EDA metric: mean 1.67% across the dataset. While small on average, individual samples show much higher rates. Visual inspection of the top outliers confirmed that the failure modes are systematic (water, shadows, occlusion) rather than random sensor noise. This finding is used as a direct justification for the generative approach: the model must infer geometry where the sensor is structurally blind.

---

## 5. Data Preparation and Preprocessing

### 5.1 Overview

The preprocessing pipeline converts the raw PothRGDB samples (RGB images, depth maps, YOLO masks) into a format compatible with Point-E's constraints:

- **2D input:** a square RGB image crop, zero-padded to preserve aspect ratio.
- **3D target:** a normalized point cloud of exactly 1024 points, bounded in a [-1, 1] Cartesian cube, with the pothole axis oriented along positive Z.

The output of the pipeline is a prepared dataset directory containing the square RGB crops, the normalized point cloud tensors, depth heatmaps for visual inspection, and a metadata file with the per-sample scale factor needed to recover physical dimensions.

### 5.2 Square Crop and Padding

Point-E's CLIP image encoder requires a square input. Stretching or cropping the pothole region would distort the geometric proportions the model needs to learn.

The first step is to compute a square bounding box centered on the YOLO mask, with a small margin of context pixels around the pothole. Because the mask may be close to the image border, the bounding box can exceed the original image boundaries. Those out-of-bounds areas must be filled in before the crop can be passed to the encoder.

A naive solution is to fill the missing area with black pixels (zero-padding). This is adequate when the pothole is well inside the image, but when the boundary falls near the image edge, large black rectangles appear next to the pothole. CLIP encoders are sensitive to this kind of hard edge artifact, since they are pretrained on natural photographs where borders like that do not occur.

To address this, the pipeline uses a **hybrid padding strategy** that selects the filling method based on how much padding is required relative to the crop size:

- **Reflect padding** (used when the missing border is small, less than 15% of the crop): the missing area is filled by mirroring the nearby image content, producing natural-looking texture continuity at the edge.
- **Inpainting** (used when the missing border is large, 15% or more): the missing area is filled using a classical inpainting algorithm that propagates surrounding texture into the gap, avoiding the hard black border that would confuse the encoder.

This strategy was chosen because both extremes are problematic: purely zero-padded crops introduce unnatural borders, while always inpainting adds unnecessary computation for crops that are mostly inside the image.

### 5.3 Geometric Leveling with RANSAC

Depth maps encode the raw distance from the camera to each pixel, not the depth of the pothole relative to the road surface. A camera mounted at an angle introduces a systematic tilt: even a flat road has a depth gradient across the frame.

To isolate the pure cavity geometry, the pipeline fits a plane to the road surface pixels around the pothole and subtracts it:

1. Project the surrounding road pixels into 3D using the approximate D415 intrinsics.
2. Fit a plane to those 3D points using RANSAC (Random Sample Consensus) to reject outlier pixels (cracks, dirt, other road features).
3. Subtract the fitted plane from the pothole depth values.
4. The result is a leveled point cloud where Z=0 is the local road surface and positive Z encodes the cavity depth.

RANSAC is used specifically because a simple mean or median of the surrounding ring is sensitive to non-planar road features and camera calibration errors.

### 5.4 Point Cloud Normalization

Point-E expects point clouds bounded in a [-1, 1] cube. The leveled pothole cloud is normalized globally:

1. Compute the extent of the cloud across all three axes.
2. Divide all coordinates by the maximum extent.
3. Record the scale factor in a metadata file per sample.

The scale factor is critical for severity assessment: after inference, multiplying the generated point cloud by the scale factor and dividing by 1000 converts it to meters, from which effective depth in cm can be computed.

### 5.5 Fixed Point Count (FPS and Random Upsampling)

Point-E requires exactly 1024 points. The raw leveled point cloud may contain anywhere from a few hundred to tens of thousands of points depending on the mask size and depth coverage.

- **If over 1024:** Farthest Point Sampling (FPS) selects the most spatially representative subset, preserving the global shape while downsampling.
- **If under 1024:** Random upsampling with replacement fills the gap.

Random upsampling is a known limitation: small potholes may have repeated points in the training target, which slightly inflates apparent point density for those samples.

### 5.6 Manual Quality Review

After generating the full prepared dataset, a manual quality review removed samples with clearly degenerate point clouds (flat or collapsed structures, obviously wrong scale, or sensor failures that survived the automated outlier filter). The final training set after this review contains **975 samples**.

### 5.7 Rui Fan Benchmark Preparation

The Rui Fan evaluation set is prepared by a dedicated benchmark preparation step, completely separate from the training pipeline:

1. Walks each model folder and collects the paired left-camera images and GT PLY sections.
2. Merges all PLY sections for a given model into one ground-truth cloud.
3. Standardizes the merged cloud to the same Point-E contract (1024 points, [-1, 1] normalization) using the same geometric standardization described in Section 5.
4. Saves the prepared images, prepared clouds, and per-sample scale metadata to a dedicated evaluation directory.

The Rui Fan prepared set is kept completely separate from the training data and is never touched until the final evaluation step.

---

## 6. Model: Point-E Fine-Tuning

### 6.1 Architecture Overview

> *Note: A full theoretical description of Point-E and the underlying diffusion framework is provided in a separate section authored by a project co-author. This section focuses only on the adaptation and fine-tuning decisions.*

Point-E is an open-source model released by OpenAI, and this project uses the official repository directly as a library. The generation pipeline, the model weights, the CLIP image encoder, and the diffusion loss are all taken from the original implementation without modification. Our contribution is the fine-tuning of those weights on the pothole domain and the surrounding data preparation and evaluation infrastructure.

Point-E uses a CLIP-conditioned image-to-point-cloud diffusion model. The base40M model generates 1024 points directly conditioned on a single square RGB image. An optional upsampler can refine the output to a higher point count, but it is not used in this project.

### 6.2 Training Inputs and Targets

Each training sample is a pair:

- **Image input:** a 512×512 square RGB crop of the pothole, zero-padded to preserve aspect ratio, encoded through Point-E's CLIP encoder.
- **Point cloud target:** the normalized 6D tensor (XYZ + RGB) produced by the preprocessing pipeline.

The model is trained using the native Point-E diffusion loss, which measures the denoising network's ability to recover the original cloud from a noise-corrupted version at a randomly sampled noise level.

---

## 7. Data Augmentation

Augmentation is applied only to the 2D image inputs. The 3D point cloud targets are never augmented independently, because decoupled augmentation would break the image-to-cloud pairing.

### 7.1 Geometric Augmentations

Only one geometric augmentation is applied: **horizontal flip**. When the image is horizontally flipped, the point cloud X coordinate is negated by the same amount. This ensures the image-cloud pair remains geometrically consistent. The correctness of this paired behavior was validated visually during prototyping.

### 7.2 Pixel-Only Augmentations

The following augmentations are applied to the image only, leaving the point cloud unchanged:

| Augmentation | Purpose |
|---|---|
| Fake shadow | Simulates shadowed regions common in field images |
| Color jitter | Varies brightness, contrast, and saturation for lighting robustness |
| Gaussian blur | Simulates optical blur from camera distance or defocus |
| Motion blur | Simulates blur from camera movement or vibration |
| Cutout | Random rectangular masking to improve robustness to partial occlusion |

### 7.3 Config-Driven Pipeline

The augmentation pipeline is fully config-driven and opt-in. Only transforms explicitly listed in the training configuration are applied. An empty configuration means samples pass through unchanged, which was intentional: it allows comparing augmented and unaugmented runs without any code changes.

Validation never receives augmentation. The validation set is always evaluated on clean, unmodified samples.

### 7.4 Augmentation Logging

Each augmented sample is recorded in a per-run JSONL augmentation log, tracking which transforms were applied to which sample ID in which epoch. A per-epoch terminal summary shows per-transform counts. This makes the augmentation history fully auditable.

---

## 8. Training System

### 8.1 Entry Point and Configuration

The training pipeline is orchestrated by a single entry point that:

1. Loads and merges a JSON configuration file.
2. Creates a timestamped artifact directory for the run.
3. Seeds all random number generators when a seed is provided, and enforces deterministic behavior.
4. Saves a resolved copy of the configuration alongside the run artifacts.
5. Initializes the Point-E base model with the pre-trained weights.
6. Builds separate train and validation dataloaders, split by sample ID.
7. Runs the training loop.
8. Writes a final run metadata summary after training completes.

### 8.2 Optimizer and Mixed Precision

The optimizer is **AdamW**, applied only to the model parameters that require gradient updates. Mixed precision training is enabled to reduce GPU memory usage and increase throughput without significant loss of gradient stability. Gradient clipping is available as a configurable option.

### 8.3 Learning Rate Scheduling

Cosine annealing scheduling was introduced in Runs 3 and 4. The scheduler reduces the learning rate from the initial value to `eta_min` following a cosine curve over the total number of epochs.

### 8.4 Sequential Training Runs

The best model was produced through four sequential fine-tuning runs. Each run resumed from the best checkpoint of the previous one.

Each augmentation transform has an independent application probability per sample (e.g. "flip 0.5" means horizontal flip is applied with 50% chance). The table below summarizes the main configuration choices and results per run.

| Run | Epochs | LR | LR Scheduler | Main augmentation probabilities | Best Epoch | Best Val Loss |
|---|---|---|---|---|---|---|
| 1 | 100 | 1e-5 | None | flip 40%, shadow 30%, jitter 40% | 89 | 0.19903 |
| 2 | 500 | 3e-5 | None | flip 30%, shadow 30%, jitter 30%, all others 30% | 370 | 0.19706 |
| 3 | 600 | 3e-5 | Cosine (η_min=1e-6) | flip 50%, shadow 50%, jitter 50% | 580 | 0.19604 |
| 4 | 800 | 1e-5 | Cosine (η_min=1e-6) | flip 50%, shadow 50%, jitter 40%, blur 25% | 590 | **0.19579** |

The progressive decrease in best validation loss across all four runs reflects the benefit of sequential fine-tuning combined with refined augmentation policies and cosine scheduling.

---

## 9. Reproducibility

Reproducibility was treated as a first-class requirement. Every training run produces a complete and self-contained audit trail.

### 9.1 Timestamped Run Directories

Each run writes to a dedicated timestamped directory (e.g., `artifacts/runs/20260601_142315/`). All outputs are isolated within this directory; multiple runs never overwrite each other.

### 9.2 Resolved Config Snapshot

Before training begins, the fully resolved configuration is saved alongside the run artifacts. The exact hyperparameters that produced any checkpoint are always co-located with that checkpoint.

### 9.3 Run Metadata Summary

A final JSON metadata file is written after training completes, capturing: run timestamp, artifact paths, base model, resumed-from path, start and end epoch, best epoch, best validation metric, elapsed time, learning rate, train and validation set sizes, and validation interval.

### 9.4 Augmentation and Validation Logs

Augmentation records are stored as JSONL under the run root (one record per augmented sample, per epoch). Validation results are stored as a separate JSONL log, capturing epoch, val loss, Chamfer Distance (if computed), and best-checkpoint events.

### 9.5 Seed and Determinism

When a seed is provided in the configuration, all random number generators are seeded and deterministic GPU operation is enforced. This does not guarantee bit-identical results across different hardware, but makes runs on the same machine reproducible.

### 9.6 Checkpoint Format

Each checkpoint stores the complete training state: the model weights, the optimizer state, the mixed-precision scaler state, the learning rate scheduler state, the best validation metric seen so far, and the epoch at which that best was achieved. Resuming from a checkpoint restores all of these, so training continues from exactly where it was interrupted without resetting the best-metric baseline.

---

## 10. Validation and Best Checkpoint Selection

### 10.1 Validation Split

The split is defined by an explicit list of reserved sample identifiers in the training configuration. **50 samples** were reserved for validation across all runs. The training set is built from all remaining samples.

### 10.2 Validation Cadence and Pass

Validation runs at the end of each epoch by default, though the interval is configurable. The pass iterates over the full validation set with the model in evaluation mode and without gradient computation, accumulating mean validation loss.

### 10.3 Best Checkpoint Criterion

The monitored metric defaults to validation loss. When a strict improvement is observed, the best checkpoint is saved and the best-epoch number is updated. This best value is persisted inside the checkpoint file so that resuming a run does not reset the selection threshold.

Chamfer Distance can be configured as the monitored metric instead. If Chamfer computation is not enabled at the same time, the system warns and falls back to validation loss.

---

## 11. Inference and Final Evaluation

### 11.1 Inference Pipeline

For final-test inference, the model is loaded from the best training checkpoint and run on each image in the prepared benchmark set. The pipeline:

1. Reads the per-sample scale metadata computed during benchmark preparation.
2. Runs the Point-E generation on each image.
3. Recovers physical scale by multiplying the normalized output by the stored scale factor and converting to meters.
4. Saves one predicted point cloud per sample. Already-completed samples are skipped, making the run resumable.

### 11.2 Evaluation Pipeline

The evaluation step matches each prediction against the prepared ground-truth cloud by sample identifier and writes:

- A per-sample report with the geometric fidelity score (Chamfer Distance), predicted and ground-truth effective depth in cm, depth error, expected and predicted severity bins, and whether the bin matched.
- An aggregate summary with mean Chamfer Distance, mean depth error, severity match rate, and a 3×3 severity confusion matrix.

### 11.3 Depth Convention and Scale Alignment

One non-trivial aspect of the evaluation is that the ground-truth clouds and the model predictions use different depth conventions internally. During data preparation, the pothole cavity is encoded with depth growing in the positive direction, essentially treating the cavity as a hill rather than a hole, because this representation was more stable for training. The severity evaluation, however, measures depth as the downward extension of the cavity, which requires the opposite sign.

Before any metric is computed, the ground-truth clouds are flipped to match the convention used by the predictions, and both sides are converted from the normalized [-1, 1] space back to metric units (centimeters) using the per-sample scale factor stored during preprocessing. Only after this alignment are the geometric and severity metrics computed.

For effective depth, the evaluation uses the 5th percentile of the deepest points rather than the absolute minimum. This choice is deliberate: generative models occasionally produce stray points far above or below the main structure. Using the minimum would make the depth estimate highly sensitive to these outliers, while the 5th percentile provides a stable and representative measure of how deep the cavity actually is.

### 11.4 Severity Classification

Effective depth is mapped to three engineering bins:

| Bin | Threshold |
|---|---|
| Low | < 7 cm |
| Medium | 7–10 cm |
| High | > 10 cm |

A severity match is recorded when predicted and ground-truth bins are equal.

### 11.5 Final Evaluation Results

The best checkpoint (Run 4, epoch 590, val loss 0.1957) was evaluated on the 27 prepared Rui Fan samples:

| Metric | Value |
|---|---|
| Evaluated samples | 27 |
| Mean Chamfer Distance | 0.0883 |
| Mean depth error (cm) | 1.51 |
| Severity match rate | 1.0 (27/27) |

**Severity confusion matrix:**

| Predicted → | Low | Medium | High |
|---|---|---|---|
| **GT: Low** | 27 | 0 | 0 |
| **GT: Medium** | 0 | 0 | 0 |
| **GT: High** | 0 | 0 | 0 |

All 27 benchmark samples fall in the low-severity bin. The model classified all of them correctly, with a mean depth error of 1.51 cm, well below the 7 cm threshold.

**Important caveat:** the benchmark's gypsum molds are relatively shallow by design, so the result only exercises the low-severity bin. The medium and high bins have zero ground-truth samples. This result cannot be extrapolated to assert correct behavior on deeper potholes, and the severity match rate of 1.0 reflects this narrow coverage, not general system robustness.

---

## 12. Limitations and Open Points

### 12.1 Calibration Uncertainty

Absolute depth values depend on camera intrinsics not provided by PothRGDB. The approximate D415 intrinsics may introduce a systematic scale offset. Relative severity ranking is more reliable than absolute geometric precision.

### 12.2 Benchmark Coverage

The Rui Fan benchmark contains only 27 samples across 3 physical pothole molds, all in the low-severity bin. The model's behavior on medium and high severity potholes is not validated by these results.

### 12.3 Upsampler Not Used

The Point-E upsampler is not integrated. All results represent the base40M output only. Adding the upsampler may improve geometric fidelity but would require additional training and evaluation work.

### 12.4 Scale Ambiguity

The generative model produces a normalized output without intrinsic scale. Scale is recovered from preprocessing metadata, which inherits the calibration uncertainty described above.

### 12.5 Random Upsampling for Small Clouds

Samples with fewer than 1024 valid depth points are upsampled by random resampling with replacement. This may cause the model to overfit to repeated points in small-pothole training samples.

### 12.6 Training Distribution

PothRGDB was collected in a limited geographic context with a single camera model. Generalization to potholes with different aspect ratios, pavement types, or illumination conditions beyond the training distribution is not guaranteed.



## Schedule

To accommodate the two-month deadline, the project will follow an 8-week schedule:

- **Weeks 1-2 (Data Preparation):** Conversion of the PothRGBD dataset from 2.5D depth maps to 3D point clouds using algebraic back-projection. Execution of Furthest Point Sampling (FPS) and extraction of Surface Normals.
- **Weeks 3-5 (Generative Model Fine-Tuning):** Setup of the 3D diffusion architecture (Point-E) and execution of fine-tuning conditioned on 2D pothole images, targeting sparse point cloud generation. This extended 3-week period accounts for the iterative training, hyperparameter tuning, and computational resources required for diffusion models.
- **Week 6 (Metrics & Baseline Setup):** Implementation of the evaluation scripts (Chamfer Distance, Outlier-Resistant MAE, Severity Bins) and validation of the testing pipeline using the Rui Fan gypsum mold ground truth dataset.
- **Week 7 (Inference & Comparison):** Generating final 3D point clouds from the test set using the Generative Model. Execution of the geometric comparative analysis back-projected to real-world scale to evaluate structural fidelity and severity accuracy.
- **Week 8 (Final Deliverables):** Final code refinements, calculation of pothole volumetrics, and elaboration of the final project report and presentation.

## Bibliographic References

[1] Confederação Nacional do Transporte (CNT), "Pesquisa CNT de Rodovias 2024", Brasília, Brazil, 2024.

[2] American Society of Civil Engineers (ASCE), "Infrastructure Report Card", 2021.

[3] Confederação Nacional do Transporte (CNT), Road Safety and Economic Impact Reports, 2024.

[4] Puente, I., González-Jorge, H., Martínez-Sánchez, J., and Arias, P., "Review of Mobile Mapping and LiDAR Technologies for Road Inspection", Measurement, vol. 90, pp. 222–231, 2016.

[5] Maeda, H., Sekimoto, Y., Seto, T., Kashiyama, T., and Omata, H., "Road Damage Detection Using Deep Neural Networks with Images Captured Through a Smartphone", arXiv:1801.09454, 2018.

[6] Fan, R., Ai, Y., and Dahnoun, N., "Road Surface 3D Reconstruction Based on Dense Subpixel Disparity Map Estimation", IEEE Transactions on Image Processing, 2019.

[7] Shahin, M. Y., Pavement Management for Airports, Roads, and Parking Lots, Springer, 2005.

[8] Grigorescu, S., Trasnea, B., Cocias, T., and Macesanu, G., "A Survey of Deep Learning Techniques for Autonomous Driving", Journal of Field Robotics, vol. 37, no. 3, pp. 362–386, 2020.

[9] Nichol, A., Jun, H., Dhariwal, P., Mishkin, P., and Chen, M., "Point-E: A System for Generating 3D Point Clouds from Complex Prompts", OpenAI, 2022.

[10] Zhou, Q.-Y., Park, J., and Koltun, V., "Open3D: A Modern Library for 3D Data Processing", arXiv:1801.09847, 2018.

BASTICO, Matteo, et al. Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation. En Thirteenth International Conference on 3D Vision. 2026. Available at: https://arxiv.org/abs/2511.05308

FAN, Rui, et al. Rethinking road surface 3-D reconstruction and pothole detection: From perspective transformation to disparity map segmentation. IEEE Transactions on Cybernetics, 2021, vol. 52, no 7, p. 5799-5808. Available at: https://arxiv.org/abs/2012.10802

GEIGER, Andreas; LENZ, Philip; URTASUN, Raquel. Are we ready for autonomous driving? the kitti vision benchmark suite. In: 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012. p. 3354-3361. Available at https://ieeexplore.ieee.org/abstract/document/6248074

HIGO, Kazuki, et al. TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models. arXiv preprint arXiv:2505.04050, 2025. Available at: https://arxiv.org/abs/2505.04050

HUANG, Zixuan, et al. Spar3d: Stable point-aware reconstruction of 3d objects from single images. En Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. p. 16860-16870. Available at: https://arxiv.org/abs/2501.04689

LI, Zhengqi; SNAVELY, Noah. Megadepth: Learning single-view depth prediction from internet photos. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. p. 2041-2050. Available at https://arxiv.org/abs/1804.00607

NICHOL, Alex et al. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. Available at: https://arxiv.org/pdf/2212.08751

RANFTL, René et al. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, v. 44, n. 3, p. 1623-1637, 2020. Available at https://arxiv.org/abs/1907.01341

RANFTL, René; BOCHKOVSKIY, Alexey; KOLTUN, Vladlen. Vision transformers for dense prediction. In: Proceedings of the IEEE/CVF international conference on computer vision. 2021. p. 12179-12188. Available at https://arxiv.org/abs/2103.13413

TANG, Xiang; LI, Ruotong; FAN, Xiaopeng. Recent Advances in 3D Object and Scene Generation: A Survey. arXiv preprint arXiv:2504.11734, 2025. Available at: https://arxiv.org/abs/2504.11734

WANG, Zhengren. 3d representation methods: A survey. arXiv preprint arXiv:2410.06475, 2024. Available at: https://arxiv.org/abs/2410.06475

YURDAKUL, Mustafa; TASDEMIR, Şakir. An enhanced yolov8 model for real-time and accurate pothole detection and measurement. arXiv preprint arXiv:2505.04207, 2025. Available at https://arxiv.org/abs/2505.04207
