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

## Project Summary Description

Potholes are severe structural road failures that pose significant safety hazards and require frequent maintenance. Traditionally, detecting and quantifying these anomalies relies on manual inspection or expensive LiDAR sensors. The use of Unmanned Aerial Vehicles (UAVs) equipped with monocular RGB cameras offers a highly scalable and cost-effective alternative for road inspection. However, extracting accurate 3D topology from 2D aerial images of potholes presents a complex inverse problem. Potholes are strictly concave ("inwards") structures that suffer from severe visual occlusions, lack of inner texture, and deep shadows. Traditional deterministic computer vision methods and regressors struggle to map these occluded areas, often resulting in noisy or incomplete geometric reconstructions that prevent accurate volume estimation.

The main goal of this project is to investigate and implement a deep generative modeling pipeline capable of reconstructing the 3D geometry of road potholes from single 2D monocular images. Instead of relying on deterministic pixel matching, this project leverages the probabilistic nature of Diffusion Models to "hallucinate" and infer the occluded inner structures of the potholes in a controlled manner, learning the physical distribution of road anomalies to synthesize structurally coherent cavities.

The output of the generative model will be a 3D Point Cloud representing the geometric structure of the pothole. The model will take a single 2D image as the condition/input and probabilistically generate the spatial coordinates (X, Y, Z) of the points that form the asphalt's concavity.

Presentation Link [here](https://docs.google.com/presentation/d/1CM4DDgOnC9EBGzu9f0BrfLRTlg05Kgunh3uYEckwZDQ/edit?usp=sharing)

## Problem Description / Motivation
> Road potholes are among the most common forms of pavement deterioration affecting transportation safety, driving comfort, and infrastructure maintenance costs. Traditional pothole inspection methods are often manual, time-consuming, and inefficient for large-scale road monitoring. Although many computer vision approaches can detect potholes from 2D images, they usually provide limited information about pothole geometry such as depth, volume, and severity.
> Recent advances in deep learning, monocular depth estimation, and 3D reconstruction have made it possible to generate spatial information from ordinary road images. In particular, point cloud representation techniques enable accurate modeling of road surface structures and pothole geometry. Furthermore, modern generative and diffusion-based 3D reconstruction methods have improved the quality and consistency of point cloud generation from visual data.
> This study proposes a framework for converting 2D road images into 3D point cloud representations for pothole detection and severity assessment. The proposed approach integrates monocular depth estimation, point cloud reconstruction, and geometric analysis to estimate pothole characteristics such as depth, area, and volume. The framework aims to provide a cost-effective and scalable solution for intelligent road condition monitoring and automated maintenance planning.

## Objective
> To develop a framework for converting 2D road surface images into 3D point cloud representations for automated pothole detection and severity assessment.
> The proposed approach moves beyond visual detection by generating geometric estimates that support engineering decisions: where the defect is, how large it is, how deep it is, and how urgently it should be repaired.
> - To collect and preprocess road surface image datasets suitable for pothole detection and 3D reconstruction tasks.
> - To implement monocular depth estimation methods for extracting depth information from 2D road images.
> - To reconstruct road surfaces into 3D point cloud representations using estimated depth maps and image features.
> - To estimate pothole severity using geometric characteristics such as pothole depth, width, area, and volume.
> - To investigate the use of modern 3D reconstruction and generative modeling techniques for improving point cloud quality and structural consistency.
> - To evaluate the performance of the proposed framework using reconstruction accuracy, detection accuracy, and severity classification metrics.

## Methodology
> The proposed framework converts road images into quantitative outputs for pothole severity assessment and maintenance planning.
### 1. Hypothesis 
> Our central hypothesis is that a generative 3D model (Point-E) can successfully reconstruct the topology of a pothole from a single monocular RGB image, enabling practical severity assessment (depth/volume) without requiring perfect metrological-grade stereo setups during inference. Operationally, we test the following:
> - **Latent Space Scaling:** Feeding carefully padded square crops to a Generative Point Cloud diffusion model allows the extraction of 3D geometry whilst maintaining physical proportions intact.
> - **Robustness via RANSAC:** Applying geometric leveling over training data guarantees that the generative model learns pure depth (the crater) without being biased by camera pitch or road inclination.
***Scope note:***
> - The project prioritizes practical utility, relative severity ranking, and successful architectural pipeline adaptation, rather than sub-millimeter full mesh reconstruction.

### 2. Geometric Core and Data Standardization
#### Geometric Leveling (RANSAC)
> Unlike early assumptions that treated the road as a flat plane parallel to the camera by calculating simple depth medians, we implemented a robust mathematical leveling algorithm. We project the real road pixels into 3D space and use RANSAC (Random Sample Consensus) to find the exact equation of the asphalt plane. By subtracting this plane from the raw depth, we isolate purely the pothole's cavity ($z=0$ at street level), completely neutralizing camera tilt and road slope.

#### Point-E Constraints and Data Normalization
> Generative 3D models like Point-E impose strict dimensional bottlenecks:
> 1. 2D Constraint: The CLIP image encoder demands perfect squares. We overcome this without distorting/stretching the pothole by implementing Synchronized Square Cropping with zero-padding.
> 2. 3D Constraint: The model natively expects exactly 1024 points bounded in a $[-1, 1]$ Cartesian cube. We fulfill this via Farthest Point Sampling (FPS) to downsample point clouds elegantly, and we isolate the global scaling factor in a `metadata.json` so the physical units (mm/cm) can be un-normalized for severity calculation post-inference.
#### Calibration Limitations Statement
> The dataset metadata (PothRGDB) does not provide exact per-device camera intrinsics.
> Implications:
> - Absolute geometry (in centimeters) may contain systematic bias.
> - Relative ranking remains meaningful.
> Current handling strategy: Use physically plausible D415 intrinsics, keep scales explicit, and perform sensitivity tests.


To implement the aforementioned pipelines within the project's timeframe, we will utilize the following frameworks and libraries:
- Python & PyTorch: The core programming language and deep learning framework for model training, fine-tuning, and tensor operations.
- [Open3D](https://github.com/isl-org/Open3D): Python library for 3D point cloud processing and visualization.
- Point-E from OpenAI

### Datasets and Evolution

> List the datasets used in the project.
> For each dataset, include a mini-table in the model below and then provide details on how it was analyzed/used, as in the example below.

> |Dataset| Source | Descriptive Summary|
> |--|--|--|
> |**PothRGDB** | Research Repository | Provides 1,000 paired RGB and depth (2.5D) images with YOLO annotations captured using an Intel RealSense camera as the primary dataset.|
> |**Rui Fan's Stereo Pothole** | Research Repository | Contains 79 pothole instances with high-precision 3D ground truth obtained from laser-scanned gypsum molds|

#### Analysis and Preprocessing
> - **PothRGDB (Primary Training & Tuning):** Utilizing the camera's intrinsic parameters, we mapped algebraic back-projection to convert depth maps into 3D point clouds. An Exploratory Data Analysis (EDA) on 998 unique samples revealed that while the central tendency indicates moderate potholes (median volume ~4.5L, median max-depth ~72mm), the dataset occasionally suffers from extreme physical sensor artifacts. Over 100 samples were flagged as implausible outliers (e.g., reported depths > 5000mm) typically caused by water reflections and harsh shadows. To prevent corrupt learning, applying log-scale IQR outlier thresholds is essential to curate the training data. This dataset provides the necessary volume to learn the general distribution of road anomalies.
> - **Rui Fan's Dataset (Testing & Validation):** Due to its limited size (79 samples) but absolute structural fidelity (achieving an RMSE of 2.23 mm), this dataset is incredibly valuable. It will be strictly reserved as an independent gold-standard test set for the final geometric evaluation to prove the pipeline's capabilities.

### Architectural Design and Use Case Alignment
#### Two-Stage Inference Pipeline
> To ensure inputs match constraints correctly when deployed, the architecture assumes a Two-Stage Pipeline for final end-user inference:
> 1. 2D Segmentation & Cropping: The pothole was isolated from the surrounding environment.
> 2.  Generative 3D Reconstruction (2D->3D): Our core model (Point-E) receives the square RGB crop and outputs a point cloud (1024 points). Using saved scale factors, this generic 3D output is re-upscaled directly into real-world topological dimensions to measure depth and severity.

#### Use Case Realignment
> The project focuses on Infrastructure Auditing and Crowdsourcing (e.g., civic reporting via smartphone, slow-moving municipal fleet cameras). It does not target high-speed autonomous driving avoidance, simplifying constraints related to real-time processing and dynamic perspective shifts.

### Sensitivity Protocol (Intrinsics Robustness)
> To quantify the effect of unknown calibration without imposing massive computational overhead, we will conduct a post-inference intrinsics sensitivity analysis.
> - Scenario A (Nominal): baseline intrinsics for D415.
> - Scenario B: focal lengths (fx, fy) shifted by +/- 5% during metric calculation.
> - Scenario C: focal lengths (fx, fy) shifted by +/- 10% during metric calculation.
> By computing the variance of the severity metrics across scenarios, we establish the robustness of the ranking criteria independently from the exact factory calibration.

### Reporting Without Overclaim
> Use conservative language.
> What can be claimed:
> - The successful architectural adaptation of a 3D foundational model (Point-E) for civic infrastructure usage.
> - Pipeline robustness and geometric stability against imperfect sensor angles.
> What should be avoided:
> - Claims of absolute sub-millimeter precision matching LIDAR scans.

### Evaluation Framework
#### Geometric Fidelity (Point Cloud Similarity)
> Because downstream classification baselines have been scoped out, our evaluation focuses entirely on the topological accuracy of the Generative Model itself.
To ensure the model accurately recreates the exact spatial geometry, we calculate point cloud regression metrics on a strictly filtered validation set (clean, dry samples curated by our top-down pipeline).
> - **Primary Metric:** Chamfer Distance (measuring the mean distance between the generated 1024 points and the ground truth 1024 points extracted by the sensor).
> - **Secondary Metric:** RMSE for max depth extracted from both clouds.
> By comparing these metrics after scaling the generated data back to its physical limits (using the recorded metadata), we validate the generative fidelity of the pipeline.

## Experiments, Results, and Discussion of Results

> In the intermediate project submission (D2), this section may contain partial results, explorations of implemented solutions, and
> discussions about such experiments, including decisions to change the project trajectory or the description of new experiments as a result of these explorations.

> In the final project submission (D3), this section should list the main results obtained (not necessarily all), which best represent the fulfillment of the project objectives.

> The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.
> It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."
> What is expected from this section is that it presents and discusses only the most relevant results, highlighting the strengths and/or limitations of the methodology, emphasizing aspects of performance, and containing content that can be classified as organized, didactic, and reproducible sharing of knowledge relevant to the community.

## Conclusion

> The Conclusion section should recover the main information already presented in the report and point to future work.
> In the intermediate project submission (D2), it may contain information about which steps or how the project will be conducted until its completion.
> In the final project submission (D3), the conclusion is expected to outline, among other aspects, possibilities for the project’s continuation.


## Schedule

To accommodate the two-month deadline, the project will follow an 8-week schedule:

- **Weeks 1-2 (Data Preparation):** Conversion of the PothRGBD dataset from 2.5D depth maps to 3D point clouds using algebraic back-projection. Execution of Furthest Point Sampling (FPS) and extraction of Surface Normals.
- **Weeks 3-5 (Generative Model Fine-Tuning):** Setup of the 3D diffusion architecture (e.g., Point-E or DiPT) and execution of LoRA fine-tuning conditioned on 2D pothole images, targeting sparse point cloud generation. This extended 3-week period accounts for the iterative training, hyperparameter tuning, and computational resources required for diffusion models.
- **Week 6 (Metrics & Baseline Setup):** Implementation of the evaluation scripts (SNC, DCD, RMSE) and validation of the testing pipeline using the Rui Fan gypsum mold ground truth dataset. Deployment of the computationally efficient alternative regression pipeline (MiDaS/DPT) for comparative testing.
- **Week 7 (Inference & Comparison):** Generating final 3D point clouds from the test set using both the Generative Model and the Regression Model. Execution of the comparative analysis using the SNC and DCD metrics to evaluate structural fidelity.
- **Week 8 (Final Deliverables):** Final code refinements, calculation of pothole volumetrics, and elaboration of the final project report and presentation.

## Bibliographic References

BASTICO, Matteo, et al. Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation. En Thirteenth International Conference on 3D Vision. 2026. Available at: https://arxiv.org/abs/2511.05308

FAN, Rui, et al. Rethinking road surface 3-D reconstruction and pothole detection: From perspective transformation to disparity map segmentation. IEEE Transactions on Cybernetics, 2021, vol. 52, no 7, p. 5799-5808. Available at: https://arxiv.org/abs/2012.10802

GEIGER, Andreas; LENZ, Philip; URTASUN, Raquel. Are we ready for autonomous driving? the kitti vision benchmark suite. In: 2012 IEEE conference on computer vision and pattern recognition. IEEE, 2012. p. 3354-3361. Available at https://ieeexplore.ieee.org/abstract/document/6248074

HIGO, Kazuki, et al. TerraFusion: Joint Generation of Terrain Geometry and Texture Using Latent Diffusion Models. arXiv preprint arXiv:2505.04050, 2025. Available at: https://arxiv.org/abs/2505.04050

HUANG, Zixuan, et al. Spar3d: Stable point-aware reconstruction of 3d objects from single images. En Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. p. 16860-16870. Available at: https://arxiv.org/abs/2501.04689

LI, Zhengqi; SNAVELY, Noah. Megadepth: Learning single-view depth prediction from internet photos. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2018. p. 2041-2050. Available at https://arxiv.org/abs/1804.00607

RANFTL, René et al. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. IEEE transactions on pattern analysis and machine intelligence, v. 44, n. 3, p. 1623-1637, 2020. Available at https://arxiv.org/abs/1907.01341

RANFTL, René; BOCHKOVSKIY, Alexey; KOLTUN, Vladlen. Vision transformers for dense prediction. In: Proceedings of the IEEE/CVF international conference on computer vision. 2021. p. 12179-12188. Available at https://arxiv.org/abs/2103.13413

TANG, Xiang; LI, Ruotong; FAN, Xiaopeng. Recent Advances in 3D Object and Scene Generation: A Survey. arXiv preprint arXiv:2504.11734, 2025. Available at: https://arxiv.org/abs/2504.11734

WANG, Zhengren. 3d representation methods: A survey. arXiv preprint arXiv:2410.06475, 2024. Available at: https://arxiv.org/abs/2410.06475

YURDAKUL, Mustafa; TASDEMIR, Şakir. An enhanced yolov8 model for real-time and accurate pothole detection and measurement. arXiv preprint arXiv:2505.04207, 2025. Available at https://arxiv.org/abs/2505.04207
