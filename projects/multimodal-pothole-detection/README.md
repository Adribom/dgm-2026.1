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

## Objective

## Methodology

> ### 1. Hypothesis 
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

To ensure the project's feasibility within a two-month timeframe and limited computational resources, our methodology will incorporate two key strategies:
- **Low-Rank Adaptation (LoRA) Fine-Tuning**: Instead of training a 3D diffusion model from scratch, we will freeze the pre-trained weights of the base model and apply LoRA fine-tuning. This will allow the network to act as a conditional 2D-to-3D generator without catastrophic forgetting or the need for extensive GPU clusters.
- **Sparse Point Cloud Generation**: Inspired by the [SPAR3D framework](https://spar3d.github.io/), we will condition the diffusion model to initially generate a sparse point cloud (e.g., 512 or 2048 points). Offloading the geometric uncertainty of the occluded pothole depth to a lightweight, sparse probabilistic generation significantly reduces inference time while maintaining topological coherence.

Because diffusion models can be computationally intensive, we also propose another plan that utilizes a regression-based pipeline. This method utilizes a Monocular Depth Estimation network (such as [MiDaS](https://pytorch.org/hub/intelisl_midas_v2/) or [DPT](https://huggingface.co/docs/transformers/model_doc/dpt)) to predict 2.5D depth maps from single RGB images. Following this, we will perform an algebraic back-projection, leveraging the camera's intrinsic parameters (focal lengths and optical center), to translate these depth values into a 3D point cloud. This approach could struggle to accurately infer geometry in heavily occluded or shadowed regions, such as the untextured bottom of a concave pothole, but it offers a more computationally efficient alternative.

To implement the aforementioned pipelines within the project's timeframe, we will utilize the following frameworks and libraries:
- Python & PyTorch: The core programming language and deep learning framework for model training, fine-tuning, and tensor operations.
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index): The primary library for instantiating, manipulating, and applying LoRA fine-tuning to the pre-trained 3D diffusion models.
- [Open3D](https://github.com/isl-org/Open3D): Python library for 3D point cloud processing and visualization.
- [PyTorch Hub](https://pytorch.org/hub/) / [HuggingFace Transformers](https://huggingface.co/docs/transformers/index): For rapidly deploying the alternative regression-based networks (MiDaS/DPT) without needing to train them from scratch.

We expect the Generative Modeling pipeline to successfully infer the occluded bottom of road potholes, providing a structurally coherent and probabilistically accurate 3D point cloud that overcomes the limitations of shadowed, textureless cavities. This high-fidelity representation will allow for later precise volumetric calculations of the road damage. In contrast, while we expect the alternative Regression-based pipeline to process images significantly faster and with lower memory footprint, it will likely exhibit smoothed, inaccurate topologies inside deep cavities, demonstrating the fundamental trade-off between computational efficiency and geometric fidelity.

Evaluating generated 3D point clouds of strictly concave surfaces requires specialized metrics. Standard metrics like the Chamfer Distance (CD) often mask clustered points and fail to capture geometric fidelity on jagged edges. To rigorously evaluate our models against the high-precision gypsum mold ground truth, we will employ the following state-of-the-art metrics:
- **Surface Normal Concordance (SNC)**: Instead of merely comparing Euclidean coordinates, SNC measures surface similarity by comparing estimated point normals. This is crucial for potholes, as it evaluates whether the model accurately captured the steep, jagged slopes of the crater rather than just outputting a flat, smoothed depression.
- **Density-Aware Chamfer Distance (DCD)**: An improvement over standard CD that penalizes points clustering unevenly, ensuring a homogeneous spatial distribution of the generated point cloud.
- **Root Mean Square Error (RMSE)**: Used strictly to quantify the absolute depth deviation between the generated topology and the ground truth, validating the viability of the model for real-world volume estimation tasks.


### Datasets and Evolution

> List the datasets used in the project.
> For each dataset, include a mini-table in the model below and then provide details on how it was analyzed/used, as in the example below.

> |Dataset| Source | Descriptive Summary|
> |--|--|--|
> |**PothRGDB** | Research Repository | Provides 1,000 paired RGB and depth (2.5D) images with YOLO annotations captured using an Intel RealSense camera as the primary dataset.|
> |**Rui Fan's Stereo Pothole** | Research Repository | Contains 79 pothole instances with high-precision 3D ground truth obtained from laser-scanned gypsum molds|

> Provide a description of what you concluded about this dataset. Suggested guiding questions or information to include:
> - What is the dataset format, size, type of annotation?
> - What transformations and preprocessing were done? Cleaning, re-annotation, etc.
> - Include a summary with descriptive statistics of the dataset(s).
> - Use tables and/or charts to describe the main aspects of the dataset that are relevant to the project.

### Workflow

> Use a tool that allows you to design the workflow and save it as an image (e.g., Draw.io). Insert the image in this section.
> You may choose to use a workflow manager (Sacred, Pachyderm, etc.), in which case use the manager to generate a diagram for you.
> Remember that the goal of drawing the workflow is to help anyone who wishes to reproduce your experiments.

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
