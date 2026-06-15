# Geracao de Contrafactuais Explicaveis para Pneumonia em Imagens de Raio-X de Torax

# Explainable Counterfactual Generation for Pneumonia in Chest X-ray Images

# Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the **first semester of 2026 (2026.1)**, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

| Name | RA | Specialization |
|--|--|--|
| Maria Fernanda Bosco | 183544 | Statistics |
| Gabriel Carvalho Freitas | 155421 | Statistics |
| Gyovana Mayara Moriyama | 216190 | Computer Science |

---


# Presentation slides

[E3 presentation](https://docs.google.com/presentation/d/1PmMhhpT-1ni9IrADmvGIo_p7RAnGzK0hgI9P65Vd-eE/edit?usp=sharing)

---

# Project Summary Description

# Abstract

> Update

This project investigates counterfactual generation for pneumonia in chest X-rays using the NIH Chest X-ray dataset. We built patient-level splits, trained classifier baselines, and implemented two generative models: a metadata-conditioned CVAE and an unpaired CycleGAN. Classifier baselines showed limited discriminative performance under severe class imbalance, with the best test AUC reaching 0.6668. The CVAE preserved structure well (SSIM = 0.8190) but had limited realism (FID = 136.5358), while CycleGAN produced sharper translations with lower mean FID (115.34). Results suggest counterfactual generation is feasible, but classifier validity and clinical plausibility remain open challenges.

# Problem Description / Motivation

Deep learning models for medical imaging are often limited by data scarcity and class imbalance, especially for less frequent pathological cases such as pneumonia in chest X-rays. In clinical applications, this limitation is especially relevant because models trained on imbalanced data may learn the dominant healthy class more effectively than the disease patterns of interest.

Chest X-ray analysis for pneumonia detection is also challenging because high classification performance alone is not enough for clinical trust. Medical users need to understand which image regions influenced a model decision and whether the model is relying on plausible radiological cues. Counterfactual generation addresses both needs by creating images that preserve patient anatomy while changing the disease condition, making it possible to inspect what the model changes when translating between healthy and pneumonia domains.

# Objective

The general objective is to develop a **generative framework for counterfactual image generation and classifier-grounded evaluation** in chest X-rays.

Instead of generating images from random noise alone, the problem is formulated as a **domain translation task** between:

- **Healthy chest X-rays**
- **Pneumonia chest X-rays**

The central questions are:

- **What would a healthy patient look like if they had pneumonia?**
- **Does the generated counterfactual change the classifier prediction in the intended direction?**

Specific objectives are:

1. Build a clean and reproducible preprocessing pipeline for the NIH Chest X-ray dataset.
2. Select healthy and pneumonia-only samples to avoid confounding labels from multiple diseases.
3. Use TorchXRayVision lung segmentation masks to guide the models toward anatomically relevant lung regions.
4. Train conditional generative models that uses image labels, patient metadata, and lung-mask information.
5. Generate counterfactual chest X-rays by changing the target condition.
6. Evaluate counterfactual validity using classifier flip rate and confidence change, replacing the previously planned visual attribution analysis.

Expected model outputs are:

- Counterfactual pneumonia and healthy chest X-ray images.
- Lung-aware reconstructions and generated images that preserve patient anatomy.
- Classifier-facing metrics describing whether counterfactuals change predictions as intended.

# Methodology

The methodology combines exploratory data analysis, metadata cleaning, patient-level data splitting, conditional generative modeling, and qualitative counterfactual inspection. The implemented models are a CVAE, a CycleGAN and classifier-based evaluation.

## Dataset

| Dataset | Web Address | Descriptive Summary |
| ------------- | ----------------- | ----------------------------------------------------- |
| NIH Chest X-rays | https://www.kaggle.com/datasets/nih-chest-xrays/data | Public chest X-ray dataset with 112,120 frontal X-ray images from 30,805 patients. Labels were obtained by text-mining associated radiology reports and are suitable for weakly supervised learning. |

The NIH Chest X-ray dataset contains 12 image folders and a `Data_Entry_2017.csv` metadata file. The metadata includes image index, finding labels, follow-up number, patient ID, patient age, patient gender, view position, original image dimensions, and pixel spacing.

The `Finding Labels` column may contain `No Finding`, a single disease label, or multiple disease labels separated by `|`, such as `Mass|Pneumonia`. To reduce ambiguity in this project, only two groups were used:

- `No Finding`: healthy controls.
- `Pneumonia`: images annotated with pneumonia only, excluding images with additional findings.

The full dataset contains 836 distinct label combinations. Among the selected records, the project identified 60,361 healthy X-rays before outlier removal and 322 pneumonia-only X-rays. This creates a severe class imbalance, which is one of the core motivations for studying generative augmentation.

### Metadata

#### Pneumonia patients

There are 322 occurrences of pneumonia-only X-rays. Patient ages range from 3 to 87 years old, and no age outliers were detected using the IQR rule.

![Pneumonia Patients Age Distribution](images/pneumonia_patients_age_distrib.png)

For pneumonia-only patients, male patients are more frequent than female patients.

![Pneumonia Patients Gender Distribution](images/pneumonia_patients_gender_distrib.png)

![Pneumonia Patients Age by Gender Distribution](images/pneumonia_age_by_gender.png)

**Data cleaning**

- No duplicated rows were found.
- No age outliers were found for pneumonia-only patients.

#### Healthy patients

There are 60,361 occurrences of healthy X-rays before cleaning. The raw age range goes from 1 to 413 years old, indicating metadata errors that require outlier removal.

![Healthy Patients Age Distribution](images/healthy_patients_age_distrib.png)

As in the pneumonia-only group, male patients are more frequent than female patients.

![Healthy Patients Gender Distribution](images/healthy_patients_gender_distrib.png)

**Data cleaning**

- No duplicated rows were found.
- Eight age outliers were removed using the IQR method.
- After cleaning, 60,353 healthy images remained.

![Healthy Patients Age Distribution After Outlier Removal](images/healthy_age_outlier_removal.png)

![Healthy Patients Age by Gender Distribution](images/healthy_age_by_gender.png)

#### Key Findings from EDA

1. **Class imbalance**: the selected data contains 60,353 healthy images and only 322 pneumonia-only images after cleaning.
2. **Age distribution**: pneumonia-only patients are distributed across children, adults, and older adults, with concentration in middle-age ranges.
3. **Gender distribution**: both selected groups contain more male than female patients.
4. **Data quality**: no duplicate metadata rows were found, but age cleaning was necessary for healthy cases.
5. **Image standardization**: all loaded images were converted to grayscale tensors and resized to 128 x 128 pixels for model training.

### Images

The original images have different spatial resolutions and are too large for efficient experimentation with the available training setup. For this reason, all images used by the CVAE pipeline are resized to 128 x 128 pixels and normalized to the range `[0, 1]`.

## Preprocessing

![Preprocessing workflow](images/preprocessing.png)

The preprocessing pipeline implemented in `utils/preprocessing.py`, `utils/dataset.py`, and `utils/xrv_lung_segmentation.py` follows these steps:

1. Download the NIH Chest X-ray dataset using `kagglehub`.
2. Load `Data_Entry_2017.csv`.
3. Remove duplicated rows and remove patient-age outliers using the interquartile range method.
4. Select pneumonia-only and healthy records.
5. Normalize age using either Min-Max scaling or standardization and encode gender as a numerical feature. Also, assign binary labels: healthy = 0 and pneumonia = 1.
6. Split the data by patient into training, validation, and test sets with proportions 70%, 15%, and 15%.
7. Load and resize images to 128 x 128 pixels.
8. Generate lung masks with the TorchXRayVision anatomical PSPNet segmenter, using the left-lung and right-lung outputs to create a binary mask.
9. Keep the grayscale image as the first channel and concatenate the lung mask as a second channel (`add_lung_mask_channel=True`). This lets the model learn from the full image while making the lung region explicit to the loss function.
10. Convert images, labels, masks, and metadata into PyTorch-compatible tensors.
11. Build PyTorch datasets and dataloaders.

The split is performed at patient level, so the same patient cannot appear in more than one split. This avoids patient leakage between training and evaluation sets. Lung masks are generated after the split for each subset and are used only as model inputs/loss guidance, not as labels for diagnosis.

## Models

### 1. Generative Models

The problem is formulated as a **domain translation task** rather than unconditional generation. 

Two generative approached are being explored:

#### 1.1 Conditional Variational Autoencoder (CVAE)

The Conditional Variational Autoencoder (CVAE) [11] is used as the first counterfactual generation baseline. Unlike an unconditional VAE, the model receives both the chest X-ray image and auxiliary conditioning variables, allowing the decoder to reconstruct or generate an image under a specified clinical condition.

In this project, the CVAE models the conditional distribution:

$$
p(x \mid z, y, m)
$$

Where:

- $x$: chest X-ray image with an additional lung-mask channel.
- $z$: latent representation.
- $y$: class condition, healthy or pneumonia.
- $m$: patient metadata, represented by normalized age and encoded gender.

Counterfactual generation is performed by encoding an input image into the latent space, replacing the original class condition with the target class condition, and decoding the same latent representation under the new label. This allows the model to answer questions such as: what would this healthy chest X-ray look like if it were conditioned as pneumonia?

Two CVAE variants exist in the repository:

- `models/cvae.py`: fully connected CVAE baseline.
- `models/cvae_cnn.py`: convolutional CVAE used in the latest experiment.

The CNN-based CVAE uses four convolutional encoder blocks and a transposed-convolution decoder. The final model uses two input channels: the grayscale X-ray and the TorchXRayVision lung mask. The decoder outputs a single reconstructed or counterfactual X-ray channel. The latent dimension is 64, and conditioning combines the binary disease label, normalized age, and a learned gender embedding.

### Implemented CVAE workflow

The following diagram summarizes the actual CVAE pipeline used in this project, from the input image and conditioning variables to the reconstructed or counterfactual output.

![CVAE-Workflow](images/cvae.png)

At inference time, the same encoder is used to obtain the latent representation, while the target label is changed before decoding, which produces the counterfactual image.

The decoder also uses skip connections from the encoder. These connections pass intermediate spatial features from the encoder to the corresponding decoder stages, helping preserve chest anatomy, lung shape, rib structure, and other patient-specific details that can be lost when all information is compressed through the latent vector alone. FiLM-style conditioning layers modulate decoder features with the label and metadata condition, making the target class explicit during reconstruction and counterfactual generation.

**Training objective and loss**

The CVAE model outputs:

* the reconstructed image $\hat{x}$,
* the latent mean $\mu$,
* and the latent log-variance $\log\sigma^2$.

During training, the model minimizes a loss composed of:

1. a global reconstruction loss between the original image channel and the reconstructed image;
2. a lung-region reconstruction loss computed inside the binary lung mask;
3. an outside-lung reconstruction loss computed outside the mask, used to discourage unnecessary background changes;
4. a KL-divergence term, which regularizes the latent space.

The total loss is defined as:

$$
\mathcal{L}_{CVAE} =
\mathcal{L}_{rec} +
\lambda_{lung}\mathcal{L}_{lung} +
\lambda_{outside}\mathcal{L}_{outside} +
\beta D_{KL}
$$

The reconstruction terms use pixel-level losses on grayscale images normalized to `[0, 1]`. The lung mask lets the model distinguish errors inside the clinically relevant lung fields from changes in the surrounding background. The configuration used was: $\beta = 0.005$, $\lambda_{lung} = 0.2$, and $\lambda_{outside} = 3.0$.

The KL-divergence term is computed from $\mu$ and $\log\sigma^2$:

$$
D_{KL} =
-\frac{1}{2}
\sum (1 + \log\sigma^2 - \mu^2 - \sigma^2)
$$

It is normalized by the batch size so that its scale is more comparable across batches. The small KL weight keeps the model focused on reconstruction quality while still maintaining a structured latent space. This is useful for counterfactual generation because it helps preserve the overall anatomy of the chest X-ray while allowing condition-related changes to be generated.

**Advantages**

- Stable training compared with adversarial models.
- Direct conditioning on label and metadata.
- Lung-mask guidance through the input and loss function.
- Skip connections that improve anatomical preservation.
- Natural support for controlled counterfactual generation.

**Current limitations**

- Reconstructions are still smoother than the original X-rays, which is common in VAE-based models.
- The strong class imbalance can bias generated images toward healthy-looking reconstructions.
- The generated counterfactuals still require classifier-based validity evaluation through flip rate and confidence change.

#### 1.2 Cycle-Consistent GAN (CycleGAN)

CycleGAN (Cycle-Consistent Generative Adversarial Network) is an unpaired image-to-image translation framework introduced by Zhu et al. (2017). It learns bidirectional mappings between two image domains without requiring paired training examples. In this project, **healthy chest X-rays** (domain H) and **pneumonia chest X-rays** (domain P). 

The framework comprises four networks trained simultaneously:

- **G_H2P**: Generator that translates Healthy → Pneumonia  
- **G_P2H**: Generator that translates Pneumonia → Healthy  
- **D_H**: Discriminator for the Healthy domain  
- **D_P**: Discriminator for the Pneumonia domain  

The total generator loss combines three components:

$$
\mathcal{L}_G = \mathcal{L}_{\text{GAN}} + \lambda_{\text{cycle}} \cdot \mathcal{L}_{\text{cycle}} + \lambda_{\text{identity}} \cdot \mathcal{L}_{\text{identity}}
$$

- **GAN loss** (LSGAN / MSE-based): encourages generators to produce images indistinguishable from the target domain.  
- **Cycle consistency loss** (L1): enforces that translating an image to the other domain and back recovers the original — $G_{P2H}(G_{H2P}(x_H)) \approx x_H$ and vice versa. This is the key constraint that preserves anatomical structure.  
- **Identity loss** (L1): regularizes each generator when fed images already in the target domain, helping preserve color and texture properties.

Each discriminator uses a **70×70 PatchGAN** architecture, which classifies overlapping image patches as real or fake rather than the whole image, encouraging high-frequency sharpness.

Cycle consistency can help preserve anatomical structure while modifying disease-related regions. Its expected advantage is sharper image generation, but its main challenge is training instability and the risk of adding unrealistic artifacts.

**Advantages:**
- Works with unpaired data
- Produces sharper and more realistic images

### 2. Classification Models

A classification model is a central component of this project, as it provides the basis for evaluating both data augmentation and explainability through counterfactuals.

The classifier is trained to perform a binary classification task:

- Input: Chest X-ray image
- Output: Pneumonia vs. Healthy

The classifier has two roles:

- **Performance benchmark**: measure whether synthetic images improve pneumonia classification.
- **Explainability anchor**: verify whether counterfactual images change the classifier prediction as intended.

## Explainability Strategy

The project no longer includes a separate visual attribution component. Instead, counterfactual usefulness is evaluated through classifier-facing counterfactual validity metrics:

- **Flip rate**: measures the proportion of generated counterfactuals that change the classifier prediction to the intended target class. For example, a healthy-to-pneumonia counterfactual is successful if the classifier prediction changes from healthy to pneumonia.

$$
FlipRate=
\frac{\#\text{successful class changes}}
{\#\text{counterfactuals}}
$$

- **Confidence change**: measures how much the classifier's confidence in the target class increases after generating the counterfactual.

$$
\Delta=
P(y_{target}|x_{cf})-
P(y_{target}|x)
$$

These metrics directly test whether the generated image changes the classifier decision, which is the practical role counterfactuals play in this project. Difference maps and change heatmaps may still be used as qualitative visual summaries of where the image changed, but they are not treated as a separate explainability method.

## Tools

| Tool | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| torchvision | Tensor utilities and image saving |
| pandas | Metadata loading and cleaning |
| NumPy | Numerical operations |
| PIL | Image loading and resizing |
| Matplotlib / Seaborn | Exploratory data analysis and visualization |
| scikit-learn | Planned classification metrics and evaluation utilities |
| Jupyter Notebook | Exploratory analysis and experimentation |
| kagglehub | Dataset download from Kaggle |

## Evaluation Methodology

The objectives are assessed through a combination of classifier performance, image-generation metrics, and classifier-facing counterfactual validity metrics. The project will be considered successful if the generated counterfactuals preserve patient anatomy, remain visually plausible, and change the classifier prediction in the intended direction.

The classification objective is assessed with ROC-AUC, accuracy, confusion matrices, and training/validation curves. ROC-AUC is the primary metric because the dataset is severely imbalanced. This objective is met only if the classifier achieves stable validation behavior and test performance clearly above chance, especially for pneumonia cases. If classifier performance is weak, flip rate and confidence change must be interpreted cautiously, because the classifier is the evaluation oracle.

The image-generation objective is assessed with SSIM, FID, and visual inspection. SSIM indicates whether counterfactuals preserve anatomical structure, while FID indicates whether generated images follow the distribution of real chest X-rays. This objective is met if generated images remain anatomically close to the input, avoid obvious artifacts, and improve realism across model variants.

The counterfactual validity objective replaces the previous visual-attribution-based explainability plan. It is assessed with flip rate and confidence change: generated images should flip the classifier prediction to the target class and move the classifier confidence in the expected direction. Change heatmaps are kept only as qualitative visual aids for inspecting image differences.

The data augmentation objective will be assessed by retraining the classifier with and without generated images. It is met if synthetic counterfactuals improve ROC-AUC or minority-class recall without increasing overfitting or introducing clinically implausible artifacts.

The evaluation will consider three aspects:

### 6.1 Classification Performance:
- Accuracy
- ROC-AUC

### 6.2 Image Generation Quality:
- SSIM (Structural Similarity Index): SSIM evaluates structural similarity between original and counterfactual images, measuring whether anatomical consistency is preserved during transformation. (high is better)
- FID (Fréchet Inception Distance): FID evaluates the realism of generated images by measuring the distance between the feature distributions of real and synthetic samples, indicating how closely the generated images resemble real chest X-rays. (low is better)
- Visual inspection of reconstructions and counterfactual pairs, with attention to lung-field preservation and artifacts.

### 6.3 Counterfactual Validity:
- Flip rate: proportion of counterfactuals that change the classifier prediction to the target class. (high is better)
- Confidence change: difference between the classifier target-class confidence before and after counterfactual generation. The expected direction is an increase for the target class and a decrease for the original class.

# Workflow

The diagram below summarizes the actual reproducibility flow implemented in this project: which data enters each stage, how it is transformed, and which outputs are produced for the experiments.

![Workflow](images/workflow-d3-vertical.png)

This diagram makes explicit that the experiment starts from the raw NIH dataset, passes through cleaning and patient-level splitting, then feeds three branches: classifier evaluation, CVAE counterfactual generation, and CycleGAN translation. The outputs of those branches are then compared using the same downstream classifier and image-quality metrics.

The original project schedule is also available:

![Schedule](images/schedule.png)

# Experiments, Results, and Discussion of Results

>In the final project submission (D3), this section should list the main results obtained (not necessarily all), which best represent the fulfillment of the project objectives.
> The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.
> It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."
> What is expected from this section is that it presents and discusses only the most relevant results, highlighting the strengths and/or limitations of the methodology, emphasizing aspects of performance, and containing content that can be classified as organized, didactic, and reproducible sharing of knowledge relevant to the community.

## Experiment 1: Classifier Baseline

The classifier is the downstream oracle used to evaluate the counterfactual validity of the generative models (CVAE and CycleGAN): a generated *pneumonia* image should flip the classifier's prediction with respect to the source *healthy* image, and vice-versa. The classifier itself is a standard binary supervised model that takes a single-channel chest X-ray as input and outputs a logit for the pneumonia class.

As the main ideia is that the classifier will be the way to evaluate the generative models, it has to be good classifier, so some test were experimented for it's development. Four families of models were trained and compared on the same train/val/test splits of the NIH Chest X-ray dataset:

- **SimpleCNN (baseline)** — a 5-block convolutional network trained from scratch.
- **SimpleCNN + augmentation** — same architecture, trained with light geometric augmentations.
- **FrozenResNet-18** — ImageNet-pretrained ResNet-18 with the backbone frozen and a new binary head.
- **FrozenDenseNet-121** — ImageNet-pretrained DenseNet-121 with the backbone frozen and a new binary head (CheXNet-style architecture).

All models output a single raw logit and are trained with `BCEWithLogitsLoss` using an attenuated `pos_weight = sqrt(n_neg / n_pos)` to address severe class imbalance (~0.5% pneumonia in train). The square-root attenuation softens the standard `n_neg / n_pos` weight (≈190) into a more stable value (~13.8) that still upweights positives without dominating the gradient.

### 1.1 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size (cache) | 128 × 128 (grayscale, normalized to [0, 1]) |
| Train / Val / Test | 42,452 / 8,798 / 9,425 |
| Pneumonia rate | train 0.525% · val 0.557% · test 0.531% |
| Batch size | 32 (pretrained) / 64 (SimpleCNN) |
| Learning rate | 3 × 10⁻⁴ |
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999), weight_decay = 1 × 10⁻⁴ |
| Loss | BCEWithLogits with `pos_weight = sqrt(n_neg / n_pos)` ≈ 13.76 |
| Scheduler | ReduceLROnPlateau on val AUC (factor 0.5, patience 3) |
| Epochs | up to 100, early-stop patience 10 (on val AUC) |

**SimpleCNN architecture** — 5 convolutional blocks (Conv3×3 → BN → ReLU → MaxPool 2) with channels `1 → 32 → 64 → 128 → 256 → 512`, followed by AdaptiveAvgPool2d(1), Dropout(0.3), `Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1)`.

**Frozen backbones** — torchvision ImageNet-pretrained ResNet-18 and DenseNet-121 with the entire backbone frozen and a new `Linear(features → 1)` head. Grayscale inputs are repeated across 3 channels and standardized with ImageNet statistics before the backbone.

**Training data augmentation** (train split only, when enabled):
- Random rotation (±10°)
- Random horizontal flip (p = 0.5)
- Random resized crop (scale = [0.95, 1.0])

**Evaluation protocol** — at the end of training the best checkpoint (max val AUC) is loaded. An optimal decision threshold is selected on the validation set via Youden's J statistic (`argmax(TPR − FPR)`), then applied to the test set. AUC-ROC is threshold-independent and is the primary metric; accuracy and confusion matrix are reported at the Youden threshold.

### 1.2 Training

Five experiments were run, each corresponding to one notebook under [notebooks/](notebooks/):

| Notebook | Goal |
|---|---|
| `1.0-train_baseline_classifier_cnn` | Establish a from-scratch baseline with no augmentation. |
| `1.2-train_augmented_classifier_cnn` | Measure the effect of light geometric augmentation on the same SimpleCNN. |
| `1.1-train_pretrained_resnet18` | Linear probe on ImageNet ResNet-18 (head-only training). |
| `1.3-train_pretrained_densenet121` | Linear probe on ImageNet DenseNet-121 (CheXNet-style head-only). |

The training/validation AUC and loss were tracked at every epoch. Early stopping triggered between epochs 12 and 20 for every model, well before the 100-epoch budget — none of the configurations were limited by training time.

### 1.3 Results

#### 1.3.1 SimpleCNN — Baseline and Augmented Variants

**Baseline (no augmentation)** — [results/baseline_cnn_20260516-164040/](results/baseline_cnn_20260516-164040/)

![SimpleCNN baseline training curves](results/baseline_cnn_20260516-164040/train_curves.png)

The SimpleCNN baseline was the only experiment in which a `pos_weight` scale of `n_neg / n_pos` (≈190). was applied. The model reached a peak validation AUC of **0.6411** at epoch 8, after which performance deteriorated. Training AUC continued climbing past 0.90 while training loss fell from 0.27 to 0.18. However, validation loss began rising after epoch 8, indicating overfitting. Early stopping triggered at epoch 18. On the test set, the model achieved AUC = **0.6668** and accuracy = **0.9947** — the latter being a misleading metric: with only 0.5% positive samples, predicting "healthy" for every instance already yields 99.5% accuracy. Given the overfitting observed, one likely contributing factor was the excessively large `pos_weight`, and subsequent experiments adopted an attenuated value of `pos_weight = sqrt(n_neg / n_pos)`.

**Augmented variant** — [results/augmented_cnn_20260517-172811/](results/augmented_cnn_20260517-172811/)

![Augmentation examples](results/augmented_cnn_20260517-172811/augmentation_examples.png)

![SimpleCNN augmented training curves](results/augmented_cnn_20260517-172811/training_curves.png)

Adding light geometric augmentation (rotation ±10°, horizontal flip, random resized crop 0.95–1.0) brought peak val AUC to **0.6284** (epoch 2) with early stopping at epoch 12. Test AUC = **0.6383**, test accuracy = **0.6857**. The augmentation slightly reduced overfitting (train and val curves stay closer together) but did **not** improve the AUC over the no-augmentation baseline. The drop in accuracy versus the baseline is expected: a different (higher) effective threshold from Youden's J trades many true negatives for slightly higher recall, but at this positive rate the trade is not favourable on raw accuracy.

#### 1.3.2 Frozen Pretrained Backbones

**FrozenResNet-18** — [results/resnet18_20260518-110522/](results/resnet18_20260518-110522/)

![ResNet18 ROC](results/resnet18_20260518-110522/training_curves.png)

The third baseline introduced a pretrained approach using ResNet18 as a frozen feature extractor, with only the classification head trained from scratch. Over 20 epochs — with no early stopping triggered — training loss decreased modestly from 0.27 to 0.24, while training AUC improved gradually but with considerable noise, oscillating between 0.68 and 0.72 in the later epochs without clear convergence. Validation AUC showed a slow, steady improvement throughout training, rising from 0.52 at epoch 1 to a peak of 0.5758 at epoch 19, with no sign of the sharp deterioration observed in the SimpleCNN baseline. Validation loss remained highly volatile across all epochs, exhibiting no consistent upward or downward trend, which suggests the frozen backbone produced representations that were not well-suited to this domain without fine-tuning.

The final metrics were: best validation AUC = 0.5758, test AUC = 0.5718, and test accuracy = 0.7211. Training AUC plateaued around 0.71, indicating that the classification head extracted as much discriminative signal as the frozen ImageNet features could provide

**FrozenDenseNet-121** — [results/densenet121_20260518-094701/](results/densenet121_20260518-094701/)

![DenseNet121 training curves](results/densenet121_20260518-094701/training_curves.png)

The fourth baseline adopted DenseNet-121 as a frozen feature extractor, following the same protocol as the ResNet18 baseline. Training dynamics showed a clear and consistent improvement: training loss fell steadily from 0.27 to 0.18, and training AUC climbed from 0.60 to 0.88 over 12 epochs — a substantially stronger training signal than observed in any previous baseline. However, validation behavior told the opposite story. Validation loss rose monotonically from 0.46 at epoch 1 to 0.88 at epoch 12, and validation AUC peaked early at 0.5728 (epoch 2) before declining and oscillating around 0.56–0.57 for the remainder of training, indicating severe overfitting from the first epochs onward. Early stopping triggered at epoch 13.

The final metrics were: best validation AUC = 0.5406 (epoch 3), test AUC = 0.6207, and test accuracy = 0.7949. Despite DenseNet-121 being the backbone behind CheXNet and considerably deeper than ResNet18, the frozen-backbone setup did not outperform the from-scratch SimpleCNN. The rapidly diverging train and validation curves suggest that the dense feature representations, while powerful in the ImageNet domain, do not generalize to chest X-ray pathology without fine-tuning — and that the classification head, trained alone, quickly memorized training-set patterns rather than learning transferable discriminative features.

#### 1.3.3 Consolidated Test-Set Results

| Notebook | Model | Test AUC ↑ | Test Acc (Youden) | Best Val AUC | Stopped at |
|---|---|---|---|---|---|
| 1.0 | SimpleCNN baseline (no aug) | 0.6668 | 0.9947* | 0.6411 | epoch 18 |
| 1.2 | SimpleCNN + augmentation | 0.6383 | 0.6857 | 0.6284 | epoch 12 |
| 1.1 | FrozenResNet-18 (linear probe) | 0.5718 | 0.7211 | 0.5758 | epoch 20 |
| 1.3 | FrozenDenseNet-121 (linear probe) | 0.6207 | 0.7949 | 0.5406 | epoch 13 |

## Experiment 2: CVAE Training

### 2.1 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 128 x 128 |
| Input channels | 2: grayscale X-ray + lung mask |
| Output channels | 1: reconstructed/generated X-ray |
| Batch size | 16 |
| Planned epochs | 300 |
| Completed epochs in saved run | 94 |
| Best epoch checkpoint | 78 |
| Learning rate | 3 x 10^-4 |
| Optimizer | Adam |
| Latent dimension | 64 |
| Conditions | binary disease label, normalized age, and gender embedding |
| KL weight $\beta$ | 0.005 with warm-up schedule |
| Lung reconstruction weight $\lambda_{lung}$ | 0.2 |
| Outside-lung reconstruction weight $\lambda_{outside}$ | 3.0 |
| Lung segmentation | TorchXRayVision anatomical PSPNet mask |
| Device | CUDA |

The final CVAE experiment uses the convolutional implementation in `models/cvae_cnn.py`. The training input concatenates the resized grayscale X-ray and its lung mask, while the decoder generates only the image channel. Skip connections pass encoder features into the decoder to preserve spatial structure, and FiLM conditioning injects label and metadata information into decoder blocks.

### 2.2 Training Loss Behavior

![CVAE Loss behavior](training-results/cvae/loss-behavior.png)

The final experiment contains 94 completed epochs. Training loss decreased from **0.3880** at the first saved epoch to **0.0170** at the last saved epoch. Validation loss decreased from **0.1562** to **0.0126**, and the validation image score remained high near the end of training, reaching approximately **0.9971** in the final logged epoch.

This behavior indicates that the CVAE learned a stable reconstruction mapping for the lung-mask-conditioned inputs. The low final losses and high validation image score suggest that the skip-connected architecture preserves global anatomy well. At the same time, the generated outputs remain smoother than the original X-rays, which is expected for a VAE-style model and limits fine radiological texture.

Example reconstruction outputs:

![CVAE reconstruction epoch 0](training-results/cvae/results/reconstruction_0.png)

![CVAE reconstruction epoch 93](training-results/cvae/results/reconstruction_93.png)

### 2.3 Counterfactual Generation

After training, counterfactual images were generated for the complete test set by flipping the input class condition:

- Healthy $\rightarrow$ Pneumonia.
- Pneumonia $\rightarrow$ Healthy.

The latent representation was extracted from the original image and decoded using the opposite disease condition. During generation, the lung-aware setup encourages the model to preserve patient-specific anatomy and avoid unnecessary changes outside the lung fields.

In total, **9,176 original images** and **9,176 counterfactual images** were evaluated.

### 2.4 Qualitative Evaluation

**Counterfactual image generation examples**

> Generate image

The figure above shows 4 example pairs for each translation direction. Blue borders denote real (input) images while red borders denote generated counteerfactual (output) images.

The generated examples show that the CVAE preserves the main anatomical layout of the input X-rays, including the lung fields, rib cage, and cardiac silhouette. The changes introduced by the CVAE are subtle, and the lung-mask channel helps make the lung region explicit during training.

The generated outputs are smoother than the original images. This suggests that the model captures global structure more strongly than fine radiological texture, losing some sharpness in bones, borders, and small artifacts. This smoothness is a known limitation of VAE-based image generation.

**Counterfactual change heatmaps**

![CVAE Change Heatmap](training-results/cvae/results/change_heatmaps/cvae_change_heatmap_002.png)

The change heatmaps show absolute pixel-level differences between each original image and its generated counterfactual. Brighter regions indicate stronger image changes. These heatmaps are used only as qualitative visual checks; the project no longer treats them as the main explainability strategy.

The changes are not uniformly distributed across the full image, which supports the counterfactual goal of localized modifications rather than arbitrary global shifts. However, the final result show highlighted areas more concentrated on the lung regions but we can see some highlighted regions are diffuse and may include areas outside the most clinically relevant lung regions. With that, we can see that using the lung segmentation algorithm helped, but not fully resolved the issue.

### 2.5 Quantitative Evaluation

| Metric | Value |
|---|---:|
| Number of SSIM pairs | 9,176 |
| Mean SSIM | 0.9957 |
| SSIM standard deviation | 0.0022 |
| Minimum SSIM | 0.9052 |
| Maximum SSIM | 0.9978 |
| Number of counterfactual images | 9,176 |
| Number of reference images | 9,176 |
| FID | 2.7269 |

The mean SSIM of 0.9954 indicates that the generated counterfactuals preserved most of the original image structure, which is important since counterfactual explanations should mainly modify disease-related regions while maintaining anatomical consistency.

The FID score of 2.7269 ...

The counterfactual validity evaluation focuses on classifier behavior:

| Metric | Value |
|---|---|
| Flip rate |  |
| Confidence change | |



## Experiment 3: CycleGAN Training

### 3.1 Training Configuration

| Hyperparameter | Value |
|---|---|
| Image size | 128 × 128 |
| Batch size | 4 |
| Epochs | 200 |
| Learning rate | 2 × 10⁻⁴ |
| Optimizer | Adam (β₁ = 0.5, β₂ = 0.999) |
| λ_cycle | 10.0 |
| λ_identity | 5.0 |
| Generator residual blocks | 6 |
| Image replay buffer size | 50 |
| Device | CUDA |

**Generator architecture** — ResNet-based encoder-decoder:
1. Initial 7×7 convolution → 64 feature maps  
2. Two strided downsampling convolutions (64 → 128 → 256)  
3. Six residual blocks at 256 channels  
4. Two transposed convolutions for upsampling (256 → 128 → 64)  
5. Final 7×7 convolution + Tanh output  
All intermediate layers use InstanceNorm2d.

**Discriminator architecture** — 70×70 PatchGAN: four convolutional layers (C64 → C128 → C256 → C512) with LeakyReLU (slope 0.2) and InstanceNorm2d, followed by a single-channel output map.

**Training data augmentation** (train split only):
- Random horizontal flip (p = 0.5)  
- Random rotation (±5°)  
- Random affine: translation ±2%, scale 0.98–1.02  
- Pixel normalization: mean = 0.5, std = 0.5 (mapping to [−1, 1])  

**Discriminator stabilization**: a replay buffer of size 50 was used for both fake-healthy and fake-pneumonia images, following the original CycleGAN paper. Discriminator loss was scaled by 0.5.

**Validation**: at the end of each epoch, the cycle consistency loss is evaluated on the validation set using both generators in evaluation mode.

### 3.2 Training

#### Baseline CycleGAN (128 × 128, 6 residual blocks)

The training run used the configuration described above as a baseline. The main goals were to:

1. Verify training stability (no mode collapse or vanishing gradients in generator/discriminator losses)  
2. Qualitatively inspect whether the generated images are visually coherent chest X-rays  
3. Assess cycle consistency (whether the reconstructed images recover the input)  

Loss curves (generator total loss, discriminator losses, cycle loss, identity loss, and validation cycle loss) were tracked across all 200 epochs and saved as `outputs/losses/loss_curve.png`. Side-by-side grids of real, translated, and reconstructed images were saved every epoch to `outputs/progress/`.

#### Training Loss Behavior

![CycleGAN Training Loss Curves](models/CycleGAN/outputs/losses/loss_curve.png)

Training ran for 200 epochs on the NIH Chest X-ray training split. All losses are plotted on a logarithmic scale.

The **generator total loss (G_loss)** started high (~5) and decreased steeply during the first ~50 epochs, then continued to decrease slowly, stabilizing around 2.0–2.3 by the end of training, with no signs of mode collapse. The **cycle consistency loss** dropped sharply from ~4 in the first 30 epochs down to approximately 0.8–1.0 by epoch 200, indicating that the generators progressively learned to preserve the anatomical structure of the input image after the round-trip translation. The **identity loss** decreased from ~0.8 to ~0.4, confirming that each generator applies minimal unnecessary changes when given an image already in its target domain. The **discriminator losses (D_H, D_P)** both stabilized around 0.15–0.20 throughout training, indicating that the discriminators remained consistently capable of distinguishing real from generated images. The **validation cycle loss** is noisy due to the small validation set, but closely tracks the training cycle loss, suggesting no overfitting to the training domain.

The most notable behavior is the monotonic upward trend of the GAN loss throughout training. Tha GAN loss is the average of `MSELoss(D_P(G_H2P(real_healthy)), 1)` and `MSELoss(D_H(G_P2H(real_pneumonia)), 1)`. Each term measures how far the discriminator's score on a fake image is from 1. For this loss to increase, the discriminators must be scoring fake images progressively lower — meaning they are getting better at detecting generated images over time, and the generators are not catching up.

The total generator loss combines the GAN term (weight 1), cycle consistency (λ_cycle = 10), and identity (λ_identity = 5) terms. The identity loss was included to preserve color and style consistency, but it may actually be hurting training: in early epochs, the cycle and identity terms dominate the gradient, giving the discriminators time to build a persistent advantage the generators never recover from. Future experiments will test reduced or removed identity weighting to check whether this improves adversarial balance.

### 3.3 Generation and Results

#### Visual Quality of Generated Images

**Counterfactual image generation examples**

![CycleGAN Generation Examples](models/CycleGAN/outputs/img_generation/generation_examples.png)

The figure above shows 4 example pairs for each translation direction, randomly sampled from the test set. Blue borders denote real (input) images; red borders denote generated (output) images.

The generated images maintain the overall chest structure (rib cage, cardiac silhouette, diaphragm position) while introducing subtle changes for both of the translations. Overall, the generation was performed successfully at 128 × 128 resolution. The generated images are visually coherent chest X-rays, though some cases show minor artifacts around the lung borders.

---

**Counterfactual change heatmaps**

![CycleGAN Change Heatmap](models/CycleGAN/outputs/img_generation/change_heatmap.png)

The heatmaps visualize the absolute per-pixel difference between the real and generated images, overlaid on the real image to preserve anatomical context. Brighter colors indicate larger pixel-level changes.

The change heatmaps confirm that the model is not modifying images uniformly. This spatial specificity is an encouraging sign for the counterfactual explainability goal. The model appears to be encoding a representation of the disease rather than introducing arbitrary global texture changes.

However, some heatmap cases show activity near the image borders and outside the lung fields, suggesting the model occasionally makes spurious peripheral changes. This may be a consequence of the small training set size or the 128 × 128 resolution limiting fine spatial encoding.

#### Quantitative Evaluation

The primary quantitative metric for generation quality is the **Fréchet Inception Distance (FID)**, computed between:

- Real pneumonia images vs. generated pneumonia images (G_H2P applied to the healthy test set)  
- Real healthy images vs. generated healthy images (G_P2H applied to the pneumonia test set)  

Lower FID indicates that the generated distribution is closer to the real distribution.

FID was evaluated on the test set using the checkpoint from epoch 200:

| Translation |  FID |
|---|---|
| Healthy → Pneumonia (G_H2P) | 121.09 |
| Pneumonia → Healthy (G_P2H)  | 109.58 |
| **Mean FID**  | **115.34** |

The FID scores are moderately high. The slightly better FID for the P→H direction may reflect that the healthy domain is larger and more varied, providing a richer target distribution. These scores serve as a baseline for comparison in future experiments.

## Discussion

### Baseline

None of the four baselines achieved satisfactory classification performance, and the training dynamics observed across experiments were largely inconsistent with the behavior expected from well-trained models. Training and validation curves frequently diverged, converged prematurely, or failed to stabilize, reflecting the compounding challenges of severe class imbalance, domain shift, and limited model capacity relative to the complexity of the task.

The classification results obtained are insufficient to reliably serve as a downstream oracle for evaluating the generative models (CVAE and CycleGAN). An effective oracle requires a classifier with strong discriminative power — particularly high sensitivity to the minority class — so that differences in the quality of synthetic images are reflected in measurable performance gaps. With AUC values hovering near chance level across all baselines, the classifier cannot be trusted to meaningfully distinguish between real and generated pneumonia images.
One factor likely contributing to the weak results is the aggressive downsampling of images to 128×128 pixels. Chest X-ray interpretation is a fine-grained task: subtle findings such as consolidation patterns, interstitial markings, and bilateral distribution cues — all relevant to pneumonia diagnosis — may be lost or distorted at this resolution. Increasing the input resolution to 256×256 is a natural next step and is expected to benefit not only the classifier but also the generative models, which would operate on richer and more diagnostically informative representations.

A further challenge is the strong class imbalance in the training set, where positive (pneumonia) cases represent a small fraction of the data. Despite mitigation strategies such as weighted sampling and loss scaling, no baseline achieved consistent recall for the minority class. A promising direction is to augment the training set with synthetic images generated by the CVAE and CycleGAN models, which would directly test whether the generative models produce samples informative enough to improve downstream classification.

### CVAE

The final CVAE implementation is a stronger baseline than the initial fully connected version because it combines convolutional encoding/decoding, skip connections, FiLM-style conditioning, and lung-mask guidance. The conditioning mechanism remains direct: disease label, normalized age, and gender are explicitly passed into the model, while the lung mask is provided as an additional input channel and used to guide reconstruction losses.

The main positive result is anatomical preservation. The skip connections help maintain global chest structure, and the saved reconstructions show that the model learns a stable mapping for the 128 x 128 lung-aware inputs. This makes the CVAE useful for producing controlled counterfactual pairs where the source image and generated image remain visually aligned.

The main limitation remains counterfactual strength. The generated images tend to be smooth and the visual changes are subtle, so qualitative inspection alone is not enough to claim that the model introduced pneumonia-related evidence. For this reason, the CVAE discussion now prioritizes classifier-facing validity metrics: flip rate and confidence change. A useful CVAE counterfactual should flip the downstream classifier to the target class and shift the target-class confidence in the expected direction.

Visual attribution maps were removed from the evaluation plan. Change heatmaps are still useful for visually checking whether modifications are diffuse or concentrated, especially relative to the lung fields, but they are not treated as a standalone explanation. The decisive question is whether the counterfactual changes alter classifier behavior.

Main limitations observed in the CVAE experiment include:

- **Smooth reconstructions**: generated images preserve global structure but lose fine radiological detail.
- **Class imbalance**: the model may be biased toward healthy-looking reconstructions because healthy images dominate the selected dataset.
- **Subtle counterfactual changes**: visual differences may be too weak to change classifier predictions consistently.
- **Oracle dependence**: flip rate and confidence change are only as reliable as the downstream classifier used to compute them.
- **Clinical plausibility**: classifier behavior does not replace expert validation, so generated changes should not be interpreted as clinically confirmed pneumonia evidence.

### CycleGAN

The choice of CycleGAN as the translation backbone was motivated by its ability to train on **unpaired data**, which is a realistic constraint in clinical settings where matched healthy/pneumonia images from the same patient are rarely available. The cycle consistency constraint is particularly well-suited to the counterfactual generation goal since it enforces that only disease-relevant features are modified, while preserving the underlying anatomy of the patient.

The use of a **128 × 128** resolution was a deliberate trade-off between image fidelity and computational cost. Chest X-rays at this resolution retain coarse pathological patterns (opacification, consolidation) while keeping training feasible.

The **identity loss** was included to avoid unnecessary texture shifts when a generator receives an image already in its target domain, which is especially important for grayscale medical images where contrast changes could be mistaken for pathology. Altough, this might be one of the reasons why we can't visualize much difference between original and generated images. 

Main limitations and future direction of the CycleGAN include:

- **Resolution**: 128 × 128 may be insufficient to capture fine-grained radiological features such as subtle consolidation boundaries. A follow-up experiment at 256 × 256 is planned if computational resources allow.
- **Class imbalance**: We still have much more cases of healthy X-ray than Pneumonia, which might be affecting the performance of the generator, specially to learn H $\rightarrow$ P translaction. The unpaired setup partially mitigates this, but a more balanced set would strengthen the evaluation.
- **FID as sole metric**: FID measures distributional similarity but not clinical relevance. Future work will complement it with a classifier-based counterfactual validity check. Generated pneumonia images should flip a downstream classifier's prediction with high probability.
- **Artifact reduction**: minor artifacts observed at lung borders in some generated images warrant investigation into whether longer training, higher resolution, or spectral normalization in the discriminator could reduce them.

# Conclusion

> In the final project submission (D3), the conclusion is expected to outline, among other aspects, possibilities for the project’s continuation.
> Can talk about enhance models, use diffusion or something like a cvae+cycleGAN. Can talk about explainability.

This work presented a generative framework for counterfactual image generation in chest X-rays, combining a CVAE and a CycleGAN trained on the NIH Chest X-ray dataset to translate images between healthy and pneumonia domains. The CVAE produced anatomically consistent counterfactuals (mean SSIM 0.82) but with limited realism (FID 136.5), while the CycleGAN achieved sharper translations (mean FID 115.3) though with residual artifacts. A downstream binary classifier was trained, with the best model reaching a test AUC of 0.67, insufficient to reliably assess counterfactual validity.

Remaining challenges include class imbalance, image resolution, and the need for classifier-grounded explainability evaluation. Future work will increase image resolution to 256×256, refine model architectures, augment the training set with synthetic images to improve classifier performance, and use the improved classifier to validate counterfactual generation and support explainability through difference maps and Grad-CAM comparisons.

# Ethical considerations

Although generative models and explainable AI methods have shown promising results in chest X-ray analysis, their use in medical imaging raises important ethical concerns. Previous studies have highlighted risks related to dataset bias, demographic imbalance, and the learning of spurious correlations that may affect model reliability and fairness across patient groups [13,14]. In this project, the dataset is predominantly composed of male and middle-aged patients, which may limit the model’s ability to generalize to other demographic groups. In addition, medical synthetic images and counterfactual explanations should not be interpreted as clinical ground truth or used as a substitute for professional diagnosis, since generated images may contain unrealistic or misleading findings. Another important limitation of this work is the absence of healthcare specialists to validate the generated counterfactuals. As a result, the evaluation relies mainly on classifier behavior and explainability metrics, which may not fully reflect clinical validity or diagnostic reliability. Therefore, transparency, careful evaluation, and responsible interpretation are essential when applying generative AI methods in healthcare contexts [13,15].

# Bibliographic References

1. Kumar, Amar, et al. "Prism: High-resolution & precise counterfactual medical image generation using language-guided stable diffusion." arXiv preprint arXiv:2503.00196 (2025).
2. Atad, Matan, et al. "Counterfactual explanations for medical image classification and regression using diffusion autoencoder." arXiv preprint arXiv:2408.01571 (2024).
3. Hou, Junlin, et al. "Self-explainable AI for medical image analysis: A survey and new outlooks." arXiv preprint arXiv:2410.02331 (2024).
4. Ahmed, Fahad et al. "Explainable artificial intelligence (XAI) in medical imaging: a systematic review of techniques, applications, and challenges." BMC Medical Imaging vol. 26, no. 1, 37. 5 Jan. 2026, doi:10.1186/s12880-025-02118-w.
5. Chen, H., Gomez, C., Huang, C. M. et al. "Explainable medical imaging AI needs human-centered design: guidelines and evidence from a systematic review." npj Digital Medicine 5, 156 (2022). https://doi.org/10.1038/s41746-022-00699-2.
6. Mertes S., Huber T., Weitz K., Heimerl A., and Andre E. "GANterfactual: Counterfactual Explanations for Medical Non-experts Using Generative Adversarial Learning." Frontiers in Artificial Intelligence 5:825565 (2022). doi:10.3389/frai.2022.825565.
7. Zia, Tehseen, Zeeshan Nisar, and Shakeeb Murtaza. "Counterfactual Explanation and Instance-Generation using Cycle-Consistent Generative Adversarial Networks." arXiv preprint arXiv:2301.08939 (2023).
8. Oakden-Rayner, L. "Exploring the ChestXray14 dataset: problems." https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/ (2017).
9. Wang, X. et al. "ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2097-2106, doi:10.1109/CVPR.2017.369 (2017).
10. Siekiera, Julia, and Stefan Kramer. "Counterfactual Explanations in Medical Imaging: Exploring SPN-Guided Latent Space Manipulation." arXiv preprint arXiv:2507.19368 (2025).
11. Sohn, Kihyuk, Honglak Lee, and Xinchen Yan. "Learning structured output representation using deep conditional generative models." Advances in Neural Information Processing Systems 28 (2015).
12. Xia, Tian, et al. "Mitigating attribute amplification in counterfactual image generation." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2024.
13. Herington J, McCradden MD, Creel K, Boellaard R, Jones EC, Jha AK, Rahmim A, Scott PJH, Sunderland JJ, Wahl RL, Zuehlsdorff S, Saboury B. Ethical Considerations for Artificial Intelligence in Medical Imaging: Data Collection, Development, and Evaluation. J Nucl Med. 2023 Dec 1;64(12):1848-1854. doi: 10.2967/jnumed.123.266080. PMID: 37827839; PMCID: PMC10690124.
14. Jones, C., Castro, D.C., De Sousa Ribeiro, F. et al. A causal perspective on dataset bias in machine learning for medical imaging. Nat Mach Intell 6, 138–146 (2024). https://doi.org/10.1038/s42256-024-00797-8
15. Rajpurkar, Pranav, et al. "CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning." arXiv, 2017. https://arxiv.org/abs/1711.05225



