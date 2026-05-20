
# EMGAN - From speech to EMG

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

|Name | RA | Specialization|
|--|--|--|
| Daniel Neto | 169408 | Computer Engineering|
| Enzo Campos | 247069 | Computer Engineering|
| Marcelo Ferreira | 300882 | Computer Engineering|

## Abstract

> Summary of the objective, methodology **and results** obtained. In submission **D2**, it is acceptable to report partial results. Suggested maximum of 100 words.

The STE-GAN model represents a promising approach in generative adversarial frameworks, yet scaling its capabilities across diverse data environments remains a significant challenge. This paper presents the mapping and structural analysis of the STE-GAN model across multiple datasets. While a wide range of repositories was successfully cataloged, their inherent technical complexity and heterogeneity prevented immediate integration into a unified training pipeline. As an initial milestone, the original STE-GAN architecture was successfully executed and validated, establishing a stable performance baseline. To overcome current integration and convergence limitations, we outline the next development phases, which focus on implementing a novel distance metric to optimize generation and introducing structural modifications to the generator. This work lays the foundation for robust multi-subject data synthesis in future iterations.

[PRESENTATION LINK](https://docs.google.com/presentation/d/122uZN8ldLYW1H7PQhn2qsqW1NxxE-XUZY3rv4bdlU4Y/edit?usp=sharing)

## Problem Description / Motivation

> Description of the generating context of the project theme. Motivation for addressing this project theme.

Surface Electromyography (sEMG) signals of articulatory muscles reflect the speech production process. As such, they are a biosignal of interest for Silent Speech Interfaces (SSIs) [1], which aim to enable speech communication without depending on acoustic speech. However, acquiring these biological signals is notoriously difficult. Subjects must undergo a tiresome recording procedure, resulting in small datasets with low subject variability. Furthermore, the collected data is often lawfully restricted. Consequently, there is a strong motivation for Speech-to-EMG (STE) modeling, which could be explored to generate new, artificial EMG signals to improve ETS model training and mitigate data scarcity.

To address the issues of data acquisition and aid in the development of robust SSIs, we propose a method of generating synthetic EMG signals based on the STE-GAN architecture [2]. 

## Objective

> Description of what the project aims to do.  
> It is possible to specify a general objective and specific objectives of the project.

The main goal of this project is to generate reliable EMG data from acoustic speech that is not only similar to the target domain, but also capable of retaining high linguistic accuracy (e.g., maintaining a low Word Error Rate) after being converted back to audio.

The model in [2] has multiple components and losses. This project will introduce and change components and losses in an attempt to surpass the metrics established in the original paper.

## Methodology

> Clearly and objectively describe, citing references, the methodology proposed to achieve the project objectives. - OK
> Describe datasets used.  - NEXT SECTION
> Cite reference algorithms. - OK
> Justify the reasons for the chosen methods. - OK
> Point out relevant tools. - OK
> Describe the evaluation methodology (how will it be assessed whether the objectives were met or not?). - OK

The generative modeling approach that will serve as the baseline for this study is the Speech-to-Electromyography Generative Adversarial Network (STE-GAN) presented in [2]. This approach was found to be particularly compelling due to its strong and reproducible results. Notably, STE-GAN directly converts acoustic speech to EMG signals in an end-to-end fashion, eliminating the need to predict intermediate features. In doing so, it achieved impressive metrics, such as a high Envelope Correlation Coefficient of 0.66 and over 80% Phoneme Accuracy on the generated signals. Furthermore, instead of naively sampling from a Gaussian distribution, the model conditions the generation on a controllable latent space—soft speech units extracted from audio — which importantly enables the model to generalize to speech of unseen speakers. Finally, this choice is strongly motivated by the fact that the authors made their code openly available, which greatly facilitates project reproducibility.

For the project's development, the main codebase will be built in Python, leveraging libraries such as PyTorch, NumPy, and other data science and signal processing tools. Jupyter Notebooks will be used for interactive coding and evaluation, while GitHub will handle version control and code hosting. Finally, to ensure the models are trained within a feasible timeframe, GPU computing will be relied upon, specifically an NVIDIA RTX 5070 Ti (16 GiB), with the possibility of incorporating additional GPUs as needed.

At first, the results presented in [2] will be reproduced. Then, by evaluating the model on additional datasets (see next session) and testing architectural modification hypotheses, it is expected to both consolidate the architecture for this type of data and improve its evaluation scores.

All the metrics presented in [2] will be maintained, as the evaluation will be made based on them. Most importantly, this project will attempt to surpass the correlation between synthetic and real EMG envelope data, as well as the correctness of generated speech from synthetic EMG (WER) reported in the original paper. Additional metrics may also be proposed to evaluate any architectural changes.

### Datasets and Evolution

> List the datasets used in the project.  
> For each dataset, include a mini-table in the model below and then provide details on how it was analyzed/used, as in the example below.

| Dataset       | Web Address       | Descriptive Summary                                   |
| ------------- | ----------------- | ----------------------------------------------------- |
| The EMG-UKA corpus for electromyographic speech processing [3] | https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus | This dataset contains ~6 seconds samples of aligned EMG, audio and phonemes from 8 subjects, that sum 1h40min of signal. The EMG is sampled at 600 Hz and audio at 16 kHz. Six facial electrodes are used to capture the EMG. |
| Digital Voicing of Silent Speech [4] | https://zenodo.org/records/4064409 | This dataset contains samples with less than 10 seconds of aligned EMG and audio from 1 subject, summing 20h in total. The EMG is sampled at 1000Hz and audio at 16kHz. 8 electrodes are used to capture the EMG from both face and neck muscles. |
| An open dataset of multidimensional signals based on different speech patterns in pragmatic Mandarin [5] | https://www.nature.com/articles/s41597-025-06213-z#:~:text=Surface%20electromyography%20,to%20enhance%20speech%20decoding%20accuracy | This dataset contains EEG/EMG recordings of  mandarin speakers. This dataset will not be used, since it's not avaiable.  |
| DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets [6] | https://osf.io/jbsu2/overview | This dataset contains ~4 seconds samples of aligned EMG, audio and phonemes from 1 subject, that sum 3h40min of signal. The EMG is sampled at 800 Hz and audio at 44.1 kHz. Eight facial electrodes are used to capture the EMG. |
| AVE Speech Dataset: A Comprehensive Benchmark for Multi-Modal Speech Recognition Integrating Audio, Visual, and Electromyographic Signals [7] | https://huggingface.co/datasets/MML-Group/AVE-Speech |The dataset consists of bipolar EMG signals sampled at 1000 Hz and synchronized audio recordings at 44.1 kHz. It includes data from 100 participants, each producing 100 Mandarin sentences multiple times. In addition to the speech signals, lip viseme information is provided. The dataset totals approximately 71 hours of recordings. |
| CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion [8] | https://www.uni-bremen.de/csl/forschung/lautlose-sprachkommunikation/csl-emg-array-corpus | This dataset contains ~5 seconds samples of aligned EMG and audio from 8 subjects, that sum 9h40min of signal. The EMG is sampled at 2048 Hz and audio at 16 kHz. 32 facial electrodes and 8 chin electrodes are used to capture the EMG. |
| emg2speech: synthesizing speech from electromyography using self-supervised speech models [9] | https://arxiv.org/pdf/2510.23969 | This dataset contains samples of EMG captured during voiced and silent speech, summing 10h of data collected from 2 subjects, from which 1h was collected from a subject with ALS. The EMG is sampled at 5000Hz, captured from 31 electrodes monopolar electrodes placed in the face and neck. Not avaiable yet. |
| SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech Recognition [10] | https://arxiv.org/pdf/2603.02847| Neckband emg recording of 4 speakers. No audio is given, this dataset will not be used. |
| Sentence-Level Silent Speech Recognition Using a Wearable EMG/EEG Sensor System with AI-Driven Sensor Fusion and Language Model [11] | https://www.mdpi.com/1424-8220/25/19/6168 | Only contains EEG and EMG signals, no audio, so this dataset will not be used. |

Here is a mapping of the position of EMG electrodes in the selected datasets:
| Corresponding muscle mentioned and approximations | EMG-UKA | Gaddy dataset | emg-VCTK | AVE Speech | CSL-EMG_Array | Cross-Speaker dataset |
|--------------------------------------------------|--------|---------------|----------|------------|----------------|------------------------|
| Zygomaticus major                                | ✔      | ✔             | ✔        |            |                | ✔                      |
| Zygomaticus minor                                |        | ~✔            |          |            |                | ~✔                     |
| Levator anguli oris                              | ✔      |               |          | ✔          |                |                        |
| Depressor anguli oris                            | ✔      | ✔             |          |            |                | ✔                      |
| Levator labii superioris                         |        | ✔             |          |            |                | ✔                      |
| Risorius                                         |        | ~✔            |          | ✔          |                | ~✔                     |
| Orbicularis oris                                 |        | ✔             |          |            |                | ✔                      |
| Mentalis                                         |        |               | ✔        |            |                |                        |
| Masseter                                         |        | ✔             | ✔        |            |                | ✔                      |
| Temporalis                                       |        |               |          |            | ✔              |                        |
| Lateral pterygoid                                |        | ✔             |          |            |                | ✔                      |
| Platysma                                         |        |               | ✔        |            |                |                        |
| Sternohyoid                                      |        | ✔             |          |            |                | ✔                      |
| Stylohyoid                                       |        |               |          |            | ✔              |                        |
| Omohyoid                                         |        | ✔             |          |            |                | ✔                      |
| Anterior belly of digastric                      | ✔      |               |          | ✔          |                |                        |
| Mylohyoid                                        |        |               |          | ✔          |                |                        |
| Genioglossus                                     |        |               |          |            | ✔              |                        |
| Hyoglossus                                       |        |               |          |            | ✔              |                        |
| Styloglossus                                     |        |               |          |            | ✔              |                        |
| Palatoglossus                                    |        |               |          |            | ✔              |                        |
| Tongue                                           | ✔      |               |          |            |                |                        |
| Unspecified                                      |        |               | ~✔       | ✔          |                | ✔                      |
> Provide a description of what you concluded about this dataset. Suggested guiding questions or information to include:
For this project, our models require datasets containing paired EMG and audio signals. Since this type of data is relatively rare, the 9 publicly available datasets listed above have all been surveyed. Their quality, compatibility, and suitability were thouroughly analysed and it was decided that, for now, the project will continue with the unique use of the Digital Voicing of Silent Speech [4] dataset. 

This decision was made based on the fact that each dataset was captured using a distinct channel setup (position and number-wise), which hinders merging between datasets. Also, it was decided that the blocks in the architecture are most important for the project, so the focus has shifted towards it.
> - What is the dataset format, size, type of annotation?
Regarding the characteristics of the datasets, all audio was available as .wav files and sampling frequency ranged from 16 kHz to 48 kHz. The EMG data came in various formats, with it's sampling frequency varying from 256 to 2048 Hz. Phonemes were captured for each sliding window of size 27ms and stride of 10ms, when available. All samples have a duration of a few seconds.
> - What transformations and preprocessing were done? Cleaning, re-annotation, etc.
No transformations were applied up to now, except for the main dataset [4], which was preprocessed as described in [2].
> - Include a summary with descriptive statistics of the dataset(s).
:)
> - Use tables and/or charts to describe the main aspects of the dataset that are relevant to the project.
This dataset uses only one subject and 6 channels, which were placed in the following muscles:

| | Location | Estimated muscle(s) |
| :--- | :--- | :--- |
| 1 | left cheek just above mouth | Zygomaticus major, w/possible crosstalk with Zygomaticus minor |
| 2 | left corner of chin | Depressor anguli oris |
| 3 | below chin back 3 cm | Sternohyoid |
| 4 | throat 3 cm left from Adam’s apple | Omohyoid |
| 5 | mid-jaw right | Masseter |
| 6 | right cheek just below mouth | Orbicularis oris, w/possible (strong) crosstalk with Risorius |
| 7 | right cheek 2 cm from nose | Levator labii superioris |
| 8 | back of right cheek, 4 cm in front of ear | Lateral pterygoid |
| ref | below left ear | - |
| bias | below right ear | - |

### Workflow

> Use a tool that allows you to design the workflow and save it as an image (e.g., Draw.io). Insert the image in this section.  
> You may choose to use a workflow manager (Sacred, Pachyderm, etc.), in which case use the manager to generate a diagram for you.  
> Remember that the goal of drawing the workflow is to help anyone who wishes to reproduce your experiments.

![Project Workflow](images/workflow.svg)

## Experiments, Results, and Discussion of Results

> In the intermediate project submission (**D2**), this section may contain partial results, explorations of implemented solutions, and  
> discussions about such experiments, including decisions to change the project trajectory or the description of new experiments as a result of these explorations.

After surveying all datasets as described above, many architectural changes were proposed. The ones that seemed more relevant are:

1. The addition of an EMG-specific pre-trained tokenizer to complement the already present EMG Encoder (trained on audio features). This will serve as a distance metric between the real and fake EMGs, as well as a new loss to be propagated back through the generator.

2. A new, attention-based generator to reduce the amount of CNN layers used originally.

For the first item, a SOTA self-supervised foundational model for EEGs and EMGs was chosen, the NeuroRVQ [12]. It surpasses already consolidated foundational models such as CBRAMOD [13] and Labram [14].

The second item's implementation details are yet to be decided. For now, no partial results regarding these changes were obtained.

> The discussion of results may be carried out in a separate section or integrated into the results section. This is a matter of style.  
> It is considered fundamental that the presentation of results should not serve as a treatise whose only purpose is to show that "a lot of work was done."  
> What is expected from this section is that it **presents and discusses** only the most **relevant results**, highlighting the **strengths and/or limitations** of the methodology, emphasizing aspects of **performance**, and containing content that can be classified as **organized, didactic, and reproducible sharing of knowledge relevant to the community**.

Only the original model [2] was run with dataset [4] and results similar to what the original authors reported were obtained.

## Conclusion

> The Conclusion section should recover the main information already presented in the report and point to future work.  
> In the intermediate project submission (**D2**), it may contain information about which steps or how the project will be conducted until its completion. 

The mapping and structural analysis of the STE-GAN model [2] has been presented, alongside multiple datasets. Although a wide range of datasets was successfully cataloged during this phase, their inherent technical complexity and heterogeneity prevented immediate integration for training.

As an initial partial result, the execution and validation of the original model were successfully consolidated, establishing a stable baseline for planned modifications. Moving forward, next steps will focus on two essential development fronts: the implementation of a new distance metric to optimize the generator and structural changes to the generator itself. These updates aim to enhance the model's generalization capabilities across diverse subjects.



## Bibliographic References


1. T. Schultz, M. Wand, T. Hueber, D. J. Krusienski, C. Herff, and J. S. Brumberg, “Biosignal-based spoken communication: A survey,” IEEE/ACM Transactions on Audio, Speech and Language
Processing, vol. 25, no. 12, pp. 2257–2271, 2017.

2. Scheck, K., Schultz, T. (2023) STE-GAN: Speech-to-Electromyography Signal Conversion using Generative Adversarial Networks. Proc. Interspeech 2023, 1174-1178, doi: 10.21437/Interspeech.2023-174

3. Wand, M., Janke, M., Schultz, T. (2014) The EMG-UKA corpus for electromyographic speech processing. Proc. Interspeech 2014, 1593-1597, doi: 10.21437/Interspeech.2014-379

4. David Gaddy and Dan Klein. 2020. Digital Voicing of Silent Speech. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5521–5530, Online. Association for Computational Linguistics.

5. Zhao, R., Bai, Y., Zhang, S. et al. An open dataset of multidimensional signals based on different speech patterns in pragmatic Mandarin. Sci Data 12, 1934 (2025). https://doi.org/10.1038/s41597-025-06213-z

6. Scheck, K., Dombeck, T., Ren, Z., Wu, P., Wand, M., Schultz, T. (2025) DiffMV-ETS: Diffusion-based Multi-Voice Electromyography-to-Speech Conversion using Speaker-Independent Speech Training Targets. Proc. Interspeech 2025, 5573-5577, doi: 10.21437/Interspeech.2025-1914

7. Zhou, D., Zhang, Y., Wu, J., Zhang, X., Xie, L., and Yin, E., “AVE Speech: A Comprehensive Multi-Modal Dataset for Speech Recognition Integrating Audio, Visual, and Electromyographic Signals”, arXiv e-prints, Art. no. arXiv:2501.16780, 2025. doi:10.48550/arXiv.2501.16780.

8. Diener, L., Vishkasougheh, M.R., Schultz, T. (2020) CSL-EMG_Array: An Open Access Corpus for EMG-to-Speech Conversion. Proc. Interspeech 2020, 3745-3749, doi: 10.21437/Interspeech.2020-2859

9. Harshavardhana T. Gowda, & Lee M. Miller. (2025). Non-invasive electromyographic speech neuroprosthesis: a geometric perspective.

10. Giusy Spacone, Sebastian Frey, Giovanni Pollo, Alessio Burrello, Daniele Jahier Pagliari, Victor Kartsch, Andrea Cossettini, & Luca Benini. (2026). SilentWear: an Ultra-Low Power Wearable System for EMG-based Silent Speech Recognition.

11. Satterlee, N.; Zuo, X.; Moon, K.; Lee, S.Q.; Peterson, M.; Kang, J.S. Sentence-Level Silent Speech Recognition Using a Wearable EMG/EEG Sensor System with AI-Driven Sensor Fusion and Language Model. Sensors 2025, 25, 6168. https://doi.org/10.3390/s25196168

12. Barmpas, K., Lee, N., Koliousis, A., Panagakis, Y., Adamos, D. A., Laskaris, N., & Zafeiriou, S. (2025). NeuroRVQ: Multi-scale EEG tokenization for generative large brainwave models. arXiv preprint arXiv:2510.13068. https://arxiv.org/abs/2510.13068

13. Wang, J., et al. (2025). CBraMod: A criss-cross brain foundation model for EEG decoding. In The Thirteenth International Conference on Learning Representations (ICLR).

14. Jiang, W.-B., et al. (2024). Large brain model for learning generic representations with tremendous EEG data in BCI. ICLR 2024.
