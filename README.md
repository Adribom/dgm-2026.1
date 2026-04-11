# `Geração de Imagens com IA: Desafios no Controle de Atributos sob Viés de Dados`

# `AI Image Generation: Challenges in Attribute Control under Data Bias`

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).


> |Name | RA | Specialization|
> |--|--|--|
> | Patrícia P. Giordano | 971352 | Electrical Engineering|
> | Gabriel Carvalho Freitas | 323616 | Electrical Engineering|
> | Silvia A. P. Olivio | 224932 | Electrical Engineering|

## Project Summary Description

### Description of the project theme, including generating context and motivation.

This project aims to evaluate the behavior of text-to-image generative AI models, focusing on their ability to correctly interpret simple textual descriptions. The rapid advancement of generative models has enabled high-quality image synthesis from text; however, these models often exhibit limitations related to reliability, bias, and control over specific attributes .

The main goal of this project is to investigate how pre-trained open-source models, such as Stable Diffusion (available through platforms like Hugging Face), handle prompts where there is a potential conflict between the user’s input and the statistical patterns learned during training.

The project is motivated by the observation that these models frequently fail to correctly apply specific attributes—such as color—to objects. For example, prompts such as:

- “white carrot”
- “pink classroom blackboard”
- “purple polar bear”

Often result in images where the model ignores the requested attribute and instead generates outputs aligned with common real-world representations (e.g., orange carrots, green/black boards, white polar bears). As the examples below:

![Resultado do experimento](imagens/Pink_board_2.jpg)

This behavior suggests that the model prioritizes learned statistical correlations over explicit user instructions, revealing a limitation in its ability to disentangle attributes (e.g., color) from object identity.

The central research question of this project is:

> #### *Is this limitation primarily caused by biases in the training data distribution, or by constraints in the model architecture itself?*

## Main Goal

The main objective of this project is to evaluate whether the application of fine-tuning techniques to pre-trained open-source text-to-image generation models—such as Stable Diffusion—improves the fidelity in representing specific visual attributes, particularly color, with respect to the instructions provided in the prompt.

Specifically, the study aims to investigate whether, after fine-tuning with datasets that present greater diversity of these attributes (e.g., different colors applied to the same object), the model is able to correctly apply the requested color to specific regions of the image (such as a classroom blackboard), or whether it still exhibits limitations in attribute localization and control, resulting in the incorrect application of color to other regions of the scene (such as walls or adjacent objects), especially in elements that exhibit strong bias in the training data, such as classroom blackboards, which are traditionally associated with green or black colors.

# Main Hypothesis
The main hypothesis of this project is that the inability of text-to-image models to correctly apply specific attributes is primarily influenced by biases in the training data distribution, which leads the model to favor statistically dominant representations over explicit user instructions.

# Secondary Questions 
- To support the main hypothesis, the project investigates the following secondary questions:
- To what extent do training data biases influence the model’s output when attributes conflict with common representations?
- Can fine-tuning with controlled data improve the model’s ability to correctly apply attributes?
- Does the model architecture itself limit the ability to disentangle attributes from objects?
- Are alternative architectures (e.g., Beta-VAE) more effective in handling attribute control?

# Expected Output of the generative model.
The output of the generative model in this project will consist of images generated from controlled textual prompts, with a specific focus on attribute manipulation, especially color.

In the initial state (pre-trained model), the system is expected to generate images influenced by learned statistical patterns, often failing to correctly apply uncommon attributes specified in the prompt.

After applying fine-tuning with a synthetic dataset composed of objects with non-standard attribute combinations (e.g., unusual colors), the expected state-of-the-art outcome is that the model becomes capable of correctly modifying object attributes according to the prompt specification.

In particular, the improved model should be able to:

- Correctly apply specified attributes (such as color) to objects, even when they contradict common real-world representations
- Reduce bias toward dominant patterns learned during pre-training
- Generate images that are more consistent with the input prompt

> # Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).

## Proposed Methodology

> For the first submission, the proposed methodology must clarify:

# Dataset
We will generate a synthetic dataset with non-standard object attributes (e.g., colored blackboards) and use LoRA fine-tuning to improve attribute control in a pre-trained text-to-image model.

## Which dataset(s) the project intends to use, justifying the choice(s).

## Which generative modeling approaches the group already sees as interesting to be studied.
The following methods are considered within the scope of this project:
- Diffusion-based models, particularly Latent Diffusion Models used in Stable Diffusion, which serve as the primary architecture for image generation.
- LoRA (Low-Rank Adaptation), which will be used as an efficient approach for fine-tuning the pre-trained model with controlled data.

**If the fine-tuned dataset does not yield satisfactory results and sufficient time is available, VAE-based models may be explored, with a focus on analyzing attribute disentanglement. However, this step is considered optional and may be addressed as future work, depending on time constraints and overall project scope.**

## Reference articles already identified and that will be studied or used as part of the project planning. 
https://arxiv.org/pdf/2112.10752 - Treinamento do difusion 
https://arxiv.org/pdf/2006.11239 - Denoising Diffusion Probabilistic Models
https://arxiv.org/pdf/2112.10741 - GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

## Tools to be used (based on the group’s current vision of the project).
- Python
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion
- LoRA (Low-Rank Adaptation)
- CLIP (for evaluation)
- Google Colab (for training and experiments)

## Expected results.

## Proposal for evaluating the synthesis results.
CLIP Score: measuring the semantic alignment between generated images and input prompts

## Schedule

> Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project.

## Bibliographic References

> Point out in this section the bibliographic references adopted in the project.
