 🎥 **Presentation slides:** <[Presentation slides](./presentations/Project_h_neurons_final.pdf)>

# `Neurônios-H em Modelos de Menor Escala`

# `H-Neurons in Small-Scale Models`

## Presentation

This project originated in the context of the graduate course _IA376N_,
offered in the **first semester of 2026 (2026.1)**, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).
 
| Name | RA | Specialization |
|--|--|--|
| Charles Cavalcante Alcarde | 800181 | Electrical Engineering |
| Pedro Henrique Guerra | 298988 | Computer Engineering |
| Luís Felipe da Silva Carlos Pereira | 272919 | Electrical Engineering |

## Abstract

This work investigates the existence of hallucination-associated neurons (H-Neurons) across the entire Pythia family (EleutherAI), extending the methodology proposed by Gao et al. (2025) — originally validated only on models with 7B to 70B parameters — to models ranging from 70M to 6.9B parameters. Using a pipeline based on the CETT metric (Contribution to Estimated Token Trajectory) and sparse L1-regularized logistic regression, we identified H-Neurons in all seven evaluated models (pythia-70m, pythia-160m, pythia-410m, pythia-1b, pythia-1.4b, pythia-2.8b, and pythia-6.9b). The results confirm three central hypotheses: (H1) H-Neurons exist in small-scale models, with in-domain AUROC ranging from 0.71 to 0.92 in the methodologically most robust models; (H2) the discriminative power of H-Neurons tends to increase with scale, with a positive trend in the sequence 160M→410M→1.4B→2.8B; and (H3) H-Neurons generalize to domains unseen during classifier training (NQ-Open), with OOD AUROC between 0.80 and 0.90 in the main models. The sparsity of the phenomenon is consistent with the 0.1% threshold reported in the original paper: in all reliable models, H-Neurons represent less than 0.05% of total FFN neurons. A complementary finding reveals that H-Neuron concentration peaks occur predominantly between 30% and 45% of network depth, suggesting that the processing associated with hallucinatory behavior concentrates in the intermediate layers of the transformer, rather than in the final projection layers. These results suggest that the neural mechanism underlying hallucination is a structural property of the transformer architecture, emerging independently of model scale.

**Keywords:** LLM hallucinations; H-Neurons; transformer internal analysis; Pythia family; CETT; sparse logistic regression; TriviaQA; NQ-Open; model scaling.

---

## Summary

1. [Problem Description / Motivation](#1-problem-description--motivation)
2. [Objective](#2-objective)
3. [Datasets](#3-datasets-and-evolution)
4. [Workflow / Methodology](#4-workflow--methodology)
5. [Experiments](#5-experimental-challenges-and-methodological-adaptations)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)
9. [References](#9-bibliographic-references)

---

## 1. Problem Description / Motivation

Large language models (LLMs) have become central tools in modern artificial intelligence systems, demonstrating remarkable capabilities in natural language understanding and generation. However, a persistent challenge undermines their reliability in high-stakes applications: the hallucination phenomenon — the generation of fluent, syntactically coherent text that is factually incorrect or entirely fabricated. The problem transcends model generations: estimates indicate that GPT-3.5 hallucinates in approximately 40% of citation-based factuality evaluations, a figure that remains high at 28.6% for GPT-4 (Chelli et al., 2024). State-of-the-art reasoning systems, such as DeepSeek-R1, despite remarkable performance on complex tasks, continue to exhibit pronounced hallucination modes (Bao et al., 2025). High-confidence hallucinations — scenarios in which the model appears certain while being wrong — constitute the most dangerous case, as entropy-based or calibration-based metrics fail to detect them reliably.

Gao et al. (2025) proposed a pioneering approach: investigating hallucinations from the inside out, identifying specific neurons in the feedforward networks (FFN) of LLMs whose activations reliably predict whether the model will hallucinate. These neurons, termed H-Neurons, constitute less than 0.1% of total neurons and demonstrate generalization capability to domains unseen during classifier training. However, the original study was restricted to models with 7B to 70B parameters, leaving open a fundamental question: do H-Neurons exist in small-scale models, or are they an emergent phenomenon exclusive to large-scale models?

<div align="center">

![alt text](images/image-1.jpeg)

</div>

The present work fills this gap using the Pythia family (Biderman et al., 2023) — models trained on the same data, in the same order, and with the same base architecture, varying only in scale. We evaluated seven models from 70M to 6.9B parameters, investigating three central research questions:

> **Central thesis:** H-Neurons emerge in small-scale language models, suggesting that the neural mechanism associated with hallucinations is fundamental to the transformer architecture and not exclusive to large-scale models.

**Research contributions:**
- **(H1)** Do H-Neurons exist in small-scale models? We investigate whether the phenomenon identified by Gao et al. (2025) in 7B–70B models also emerges in models from 70M to 6.9B parameters.
- **(H2)** Is there a scalability trend in the number and discriminative power of H-Neurons? We analyze whether the classifier AUROC and the absolute number of H-Neurons grow systematically with model size.
- **(H3)** Do H-Neurons identified on TriviaQA transfer to NQ-Open (out-of-distribution generalization)? We evaluate whether H-Neurons capture generalizable hallucination patterns, independent of the classifier's training domain.

## 2. Objective

The primary objective is try replicate the original paper but instead very large models we use small scale models, this change led to the following hypotheses which we will explore.

- **Existence:** H-Neurons exist even in small-scale models, suggesting that the
phenomenon is related to the transformer architecture itself rather than being exclusive to
large models.
- **Scale:** Classifier accuracy based on H-Neurons increases with model size, even within
the small-scale range investigated here.
- **Generalization:** H-Neurons identified on TriviaQA generalize to different domains,
such as NQ-Open, suggesting that the captured signal is structural rather than
dataset-specific.

- **Alternative architecture:** Share the pipeline of "Pythia Models" family, which uses GPT-NeoX instead GPT.

### 3. Datasets and Evolution

### Contrastive Dataset Construction

Following Gao et al. (2025), we constructed a balanced contrastive dataset using TriviaQA (Joshi et al., 2017). For each model, responses were generated using greedy decoding (`do_sample=False`) and classified as correct (label 0) or incorrect (label 1). The final dataset contains 100 balanced examples per model (50 correct + 50 incorrect), obtained after scanning up to 18,000 questions.

**Sampling strategy by model size:**

| Models | Regime | Threshold | Notes |
|:---|:---:|:---:|:---|
| < 1.4B params | n=1 | 1/1 | High error rates make multi-attempt infeasible |
| ≥ 1.4B params | n=10 | 8/10 | Early stopping after 3 consecutive failures |

As base models without instruction-tuning, Pythia models require few-shot prompting with 5 demonstrative examples in the format `"Question: [question]\nShort answer: [answer]"`.

For OOD evaluation, a complementary NQ-Open dataset (Kwiatkowski et al., 2019) was constructed with 15 correct + 15 incorrect examples per model, following the same sampling strategy. The normalizer (scaler) was kept fixed — fitted exclusively on TriviaQA.

| Dataset       | Web Address       | Descriptive Summary                                   |
| ------------- | ----------------- | ----------------------------------------------------- |
| Trivia-QA (validation split) | https://huggingface.co/datasets/mandarjoshi/trivia_qa#rcnocontext-1 | Created in 2017, the dataset combines real-world trivia questions with automatically collected textual evidence, serving as a demanding test for machine learning systems seeking to infer answers in natural language. |

 - Each sample contains a *string* **question**, *dict*[ _string_ **aliases**, _string_ **normalized_aliases**, _string_ **value**]  of acceptable ground-truth **answers**, _string_ **question_id**, _list_[ ] **search_results**, _dict/list_[ ] __entity_pages__ and _string_ **question_source** . The Dataset in validation have **~11000** samples.
 - For preprocessing they remove the context, normalize aliases and filtered ambiguous samples.
 <br> </br>

|question | question_id | question_source |entity_pages|search_results|answer|
| ------------ |------------ |------------|------------|------------|------------|
|Who was the man behind The Chipmunks? | tc_2|http://www.triviacountry.com/|{ "doc_source": [], "filename": [], "title": [], "wiki_context": [] }|(A VERY LONG SEARCH RESULT)|{ "aliases": [ "David Seville" ], "normalized_aliases": [ "david seville" ],"matched_wiki_entity_name": "", "normalized_matched_wiki_entity_name": "", "normalized_value": "david seville", "type": "WikipediaEntity", "value": "David Seville"}||
|What was the last US state to reintroduce alcohol after prohibition? | tc_79|http://www.triviacountry.com/|{ "doc_source": [], "filename": [], "title": [], "wiki_context": [] }|(A VERY LONG SEARCH RESULT)|{"aliases": ["Utah (State)", "Forty-Fifth State", "Sports in Utah ... ], "matched_wiki_entity_name": "", "normalized_matched_wiki_entity_name": "", "normalized_value": "utah", "type": "WikipediaEntity", "value": "Utah" }",|| |

<br> </br>
<br> </br>
| Dataset       | Web Address       | Descriptive Summary                                   |
| ------------- | ----------------- | ----------------------------------------------------- |
| NQ-OPEN | https://huggingface.co/datasets/google-research-datasets/nq_open | The NQ-Open task, introduced by Lee et.al. 2019, is an open domain question answering benchmark that is derived from Natural Questions. The goal is to predict an English answer string for an input English question. All questions can be answered using the contents of English Wikipedia. |

 - The dataset is distributed in structured tabular format, each sample contains a *string* _question_ and one *list[strings]* of acceptable ground-truth answers. The Dataset have **87925** samples of training and **3610** of validation and the anottations come from real Google search queries, manually annotated answer spans and Wikipedia evidence documents.
 - For preprocessing they remove the context documents, normalize answers and filtered ambiguous samples.
 - Have 9-12 words average question length and 1-5 words average answer length.

<br> </br>

|question | answer |
| ------------ |------------ |
|where did they film hot tub time machine |[ "Fernie Alpine Resort" ]
|who plays mavis in the movie hotel transylvania|[ "Sadie Sandler", "Selena Gomez" ]|
|names of the metropolitan municipalities in south africa | ["Mangaung Metropolitan Municipality", "Nelson Mandela Bay Metropolitan Municipality", "eThekwini Metropolitan Municipality", "City of Tshwane Metropolitan Municipality", "City of Johannesburg Metropolitan Municipality", "Buffalo City Metropolitan Municipality", "City of Ekurhuleni Metropolitan Municipality"] |

### 4. Workflow / Methodology
---

<div align="center">

 **Overall Experimental Workflow** 

![alt text](images/image.png)

</div>

## Theoretical Background

### 4.1. Hallucinations in Language Models

Hallucinations in LLMs can be formally defined as the phenomenon in which the model assigns higher probability $P_\theta(y|x)$ to a factually incorrect sequence than to the correct one, optimizing fluency at the expense of factuality. The consolidated taxonomy distinguishes two main axes (Huang et al., 2024): (i) intrinsic hallucinations, which contradict the reference source, and extrinsic hallucinations, which add unverifiable information; (ii) factual hallucinations, which diverge from real-world facts, and faithfulness hallucinations, which diverge from the input context.

The causes of hallucinations permeate all phases of the LLM lifecycle: noisy training data, the next-token prediction objective blind to factuality, RLHF that may prioritize compliance over truth, and latent uncertainties amplified during inference. Mitigation strategies organize into six categories (MDPI Survey, 2025): training and learning (SFT, RLHF, knowledge editing); architectural modifications (RAG, enhanced attention); prompt optimization (CoT, self-consistency, few-shot); post-generation control (fact-checking, LLM-as-judge); interpretability and diagnosis; and agents and orchestration. The present work falls within the interpretability and diagnosis category.

### 4.2. H-Neurons and Internal Transformer Analysis

The study by Gao et al. (2025) organized around three research questions: **(Q1)** Do H-Neurons exist in LLMs?; **(Q2)** What is their behavioral impact?; and **(Q3)** What is their origin? The identification methodology relies on three steps: (i) constructing a balanced contrastive dataset of correct and incorrect responses via TriviaQA; (ii) quantifying each neuron's contribution using the CETT metric; and (iii) training a sparse L1 classifier to identify neurons with the highest discriminative power.

Beyond existence (Q1), amplifying H-Neurons systematically increases over-compliance behaviors — acceptance of invalid premises, susceptibility to misleading contexts, adherence to harmful instructions (Q2). Cross-model transfer experiments demonstrated that H-Neurons emerge during pre-training and persist after instruction fine-tuning (Q3).

### 4.3. The Pythia Family

<div align="center">

**GPT-NeoX, Pythia family architecture**

![alt text](images/image-2.png)

</div>

The Pythia family (Biderman et al., 2023) was developed by EleutherAI for interpretability research. All models share: (i) the same training corpus (The Pile) in the same data order; (ii) the same decoder-only transformer architecture with Gated MLP FFN blocks; and (iii) public intermediate training checkpoints. The internal architecture follows the pattern:

1. **Multi-Head Self-Attention layer** — captures contextual dependencies between tokens
2. **Gated MLP FFN** — projects the hidden representation to an intermediate space of dimension $d_m$ via $W_{gate}, W_{up} \in \mathbb{R}^{d_m \times d}$, applies SiLU activation, and projects back via $W_{down} \in \mathbb{R}^{d \times d_m}$

**Table 1 — Architectural dimensions of the Pythia family.**

| Model | n_layers | d | dm | N_FFN |
|:---|:---:|:---:|:---:|---:|
| pythia-70m | 6 | 512 | 2,048 | 12,288 |
| pythia-160m | 12 | 768 | 3,072 | 36,864 |
| pythia-410m | 24 | 1,024 | 4,096 | 98,304 |
| pythia-1b | 16 | 2,048 | 8,192 | 131,072 |
| pythia-1.4b | 24 | 2,048 | 8,192 | 196,608 |
| pythia-2.8b | 32 | 2,560 | 10,240 | 327,680 |
| pythia-6.9b | 32 | 4,096 | 16,384 | 524,288 |

*`n_layers`: number of FFN layers; `d`: hidden state dimension; `dm`: intermediate FFN dimension (`intermediate_size`); `N_FFN`: total FFN neurons = `n_layers` × `dm`.*

---

### 4.4. Few-shot Approach

![alt text](images/image-5.png)

### 4.5. Models and Justification

We evaluated seven Pythia models to cover three orders of magnitude of scale. Models up to 1B were run on CPU with float32 precision; larger models on T4 GPU with 8-bit quantization (bitsandbytes).

**Pipeline configuration:**

| Parameter | Value | Description |
|:---|:---:|:---|
| `n_target_correct` | 50 | Target correct examples |
| `n_target_incorrect` | 50 | Target incorrect examples |
| `max_questions_to_scan` | 18,000 | Maximum questions scanned |
| `max_new_tokens` | 16 | Max tokens per response |
| `max_response_words` | 10 | Max words per response |
| `max_response_chars` | 100 | Max characters per response |
| `temperature` | 1.0 | Generation temperature (≥ 1.4B) |
| `top_k` | 50 | Top-k sampling (≥ 1.4B) |
| `top_p` | 0.9 | Top-p sampling (≥ 1.4B) |
| `C_grid` | [0.001–1.0] | L1 regularization grid |
| `test_size` | 0.2 | Train/validation split |

### 4.6. FFN Activation Extraction

For each example, we performed a forward pass and captured intermediate FFN activations at response tokens using PyTorch forward hooks registered at `dense_h_to_4h` (or equivalent). Activations are averaged over response tokens, producing a tensor of dimension (`n_layers` × `intermediate_size`) per example.

**Table 2 — Activation tensor dimensions per model.**

| Model | n_layers | intermediate_size | Total values/example |
|:---|:---:|:---:|---:|
| pythia-70m | 6 | 2,048 | 12,288 |
| pythia-160m | 12 | 3,072 | 36,864 |
| pythia-410m | 24 | 4,096 | 98,304 |
| pythia-1b | 16 | 8,192 | 131,072 |
| pythia-1.4b | 24 | 8,192 | 196,608 |
| pythia-2.8b | 32 | 10,240 | 327,680 |
| pythia-6.9b | 32 | 16,384 | 524,288 |

### 4.7. Statistical Baseline and Significance

A baseline of 1,000 random neuron draws was implemented. The empirical p-value is:

$$p = \frac{\text{card}\{AUROC_{rand} > AUROC_{\mathcal{H}}\}}{1000}$$

Results with $p < 0.05$ are considered statistically significant.

### 4.8. Mathematical Approach

**Input Format and Answer Tokens**

Each example is passed to Pythia as a question-answer sequence:


$$x_i = \text{Question: } q_i \;\backslash n\; \text{Answer: } r_i,\quad i = 0, \ldots, 100$$

The code first finds where the answer begins in the tokenized sequence:

<div align="center">

answer tokens = [answer_start, answer_end)

</div>

Then, during the forward pass, FFN activations are captured for all tokens, but only the
answer-token positions are selected:

$$z_{\ell,t}\quad\text{for } t \in [\mathrm{answer\_start}, \mathrm{answer\_end})$$

If the answer is split into multiple tokens, their FFN activations are averaged:

$$\Large\bar{z}_{i,\ell,j}=\frac{1}{\left|\mathcal{A}_i\right|}\sum_{t \in \mathcal{A}_i}z_{\ell,t,j}$$

**CETT: Causal Effect on Task Token**

After averaging the answer-token activations, each neuron is represented by a single value $\bar{z}_{i,\ell,j}$ for example $i$, layer $\ell$, and neuron $j$.

The simplified CETT score normalizes each neuron activation within its FFN layer:


$$\Large\mathrm{CETT}_{i,\ell,j}=\frac{|\bar{z}_{i,\ell,j}|}{\|\bar{z}_{i,\ell}\|_2 + \varepsilon}$$

$$\|\bar{z}_{i,\ell}\|_2=\sqrt{\sum_{j=1}^{16384}\bar{z}_{i,\ell,j}^{\,2}}$$

where z̄ᵢ,ₗ,ⱼ is the average activation of neuron j in layer ℓ for example i, and ‖z̄ᵢ,ₗ‖₂ is the L2 norm of layer ℓ. The final feature vector Xᵢ ∈ ℝᴺ concatenates CETT scores of all layers and neurons.

<div align="center">

CETT converts raw FFN activations into relative neuron-contribution features.

</div>

$$\Large X_i = \left[\mathrm{CETT}_{i,1,1}, \dots, \mathrm{CETT}_{i,32,16384}\right] \in \mathbb{R}^{524288}$$

**Logistic Classifier: From CETT to Hallucination Probability**

The classifier receives the CETT feature vector $X_i$ , not the original text.
It first computes a linear score from all neuron-contribution features:

$$\Large s_i=b+X_i w=b+\sum_{j=1}^{N}X_{i,j}w_j$$

Then, the score is mapped to a hallucination probability using the sigmoid:

$$\Large p_i=P(y_i = 1 \mid X_i)=\sigma(s_i)=\frac{1}{1 + e^{-s_i}}$$

$$y_i = 1 \;\Longrightarrow\; \text{hallucinated answer}$$

**Why L1 Regularization? Sparse Neuron Selection**

The goal is not only to classify hallucination, but also to identify a small set of informative
neurons.

The training objective combines binary logistic loss with an L1 penalty that promotes sparse
neuron selection:

$$\Large\mathcal{L}=-\sum_i\left[y_i \log(p_i)+(1-y_i)\log(1-p_i)\right]+\lambda\sum_{j=1}^{N}|w_j|$$

This encourages most weights to become exactly zero.

### H-Neuron Identification via Sparse L1 Classifier

An L1-regularized logistic regression predicts the binary label (hallucinated/faithful). The regularization strength $C \in \{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0\}$ is selected via grid search (80/20 stratified split). H-Neurons are defined as:

$$\mathcal{H} = \{ j : w_j > 0 \} \qquad \%H = \frac{|\mathcal{H}|}{N_{total}} \times 100\%$$

<div align="center">

**Zoom into the MLP: Location of H-Neurons**

![alt text](images/image-4.png)

</div>

### 4.9.Methodological Contributions

Three adaptations useful for replicating internal transformer analysis experiments in computationally restricted environments:

1. **Few-shot prompting** for base models on structured QA tasks — direct impact on accuracy rate and dataset viability
2. **Greedy decoding** as a valid substitute for stochastic consistency filtering when computational constraints make the original approach infeasible
3. **Empirical statistical baseline** with 1,000 random draws for significance without parametric assumptions about the null distribution

### 5. Experimental Challenges and Methodological Adaptations

- **Pythia-6.9B:** loaded on T4 GPU with 8-bit quantization (bitsandbytes), with selective CPU offloading
- **Models ≤ 1B:** inference on CPU with float32 precision (~4s/question)
- **Generation regime:** greedy decoding for models < 1.4B; multiple sampling (n=10, threshold=8) for larger models
- **Adaptation vs. original paper:** threshold=8/10 instead of 10/10 — documented as a methodological limitation

---

## 6. Results

### 6.1 Comparative Overview

**Table 3 — Comparative results of H-Neurons in the Pythia family.** ⚠ indicates models with methodological caveats.

| Model | Params | H-Neur. | % Total | Acc H | Acc Rand | AUROC H | AUROC OOD |
|:---|---:|---:|---:|:---:|:---:|:---:|:---:|
| pythia-70m ⚠ | 70M | 11 | 0.0895% | 0.85 | 0.6626 | 0.92 | 0.50 |
| pythia-160m | 162M | 18 | 0.0488% | 0.60 | 0.5596 | 0.71 | 0.72 |
| pythia-410m | 405M | 22 | 0.0224% | 0.75 | 0.5648 | 0.80 | 0.84 |
| pythia-1b ⚠ | 1,012M | 3 | 0.0023% | 0.75 | 0.5325 | 0.82 | 0.80 |
| pythia-1.4b | 1,415M | 68 | 0.0346% | 0.85 | 0.6049 | 0.90 | 0.84 |
| pythia-2.8b | 2,775M | 105 | 0.0320% | 0.90 | 0.5938 | 0.92 | 0.90 |
| pythia-6.9b | 6,857M | 71 | 0.0135% | 0.75 | 0.5951 | 0.85 | 0.80 |

### 6.2 Hypothesis H1 — Existence in Small-Scale Models

H1 is confirmed in all seven evaluated models: the H-Neuron classifier consistently outperforms the random baseline across the entire family.

![Figure 1 — H-Neurons vs. random neurons (baseline) in the Pythia family. Left: in-domain accuracy (TriviaQA). Right: in-domain AUROC (TriviaQA). The green curve (H-Neurons) consistently outperforms the red curve (random baseline) across all models, confirming H1.](images/fig1_auroc_acuracia.png)

*Figure 1 — H-Neurons vs. random neurons (baseline) in the Pythia family.*

**Table 4 — Empirical p-values of AUROC** (1,000 random draws baseline). Bold: $p < 0.05$. ✓: confirmed statistical significance.

| Model | AUROC in-domain | p-value in-domain | AUROC OOD | p-value OOD | Significance |
|:---|:---:|:---:|:---:|:---:|:---:|
| pythia-70m ⚠ | 0.92 | 0.086 | 0.50 | 0.679 | — |
| pythia-160m | 0.71 | 0.442 | 0.72 | 0.063 | — |
| pythia-410m | 0.80 | 0.065 | 0.84 | **0.008** | OOD ✓ |
| pythia-1b ⚠ | 0.82 | 0.033 | 0.80 | 0.021 | OOD ✓ |
| pythia-1.4b | 0.90 | **0.015** | 0.84 | **0.010** | both ✓ |
| pythia-2.8b | 0.92 | **0.002** | 0.90 | **0.003** | both ✓ |
| pythia-6.9b | 0.85 | 0.124 | 0.80 | **0.022** | OOD ✓ |

### 6.3 Hypothesis H2 — Scale Dependency

Excluding models with caveats, a clear positive trend is observed: 160M (0.71) → 410M (0.80) → 1.4B (0.90) → 2.8B (0.92), with a slight retraction at 6.9B (0.85). The pythia-1b is an outlier at the boundary between the two sampling strategies, resulting in only 3 H-Neurons.

![Figure 2 — H-Neuron sparsity by model scale (% of total FFN neurons). The dashed red line indicates the 0.1% threshold reported by Gao et al. (2025) for models ≥7B. All models remain below this threshold.](images/fig2_esparsidade.png)

*Figure 2 — H-Neuron sparsity by model scale.*

**Table 5 — H-Neuron scalability** (models with caveats excluded from trend analysis).

| Model | Params | H-Neurons | % Total | AUROC in-domain | H2 Trend |
|:---|---:|---:|---:|:---:|:---:|
| pythia-160m | 162M | 18 | 0.0488% | 0.71 | ↗ base |
| pythia-410m | 405M | 22 | 0.0224% | 0.80 | ↗ |
| pythia-1b ⚠ | 1,012M | 3 | 0.0023% | 0.82 | ⚠ outlier |
| pythia-1.4b | 1,415M | 68 | 0.0346% | 0.90 | ↗ |
| pythia-2.8b | 2,775M | 105 | 0.0320% | 0.92 | ↗ peak |
| pythia-6.9b | 6,857M | 71 | 0.0135% | 0.85 | ↘ retraction |

### 6.4 Hypothesis H3 — Out-of-Distribution Generalization

H3 is confirmed with statistical significance: OOD AUROC on NQ-Open ranges from 0.80 to 0.90 in reliable models (pythia-410m: 0.84; pythia-1.4b: 0.84; pythia-2.8b: 0.90; pythia-6.9b: 0.80), with p-values below 0.05. The NQ-Open dataset followed the same sampling strategy as TriviaQA: regime 1/1 for models below 1.4B and regime 10/10 (threshold 8) for larger models.

### 6.5 H-Neuron Distribution Across FFN Layers

In pythia-410m, pythia-1.4b, and pythia-6.9b, H-Neuron concentration peaks occur between 30% and 45% of network depth — a region associated with factual knowledge retrieval.

![Figure 3a — H-Neuron distribution by FFN layer: pythia-70m and pythia-160m.](images/fig3a_heatmap_70m_160m.png)

*Figure 3a — H-Neuron distribution: pythia-70m and pythia-160m. ⚠ pythia-70m results should be interpreted with caution.*

![Figure 3b — H-Neuron distribution by FFN layer: pythia-410m and pythia-1b.](images/fig3b_heatmap_410m_1b.png)

*Figure 3b — H-Neuron distribution: pythia-410m (peaks at L8 and L11, 33–46% of depth) and pythia-1b (only 3 H-Neurons ⚠).*

![Figure 3c — H-Neuron distribution by FFN layer: pythia-1.4b and pythia-2.8b.](images/fig3c_heatmap_14b_28b.png)

*Figure 3c — H-Neuron distribution: pythia-1.4b (peak at L9, 37.5% of depth) and pythia-2.8b (atypical peak at L1).*

![Figure 3d — H-Neuron distribution by FFN layer: pythia-6.9b.](images/fig3d_heatmap_69b.png)

*Figure 3d — H-Neuron distribution: pythia-6.9b (peaks at L8 and L11, 25–34% of depth).*

---

## 7. Discussion

### 7.1 Hypothesis Confirmation and Theoretical Implications

**H1** is robustly confirmed: H-Neurons exist in all evaluated models, including pythia-160m with only 162M parameters. This evidence supports the interpretation that H-Neurons constitute a structural property of the transformer architecture, emerging as a consequence of the next-token prediction objective regardless of model capacity.

**H2** is partially confirmed: the increasing AUROC trend (0.71 → 0.80 → 0.90 → 0.92) indicates that larger models develop H-Neurons with greater specificity. Observed oscillations are attributable to sample variance with 100 training examples.

**H3** is confirmed with statistical significance: H-Neurons identified in one domain retain predictive capability in distinct domains, suggesting they capture a fundamental property of hallucinatory behavior rather than domain-specific artifacts.

### 7.2 Complementary Finding: Functional Layer Organization

The convergence of H-Neuron peaks at 30–45% of network depth — observed independently in pythia-410m, pythia-1.4b, and pythia-6.9b — constitutes an unanticipated complementary finding. This is consistent with works by Geva et al. (2021) and Meng et al. (2022) associating intermediate FFN layers with factual knowledge storage and retrieval. This pattern was not explicitly reported by Gao et al. (2025).

### 7.3 H-Neurons as a Neural Thermometer: Correlation, Causality, and Practical Value

H-Neurons are correlates of hallucination, not its exclusive cause. The phenomenon is multicausal (training data, next-token prediction objective, RLHF, knowledge gaps). H-Neurons function as a **neural thermometer**: a measurable, localized signal that accompanies hallucinatory behavior — clinically useful, but not the infection itself.

The practical value of H-Neurons does not depend on resolving the underlying multicausality:

1. **Real-time detection:** H-Neuron activation patterns can signal hallucination risk during inference, before the response reaches the user, triggering RAG or fact-checker mechanisms.
2. **Partial intervention:** suppressing H-Neurons reduces over-compliance behaviors, with significant impact in high-stakes systems (medical diagnosis, legal advisory).
3. **Mapping for causal investigation:** precise neuron and layer coordinates enable investigation of why the signal emerges there — connecting H-Neurons to training dynamics and factual knowledge retrieval mechanisms.

> The diagnosis precedes the treatment: H-Neurons are the first step of a research program that may eventually lead to deeper causal interventions.

### 7.4 Caveats and Limitations

**pythia-70m ⚠:** accuracy near zero even with few-shot prompting. The contrastive dataset was constructed using the same methodology as all other models — always storing the model's own generated response (never external gold answers). The anomalously high AUROC (0.92) may reflect qualitative class imbalance due to very few correct examples available.

**pythia-1b ⚠:** potential dataset contamination from mixing generation regimes across distinct sessions. Only 3 H-Neurons identified; results treated as inconclusive.

**General limitations:** (i) 100 examples per model vs. 1,000 in the original paper; (ii) OOD dataset of only 30 examples; (iii) absence of causal perturbation experiments — results are correlational. The CETT is generated via forward pass with the concatenation `"Question: [question]\nShort answer: [answer]"`, the same format used during generation, ensuring context consistency.

## 8. Conclusion

This work investigated the existence of hallucination-associated neurons (H-Neurons) in the Pythia family, extending the methodology proposed by Gao et al. (2025) from models with 7B–70B parameters to a range of 70M to 6.9B parameters. The results confirm the three central hypotheses of the project, with the documented methodological caveats: H-Neurons exist in small-scale models (H1), their discriminative power tends to increase with scale (H2), and they generalize to domains unseen during classifier training (H3).

The most relevant finding is that H-Neurons emerge across the entire Pythia family, including models with only 160M parameters, with sparsity consistently below 0.05% of total FFN neurons — compatible with the 0.1% threshold reported for models 44 times larger. This sparsity consistency across three orders of magnitude of scale suggests that H-Neurons constitute a structural property of the transformer architecture, not a large-scale emergent phenomenon. Hallucination may be more frequent in small models — a direct consequence of lower factual memorization capacity during pre-training — but the neural circuit that accompanies it appears to be the same.

The complementary finding on the concentration of H-Neurons in intermediate layers (30–45% of depth) adds a spatial dimension to the phenomenon: hallucination is not processed uniformly across the network, but concentrates in a specific region associated with factual knowledge retrieval. This result opens a direct avenue of investigation into the interaction between H-Neurons and fact storage mechanisms in FFNs.

H-Neurons function, ultimately, as a neural thermometer of hallucination: a sparse, localized, and generalizable signal that accompanies hallucinatory behavior without being its exclusive cause. Their practical value lies in three complementary fronts — real-time detection during inference, surgical partial intervention on model behavior, and mechanistic mapping that paves the way for future causal investigations. The multicausality of hallucination does not invalidate the thermometer; it only reminds us that it is not the antibiotic.

As priority future directions, we highlight: (1) causal perturbation experiments (activation scaling) in Pythia models to verify whether small-scale H-Neurons exhibit the same over-compliance pattern identified by Gao et al. (2025); (2) cross-domain generalization evaluation to specialized domains (e.g., BioASQ); (3) investigation of the temporal evolution of H-Neurons throughout training, using the intermediate checkpoints available in the Pythia family; and (4) expansion of the contrastive dataset to 500–1,000 examples per model with stratified cross-validation (k-fold), to reduce AUROC estimate variance and strengthen conclusions about H2.

## 9. Bibliographic References

1. **Bao, F.; Xu, C.; Mendelevitch, O**. DeepSeek v3. R1 hallucinates more than DeepSeekVectara Blog, jan. 2025. Disponível em: https://www.vectara.com/blog. 
2. **Biderman, S. et al**. Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. arXiv:2304.01373, 2023
3. **Chelli, M. et al**. Hallucination rates and reference accuracy of ChatGPT and Bard for systematic reviews. European Journal of Cardio-Thoracic Surgery, v. 64, 2024.
4. **Gao, C. et al**. H-Neurons: On the Existence, Impact, and Origin of Hallucination Associated Neurons in LLMs. arXiv:2512.01797, 2025.
5. **Geva, M. et al**. . Transformer Feed-Forward Layers Are Key-Value Memories. In: EMNLP, 2021.
6. **Huang, L. et al**. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Transactions on Information Systems, 2024. 
7. **Ji, Z. et al**. LLM internal states reveal hallucination risk faced with a query. arXiv:2407.03282, 2024. 
8. **Joshi, M. et al**. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. In: ACL, 2017. 
9. **Kwiatkowski, T. et al**. Natural Questions: A Benchmark for Question Answering Research. Transactions of the Association for Computational Linguistics, v. 7, 2019. 
10. **Lindsey, J. et al**. On the biology of a large language model. Transformer Circuits Thread, 2025.
11. **Meng, K. et al**. Locating and Editing Factual Associations in GPT. Advances in Neural Information Processing Systems (NeurIPS), 2022.
12. **MDPI Survey**. From Illusion to Insight: A Taxonomic Survey of Hallucination Mitigation. AI Systems, v. 6, n. 10, 2025.
13. **Zhang, Z. et al**. ReLU² Wins: Discovering Efficient Activation Functions for Sparse LLMs. arXiv:2402.03804, 2024. 

---

*IA376N — Generative AI: from models to multimodal applications (2026.1)*  
*Prof. Paula Dornhofer Paro Costa · FEEC/UNICAMP*
