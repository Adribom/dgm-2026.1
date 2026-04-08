# EEGAN - O modelo...

# EEGAN - The model...

## Presentation

This project originated in the context of the graduate course _IA376N - Generative AI: from models to multimodal applications_,
offered in the first semester of 2026, at Unicamp, under the supervision of Prof. Dr. Paula Dornhofer Paro Costa, from the Department of Computer and Automation Engineering (DCA) of the School of Electrical and Computer Engineering (FEEC).

> Include name, RA, and specialization focus of each group member. Groups must have at most three members.
> |Name | RA | Specialization|
> |--|--|--|
> | Daniel Neto | 169408 | Computer Engineering|
> | Enzo | 123456 | Computer Engineering|
> | Marcelo | 123456 | Computer Engineering|

## Project Summary Description

> Description of the project theme, including generating context and motivation.

This project is in the context of biological signals. These signals are of difficult acquisition, because subjects must undergo a tiresome procedure to create a small amount of data and the number of subjects willing to undergo such a procedure is small. Since biological signals are conditioned by the subjects' individual characteristics, even databases with hundreds of hours can present low signal variation. To address this issue, we propose a method of generating XXXXXXX signals, based on the work of [1].

> Description of the main goal of the project.

The main goal of the project is to generate reliable EMG data, which is not only similar to the target domain, but is also capable of improving the classification scores on (hand-joint tasks ???).

> Clarify what the output of the generative model will be.

The output of the generative model will be EMG signals of the same dimension as the input.

> Include in this section a link to the presentation video of the project proposal (maximum 5 minutes).

## Proposed Methodology

> For the first submission, the proposed methodology must clarify:
>
> - Which dataset(s) the project intends to use, justifying the choice(s).

For this model, the datasets used will be emg2pose ([2]) and XXXXXXX.

> - Which generative modeling approaches the group already sees as interesting to be studied.

The generative modeling approach that will be the base for this study is the GAN presented in [1]. We found it to be interesting because it obtained a significant result for the time and for introducing a novelty regarding GANS: instead of naice sampling from a Gaussian in the Z latent space, it sampled from a known, controllable latent space, that is, an audio signal. In our work we propose XXXXXXX to do something similar, by sampling from hand joint angles.

> - Reference articles already identified and that will be studied or used as part of the project planning.

Cited in bibliography.

> - Tools to be used (based on the group’s current vision of the project).

PyTorch

> - Expected results.

We expect the greater variation introduced by the synthetic signals to improve the classification score of Benchmark models.

> - Proposal for evaluating the synthesis results.

The quality of the results will be measured by the proximity of the synthetic data to samples from the target distribution (through RMS, spectrum comparison), as well as the classification accuracy difference when synthetic data is added to the input. 

## Schedule

> Proposed schedule. Try to estimate how many weeks will be spent on each stage of the project.

WEEK 1 to 2: study the database and integrate it with the generator from [1]. Test proximity to target distribution

WEEK 3 to 4: train the Encoder that maps the EMG (generator) back to the hand joint angles. Does the new loss term improve training?

WEEK 4 to 5: if results are satisfactory, test the synthetic data on classifiers. Was there improvement in relation to the real data only?

WEEK 6 to 7: change the GAN for another model, sample from a Gaussian instead from known joint angles. Does the result improve?

WEEK 8 to 11: final changes, writing of repository and results, prepare presentation

## Bibliographic References

> Point out in this section the bibliographic references adopted in the project.
1. Scheck, K., Schultz, T. (2023) STE-GAN: Speech-to-Electromyography Signal Conversion using Generative Adversarial Networks. Proc. Interspeech 2023, 1174-1178, doi: 10.21437/Interspeech.2023-174
2. Bhasin, Rohin & Bolarinwa, Anuoluwapo & Cai, Yujun & Danielson, Nathan & Han, Shangchen & Marshall, Jesse & Merel, Josh & Pnevmatikakis, Eftychios & Salter, Sasha & Schlager, Collin & Spurr, Adrian & Walkington, Peter & Wang, Robert & Warren, Richard. (2024). emg2pose: A Large and Diverse Benchmark for Surface Electromyographic Hand Pose Estimation. 55703-55728. 10.52202/079017-1770. 
