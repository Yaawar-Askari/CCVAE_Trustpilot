# Disentangling Factors of Variation with Cycle-Consistent Variational Auto-Encoders

This repository contains the code for the paper: Disentangling Factors of Variation with Cycle-Consistent Variational Auto-encoders (https://arxiv.org/abs/1804.10469). 



The repository provides the files necessary to train our architecture on Trustpilot dataset, while it can be generalised to other datasets in our paper by changing the **data_loader** and **networks** files.


## Abstract

Generative models that learn disentangled representations for different factors of variation in an image can be very useful for targeted data augmentation. By sampling from the disentangled latent subspace of interest, we can efficiently generate new data necessary for a particular task. Learning disentangled representations is a challenging problem, especially when certain factors of variation are difficult to label. In this paper, we introduce a novel architecture that disentangles the latent space into two complementary subspaces by using only weak supervision in form of pairwise similarity labels. Inspired by the recent success of cycle-consistent adversarial architectures, we use cycle-consistency in a variational auto-encoder framework. Our non-adversarial approach is in contrast with the recent works that combine adversarial training with auto-encoders to disentangle representations. We show compelling results of disentangled latent subspaces on three datasets and compare with recent works that leverage adversarial training.


## Architecture

### Forward cycle

<p align="center">
<img src="images/forward_phase.png" alt="Forward Cycle" width="457px" />
</p>

### Reverse cycle

<p align="center">
<img src="images/backward_phase.png" alt="Reverse cycle" width="587px" />
</p>

## Results

#### Empirical comparison of quality of disentangled representations

<p align="center">
<img src="images/table.png" alt="Classification accuracy table" width="581px" align="middle" />
</p>

#### Comparison of image generation capabilities of various architectures

<p align="center">
<img src="images/mnist_swap.png" alt="Image generation" width="593px" align="middle" />
</p>

#### t-SNE plots to show leakage of information pertaining to specified factors of variation into the unspecified latent space

<p align="center">
<img src="images/t_sne.png" alt="t-SNE plot" width="672px" align="middle" />
</p>
