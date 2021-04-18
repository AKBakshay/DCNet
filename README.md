# DCNet: Dark Channel Network for single-image dehazing
### [Paper](https://rdcu.be/chalm)
## Abstract 
Single-image dehazing is an extensively studied field and an ill-posed problem faced by vision-based systems in an outdoor environment. This paper proposes a dark channel network to estimate the transmission map of an input hazy scene for single-image dehazing. The architecture constitutes two major components—feature extraction layer and convolutional neural network layer. The former extracts the haze relevant features, while latter convolve these features with filter kernels to estimate the true scene transmission. Finally, the estimated transmission map is used to obtain the dehazed image using atmospheric scattering model. The experiments have been performed on synthetic hazy images and benchmark hazy dataset available in the literature. The performance of the proposed architecture outperforms the existing models in terms of standard quantitative metrics—mean square error, structural similarity index, and peak signal-to-noise ratio.

## Code
**Code will be uploaded soon**

## Results
Experiment results of DCNet on standard images used for dehazing are shown below.

Comparison of DehazeNet and DCNet on “bird” image is shown below. From left to right and top to bottom: Clear image and its feather region, hazy image and its feather region, dehazed image using DehazeNet and its feather region, and dehazed image using DCNet and its feather region

<p align="center">
  <img width="600"  src="https://github.com/AKBakshay/DCNet/blob/main/docs/images/bird_feathers_comparison.png">
</p>

Comparative visualization of dehazing methods on Canyon image. The zoomed views are shown for sky region in input hazy image (yellow box) and dehazed image using DCNet (red box) 

<p align="center">
  <img width="1000"  src="https://github.com/AKBakshay/DCNet/blob/main/docs/images/canyon_comparison.png">
</p>

Comparative visualization of dehazing methods on Girls image. The zoomed views are shown for region with illumination and haze both, in input hazy image (yellow box) and dehazed image using DCNet (red box)

<p align="center">
  <img width="1000"  src="https://github.com/AKBakshay/DCNet/blob/main/docs/images/girls_comparison.png">
</p>

---

Comparison on Synthetic dataset
<p align="center">
  <img width="800"  src="https://github.com/AKBakshay/DCNet/blob/main/docs/images/DCNet_synthetic_dataset.png">
</p>

Comparison on Middleburry-stereo dataset
<p align="center">
  <img width="800"  src="https://github.com/AKBakshay/DCNet/blob/main/docs/images/DCNet_middleburry_dataset.png">
</p>

## Bibtex
```
@article{DBLP:journals/mva/BholaSV21,
  author    = {Akshay Bhola and Teena Sharma and Nishchal K. Verma},
  title     = {DCNet: Dark Channel Network for single-image dehazing},
  journal   = {Mach. Vis. Appl.},
  volume    = {32},
  number    = {3},
  pages     = {62},
  year      = {2021},
  url       = {https://doi.org/10.1007/s00138-021-01173-x},
  doi       = {10.1007/s00138-021-01173-x},
  timestamp = {Wed, 07 Apr 2021 16:01:22 +0200},
  biburl    = {https://dblp.org/rec/journals/mva/BholaSV21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
