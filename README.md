# nk-diffusion based on Jittor and JDiffusion

## Introduction

nk-diffusion is a project **dedicated to transferring amazing diffusion model-based projects** from PyTorch to [**Jittor**](https://github.com/Jittor/jittor), **harnessing Jittor's high performance and unique advantages**.

At the core of Jittor is its JIT compiler, which converts Python code into **efficient CUDA instructions in real-time**, automatically optimizing computations for *speed and memory efficiency* based on input shapes and types. 

By leveraging these features, nk-diffusion not only enhances performance but also provides flexibility and ease of use. Furthermore, these projects serve as exemplars for future high-quality Jittor projects, showcasing the framework's potential for research, education, and production environments.

## Installation

Our Work is based on:
- [jittor](https://github.com/JittorRepos/jittor)
- [jtorch](https://github.com/JittorRepos/jtorch)
- [diffusers_jittor](https://github.com/JittorRepos/diffusers_jittor)
- [transformers_jittor](https://github.com/JittorRepos/transformers_jittor)

### 0. Clone JDiffusion & Prepare Env
```bash
git clone https://github.com/JittorRepos/JDiffusion.git
#We recommend using conda to configure the Python environment.
conda create -n jdiffusion python=3.9
conda activate jdiffusion
```
### 1. Install Requirements

Our code is based on JTorch, a high-performance dynamically compiled deep learning framework fully compatible with the PyTorch interface, please install our version of library.

```bash
pip install git+https://github.com/JittorRepos/jittor
pip install git+https://github.com/JittorRepos/jtorch
pip install git+https://github.com/JittorRepos/diffusers_jittor
pip install git+https://github.com/JittorRepos/transformers_jittor
```
or just
```bash
pip install -r requirement.txt
```
### 2. Install JDiffusion

```bash
cd JDiffusion
pip install -e .
```
We also provide a [docker image](https://cg.cs.tsinghua.edu.cn/jittor/assets/docker/jdiffusion.tar) (md5:62c305028dae6e62d3dff885d5bc9294) about our environment.

### 3.Optional Requirements
 If you encounter `No module named 'cupy'`:
```bash
# Install CuPy from source
pip install cupy
# Install CuPy for cuda11.2 (Recommended, change cuda version you use)
pip install cupy-cuda112
```
### 4.Install Specific App

Follow the README in example folder to install the application you want! 

The projects we currently support in the Jittor version: 

- [PhotoMaker](https://github.com/TencentARC/PhotoMaker) <img src="https://photo-maker.github.io/assets/logo.png" height=25>: Customizing Realistic Human Photos via Stacked ID Embedding
- [StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion) <img src="https://github.com/HVision-NKU/StoryDiffusion/assets/49511209/f79da6b7-0b3b-4dd7-8dd0-ba0b15306fe6" height=25>: Consistent Self-Attention for Long-Range Image and Video Generation 
- [1Prompt1Story](https://github.com/byliutao/1Prompt1Story) : Free-Lunch Consistent Text-to-Image Generation Using a Single Prompt
- [DIRECT](https://github.com/Gong1130/DIRECT) <img src="./example/DIRECT/assets/direct_logo1.png" height=25>: Direct 3D-Aware Object Insertion via Decomposed Visual Proxies
- [StyleExpert](https://github.com/HVision-NKU/StyleExpert): Mixture of Style Experts for Diverse Image Stylization
