# UViT experiment
Code here is based on the following 2 repos:
- [BayesDiff](https://github.com/karrykkk/BayesDiff)
- [UViT](https://github.com/baofff/U-ViT)

## Setup
- Download the UViT model checkpoint using [this link](https://drive.google.com/file/d/13StUdrjaaSXjfqqF7M47BzPyhMAArQ4u/view?usp=share_link.pt) and update `uvit_path` in `main.sh`
- Download the Stable Diffusion's Autoencdoer checkpoint using [this link](https://drive.google.com/drive/folders/1yo-XhqbPue3rp5P57j6QbA5QZx6KybvP?usp=sharing) and update `encoder_path` in `main.sh`
- Download the [ImageNet dataset](https://www.image-net.org/) and update `config.dataset.path` in `./configs/imagenet256_uvit_huge.py`
- Make sure to complete the **Setup** steps in `../eval/README.md` 

## Experiment
1. Generate samples `bash main.sh`
2. Compute *generative uncertainty* via `python ../semantic_likelihood.py --path PATH`
    - for `PATH` use `exp_dir` returned from Step 1
2. Compute metrics (FID, precision, recall) via `bash eval.sh` (make sure to run the script from `../eval/` folder and to update `PATH` in `eval.sh`)


