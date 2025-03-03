# ADM experiment
Code here is based on the following 2 repos:
- [BayesDiff](https://github.com/karrykkk/BayesDiff)
- [ADM](https://github.com/openai/guided-diffusion)

## Setup
- Download the ADM model checkpoint using [this link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/128x128_diffusion.pt) and update `model.ckpt_dir` in `./configs/imagenet128_guided.yml`
- Download the [ImageNet dataset](https://www.image-net.org/) and update `data.path` in `./configs/imagenet128_guided.yml`
- Make sure to complete the **Setup** steps in `../eval/README.md` 

## Experiment
1. Generate samples `bash main.sh`
2. Compute *generative uncertainty* via `python semantic_likelihood.py --path PATH`
    - for `PATH` use `exp_dir` returned from Step 1
2. Compute metrics (FID, precision, recall) via `bash eval.sh` (make sure to run the script from `../eval/` folder and to update `PATH` in `eval.sh`)


