# Generative Uncertainty for Diffusion Models
Code for paper [Generative Uncertainty in Diffusion Models](TODO:insert-arxiv-link).

## Main Dependencies
* python = 3.9
* torch = 2.2.0
* [laplace-torch](https://github.com/aleximmer/Laplace)

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name diff-uq python=3.9`
3. Activate conda environment:  `conda activate diff-uq`
4. Install dependencies, using `pip install -r requirements.txt`

## Code
- For ADM experiments on ImageNet 128x128, see README.md in `./ADM`
- For UViT experiments on ImageNet 256x256, see README.md in `./UViT`

In case of issues with the code, open a github issue (preferred) or reach out to m.jazbec@uva.nl .

## Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support. 