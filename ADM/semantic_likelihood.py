import os
from PIL import Image

import torch
import numpy as np
import clip


def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float) -> np.ndarray:
    """
    Calculate the entropy of multivariate Gaussian distributions with covariance
    Diag(1/M * Σ(μₘ²) - μ̄²) + σ²I in batch mode.
    """
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    _, _, D = mu_array.shape

    diagonal_terms = np.mean(mu_array**2, axis=1) - np.mean(mu_array, axis=1) ** 2
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)  # because with only M=6 samples there are some negative values
    eigenvalues = diagonal_terms + sigma_squared  # Shape: (N, D)
    log_det = np.sum(np.log(eigenvalues), axis=1)  # Shape: (N,)

    entropy = 0.5 * log_det + 0.5 * D * (np.log(2 * np.pi) + 1)

    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy


def compute_generative_uncertainty(path, M, eu_type="entropy"):
    print(f"Loading samples from {M} models from {path}")

    #### 1) Compute clip features

    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # count number of images in path
    N = len(os.listdir(f"{path}/{0}/imgs"))

    for m in range(M):
        print(f"Processing model {m}")
        clip_vecs = []
        for i in range(N):
            image = preprocess(Image.open(f"{path}/{m}/imgs/{i:05d}.png")).unsqueeze(0).to(device)

            with torch.no_grad():
                clip_vecs.append(model.encode_image(image))

        clip_vecs = torch.concat(clip_vecs, dim=0)
        print(clip_vecs.shape)
        torch.save(clip_vecs, f"{path}/{m}/clip_features.pt")

    #### 2) Compute the entropy of the semantic likelihood

    features = []
    for m in range(M):
        path = f"{path}/{m}/clip_features.pt"
        features.append(torch.load(path))

    features = torch.stack(features, dim=0)
    features = np.transpose(features.cpu().numpy(), (1, 0, 2))
    print(features.shape)

    if eu_type == "entropy":
        eu = gaussian_entropy(features, sigma_squared=1e-3)
    else:
        raise ValueError(f"Unknown epistemic uncertainty type: {eu_type}")

    print(f"Saving: {path}/{eu_type}_clip.npy")
    np.save(f"{path}/{eu_type}_clip.npy", eu)


if __name__ == "__main__":

    PATH = "/nvmestore/mjazbec/diffusion/bayes_diff/exp_repo_clean/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"
    M = 6

    compute_generative_uncertainty(PATH, M, eu_type="entropy")
