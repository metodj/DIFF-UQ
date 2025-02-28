import os
from PIL import Image

import torch
import numpy as np

import clip


def npy2png(array, png_path):
    if not os.path.exists(png_path):
        os.makedirs(png_path)

    for i in range(array.shape[0]):
        img = array[i].transpose(1, 2, 0)  # Change shape from (3, 256, 256) to (256, 256, 3)
        img = (img * 255).astype(np.uint8)  # Convert to uint8
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(png_path, f"{i:05d}.png"))


def gaussian_entropy(mu_array: np.ndarray, sigma_squared: float) -> np.ndarray:
    """
    Calculate the entropy of multivariate Gaussian distributions with covariance
    Diag(1/M * Σ(μₘ²) - μ̄²) + σ²I in batch mode.

    Args:
        mu_array: Array of means with shape (N, M, D) where:
                 N is the batch size
                 M is the number of samples per batch
                 D is the dimension
                 Can also accept (M, D) for single batch
        sigma_squared: The variance parameter σ² for the identity matrix term

    Returns:
        np.ndarray: Array of entropies with shape (N,) for batch input
                   or float for single input
    """
    # Handle single batch case
    if len(mu_array.shape) == 2:
        mu_array = mu_array[np.newaxis, ...]

    _, _, D = mu_array.shape

    diagonal_terms = np.mean(mu_array**2, axis=1) - np.mean(mu_array, axis=1) ** 2

    # clip diagonal terms to be non-negative
    diagonal_terms = np.clip(diagonal_terms, 0.0, None)  # because with only M=6 samples there are some negative values

    # Add σ² to each diagonal term
    eigenvalues = diagonal_terms + sigma_squared  # Shape: (N, D)

    # Calculate log determinant for each batch
    log_det = np.sum(np.log(eigenvalues), axis=1)  # Shape: (N,)

    # Calculate entropy
    entropy = 0.5 * log_det + 0.5 * D * (np.log(2 * np.pi) + 1)

    # If input was single batch, return scalar
    if len(mu_array.shape) == 2:
        return entropy[0]
    return entropy


def compute_generative_uncertainty(path, N, M, B, eu_type="entropy"):
    print(f"Loading samples from {M} models from {path}")

    #### 1) Load samples and convert to png

    for m in range(M):
        all_samples_m = []
        for b in range(B):
            all_samples_m.append(torch.load(f"{path}/{m}/{b}.pt"))
        all_samples_m = np.concatenate(all_samples_m, axis=0)

        if m == 0:
            np.save(f"{PATH}/{m}/all_imgs.npy", all_samples_m)

        npy2png(all_samples_m, f"{path}/{m}/imgs")

    #### 2) Compute clip features

    device = "cuda"
    model, preprocess = clip.load("ViT-B/32", device=device)

    for m in range(M):
        print(f"Processing model {m}")
        clip_vecs = []
        for i in range(N):
            image = preprocess(Image.open(f"{PATH}/{m}/imgs/{i:05d}.png")).unsqueeze(0).to(device)

            with torch.no_grad():
                clip_vecs.append(model.encode_image(image))

        clip_vecs = torch.concat(clip_vecs, dim=0)
        print(clip_vecs.shape)
        torch.save(clip_vecs, f"{path}/{m}/clip_features.pt")

    #### 3) Compute the entropy of the semantic likelihood

    features = []
    for model_id in range(M):
        path = f"{PATH}/{model_id}/clip_features.pt"
        features.append(torch.load(path))

    features = torch.stack(features, dim=0)
    features = np.transpose(features.cpu().numpy(), (1, 0, 2))
    print(features.shape)

    if eu_type == "entropy":
        eu = gaussian_entropy(features, sigma_squared=1e-3)
    else:
        raise ValueError(f"Unknown epistemic uncertainty type: {eu_type}")

    print(f"Saving: {PATH}/{eu_type}_clip.npy")
    np.save(f"{PATH}/{eu_type}_clip.npy", eu)


if __name__ == "__main__":

    PATH = "/nvmestore/mjazbec/diffusion/bayes_diff/exp_repo_clean/IMAGENET128/ddim_fixed_class10000_train%100_step50_S5_epi_unc_1234"
    M = 6
    N = 12032
    B = 47

    compute_generative_uncertainty(PATH, N, M, B, eu_type="entropy")
