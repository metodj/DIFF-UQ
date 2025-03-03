# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Script for calculating Frechet Inception Distance (FID)."""

import os
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import dnnlib
from torch_utils import distributed as dist
from training import dataset
import clip
from torchvision import transforms
from PIL import Image

# ----------------------------------------------------------------------------


def calculate_inception_stats(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
    idx_path=None,
    fid_features=None,
    clip_features=None,
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if clip_features is not None:
        dist.print0("Loading CLIP model...")
        model, preprocess = clip.load("ViT-B/32", device=device)

    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0("Loading Inception-v3 model...")
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=True)
    # detector_kwargs = dict(return_features=False)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)

    if idx_path is not None:
        idx = np.load(idx_path)
        dataset_obj = torch.utils.data.Subset(dataset_obj, idx)

    if num_expected is not None and len(dataset_obj) < num_expected and idx_path is None:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but expected at least {num_expected}")
    if len(dataset_obj) < 2:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics")
    if idx_path is not None:
        assert len(dataset_obj) == len(idx), f"Found {len(dataset_obj)} images, but expected {len(idx)}"

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Accumulate statistics.
    dist.print0(f"Calculating statistics for {len(dataset_obj)} images...")
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    to_pil = transforms.ToPILImage()
    features_all = []
    for images, _labels in tqdm.tqdm(data_loader, unit="batch", disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])

        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)

        if clip_features is not None:
            pil_images = [to_pil(img) for img in images]
            images = [preprocess(img).unsqueeze(0).to(device) for img in pil_images]
            with torch.no_grad():
                for image in images:
                    features_all.append(model.encode_image(image))
        else:
            features_all.append(features)

        mu += features.sum(0)
        sigma += features.T @ features

    features_all = torch.cat(features_all, dim=0)
    print(features_all.shape)
    # save features
    if fid_features is not None:
        dist.print0(f"saving features to {fid_features}")
        if not fid_features.endswith(".pt"):
            fid_features = fid_features + "/fid_features.pt"
        torch.save(features_all, fid_features)
    elif clip_features is not None:
        dist.print0(f"saving features to {clip_features}")
        if not clip_features.endswith(".pt"):
            clip_features = clip_features + "/clip_features.pt"
        torch.save(features_all, clip_features)
    # dist.print0(f'saving features to {image_path}/fid_features.pt')
    # torch.save(features_all, f'{image_path}/fid_features_realism.pt')
    # torch.save(features_all, '/nvmestore/mjazbec/diffusion/edm/cifar_train_fid_features.pt')
    # torch.save(features_all, '/ivi/xfs/mjazbec/edm/datasets/cifar_test_fid_features.pt')
    # torch.save(features_all, '/ivi/xfs/mjazbec/edm/datasets/image_net_val_fid_features.pt')

    # Calculate grand totals.
    torch.distributed.all_reduce(mu)
    torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def extract_predictives(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    dist.print0("Loading Inception-v3 model...")
    detector_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"
    detector_kwargs = dict(return_features=False)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=(dist.get_rank() == 0)) as f:
        detector_net = pickle.load(f).to(device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)
    if num_expected is not None and len(dataset_obj) < num_expected:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but expected at least {num_expected}")
    if len(dataset_obj) < 2:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics")

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Accumulate statistics.
    dist.print0(f"Calculating statistics for {len(dataset_obj)} images...")
    features_all = []
    for images, _labels in tqdm.tqdm(data_loader, unit="batch", disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        features_all.append(features)

    features_all = torch.cat(features_all, dim=0)
    print(features_all.shape)
    # save features
    dist.print0(f"saving predictives to {image_path}/fid_predictives.pt")
    torch.save(features_all, f"{image_path}/fid_predictives.pt")


def clip_features(
    image_path,
    num_expected=None,
    seed=0,
    max_batch_size=64,
    num_workers=3,
    prefetch_factor=2,
    device=torch.device("cuda"),
    idx_path=None,
    fid_features=None,
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # List images.
    dist.print0(f'Loading images from "{image_path}"...')
    dataset_obj = dataset.ImageFolderDataset(path=image_path, max_size=num_expected, random_seed=seed)

    if idx_path is not None:
        idx = np.load(idx_path)
        dataset_obj = torch.utils.data.Subset(dataset_obj, idx)

    if num_expected is not None and len(dataset_obj) < num_expected and idx_path is None:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but expected at least {num_expected}")
    if len(dataset_obj) < 2:
        raise click.ClickException(f"Found {len(dataset_obj)} images, but need at least 2 to compute statistics")
    if idx_path is not None:
        assert len(dataset_obj) == len(idx), f"Found {len(dataset_obj)} images, but expected {len(idx)}"

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide images into batches.
    num_batches = ((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.arange(len(dataset_obj)).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(
        dataset_obj, batch_sampler=rank_batches, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    # Accumulate statistics.
    to_pil = transforms.ToPILImage()

    dist.print0(f"Calculating statistics for {len(dataset_obj)} images...")
    features_all = []
    for images, _labels in tqdm.tqdm(data_loader, unit="batch", disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        # features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        # features_all.append(features)

        pil_images = [to_pil(img) for img in images]
        images = [preprocess(img).unsqueeze(0).to(device) for img in pil_images]
        with torch.no_grad():
            for image in images:
                features_all.append(model.encode_image(image))

    features_all = torch.cat(features_all, dim=0)
    print(features_all.shape)
    # save features
    if fid_features is not None:
        dist.print0(f"saving features to {fid_features}")
        if not fid_features.endswith(".pt"):
            fid_features = fid_features + "/fid_features.pt"
        torch.save(features_all, fid_features)


# ----------------------------------------------------------------------------


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


# ----------------------------------------------------------------------------


@click.group()
def main():
    """Calculate Frechet Inception Distance (FID).

    Examples:

    \b
    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=1 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Calculate FID
    torchrun --standalone --nproc_per_node=1 fid.py calc --images=fid-tmp \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

    \b
    # Compute dataset reference statistics
    python fid.py ref --data=datasets/my-dataset.zip --dest=fid-refs/my-dataset.npz
    """


# ----------------------------------------------------------------------------


@main.command()
@click.option("--images", "image_path", help="Path to the images", metavar="PATH|ZIP", type=str, required=True)
@click.option("--ref", "ref_path", help="Dataset reference statistics ", metavar="NPZ|URL", type=str, required=True)
@click.option(
    "--num",
    "num_expected",
    help="Number of images to use",
    metavar="INT",
    type=click.IntRange(min=2),
    default=50000,
    show_default=True,
)
@click.option(
    "--seed", help="Random seed for selecting the images", metavar="INT", type=int, default=0, show_default=True
)
@click.option(
    "--batch", help="Maximum batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option("--idx_path", help="Path to subset indices.", metavar="PATH", type=str, required=False)
@click.option("--fid_features", help="Path to save FID features.", metavar="PATH", type=str, required=False)
@click.option("--clip_features", help="Path to save CLIP features.", metavar="PATH", type=str, required=False)
def calc(image_path, ref_path, num_expected, seed, batch, idx_path=None, fid_features=None, clip_features=None):
    """Calculate FID for a given set of images."""
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    dist.print0(f'Loading dataset reference statistics from "{ref_path}"...')
    ref = None
    if dist.get_rank() == 0:
        with dnnlib.util.open_url(ref_path) as f:
            ref = dict(np.load(f))

    mu, sigma = calculate_inception_stats(
        image_path=image_path,
        num_expected=num_expected,
        seed=seed,
        max_batch_size=batch,
        idx_path=idx_path,
        fid_features=fid_features,
        clip_features=clip_features,
    )
    dist.print0("Calculating FID...")
    if dist.get_rank() == 0:
        fid = calculate_fid_from_inception_stats(mu, sigma, ref["mu"], ref["sigma"])
        print(f"{fid:g}")
    torch.distributed.barrier()


# ----------------------------------------------------------------------------


@main.command()
@click.option("--images", "image_path", help="Path to the images", metavar="PATH|ZIP", type=str, required=True)
@click.option(
    "--num",
    "num_expected",
    help="Number of images to use",
    metavar="INT",
    type=click.IntRange(min=2),
    default=50000,
    show_default=True,
)
@click.option(
    "--seed", help="Random seed for selecting the images", metavar="INT", type=int, default=0, show_default=True
)
@click.option(
    "--batch", help="Maximum batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True
)
def preds(image_path, num_expected, seed, batch):
    """Extract inception-net predictives from images."""
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    extract_predictives(image_path=image_path, num_expected=num_expected, seed=seed, max_batch_size=batch)
    torch.distributed.barrier()


# ----------------------------------------------------------------------------


@main.command()
@click.option("--data", "dataset_path", help="Path to the dataset", metavar="PATH|ZIP", type=str, required=True)
@click.option("--dest", "dest_path", help="Destination .npz file", metavar="NPZ", type=str, required=True)
@click.option(
    "--batch", help="Maximum batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option("--fid_features", help="Path to save FID features.", metavar="PATH", type=str, required=False)
def ref(dataset_path, dest_path, batch, fid_features=None):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    mu, sigma = calculate_inception_stats(image_path=dataset_path, max_batch_size=batch, fid_features=fid_features)
    dist.print0(f'Saving dataset reference statistics to "{dest_path}"...')
    if dist.get_rank() == 0:
        if os.path.dirname(dest_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        np.savez(dest_path, mu=mu, sigma=sigma)

    torch.distributed.barrier()
    dist.print0("Done.")


# ----------------------------------------------------------------------------


@main.command()
@click.option("--data", "dataset_path", help="Path to the dataset", metavar="PATH|ZIP", type=str, required=True)
@click.option(
    "--batch", help="Maximum batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option("--fid_features", help="Path to save FID features.", metavar="PATH", type=str, required=False)
def fclip(dataset_path, batch, fid_features=None):
    """Calculate dataset reference statistics needed by 'calc'."""
    torch.multiprocessing.set_start_method("spawn")
    dist.init()

    clip_features(image_path=dataset_path, max_batch_size=batch, fid_features=fid_features)

    torch.distributed.barrier()
    dist.print0("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
