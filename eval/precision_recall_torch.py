"""
Code adapted from: https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from time import time

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Precision and Recall Calculation")
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference features")
    parser.add_argument("--eval", type=str, required=True, help="Path to the evaluation features")
    parser.add_argument("--realism", type=str, required=False, default=None, help="Path for saving realism score")
    return parser.parse_args()


# ----------------------------------------------------------------------------


def batch_pairwise_distances(U, V):
    """Compute pairwise distances between two batches of feature vectors."""
    # Squared norms of each row in U and V.
    norm_u = torch.sum(U**2, dim=1)
    norm_v = torch.sum(V**2, dim=1)

    # norm_u as a column and norm_v as a row vectors.
    norm_u = norm_u.view(-1, 1)
    norm_v = norm_v.view(1, -1)

    # Pairwise squared Euclidean distances.
    D = torch.clamp(norm_u - 2 * torch.mm(U, V.T) + norm_v, min=0.0)

    return D


# ----------------------------------------------------------------------------


class DistanceBlock:
    """Provides multi-GPU support to calculate pairwise distances between two batches of feature vectors."""

    def __init__(self, num_features, num_gpus):
        self.num_features = num_features
        self.num_gpus = num_gpus

    def pairwise_distances(self, U, V):
        """Evaluate pairwise distances between two batches of feature vectors using multi-GPU support."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        U, V = U.to(device), V.to(device)

        if self.num_gpus > 1:
            V_split = torch.chunk(V, self.num_gpus, dim=0)
            distances_split = []
            for gpu_idx, V_part in enumerate(V_split):
                with torch.cuda.device(gpu_idx):
                    distances_split.append(batch_pairwise_distances(U, V_part))
            return torch.cat(distances_split, dim=1)
        else:
            return batch_pairwise_distances(U, V)


# ----------------------------------------------------------------------------


class ManifoldEstimator:
    """Estimates the manifold of given feature vectors."""

    def __init__(
        self,
        distance_block,
        features,
        row_batch_size=25000,
        col_batch_size=50000,
        nhood_sizes=[3],
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = torch.tensor(features, dtype=torch.float32)
        self._distance_block = distance_block

        num_images = features.shape[0]
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = self._ref_features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = self._ref_features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0 : end1 - begin1, begin2:end2] = (
                    self._distance_block.pairwise_distances(row_batch, col_batch).cpu().numpy()
                )

            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)[
                :, self.nhood_sizes
            ]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        eval_features = torch.tensor(eval_features, dtype=torch.float32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0 : end1 - begin1, begin2:end2] = (
                    self._distance_block.pairwise_distances(feature_batch, ref_batch).cpu().numpy()
                )

            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                self.D[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions


# ----------------------------------------------------------------------------


def knn_precision_recall_features(
    ref_features, eval_features, nhood_sizes=[3], row_batch_size=10000, col_batch_size=50000, num_gpus=1, realism=False
):
    """Calculates k-NN precision and recall for two sets of feature vectors."""
    state = dict()
    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold = ManifoldEstimator(distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes)
    eval_manifold = ManifoldEstimator(distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    print("Evaluating k-NN precision and recall with %i samples..." % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    if realism:
        precision, realism_score = ref_manifold.evaluate(eval_features, return_realism=True)
    else:
        precision = ref_manifold.evaluate(eval_features)
    state["precision"] = precision.mean(axis=0)

    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state["recall"] = recall.mean(axis=0)

    print("Evaluated k-NN precision and recall in: %gs" % (time() - start))

    if realism:
        return state, realism_score

    return state


def main():
    args = parse_args()
    ref_features_path = args.ref
    eval_features_path = args.eval
    realism_path = args.realism

    ref_features = torch.load(ref_features_path).cpu().numpy()
    eval_features = torch.load(eval_features_path).cpu().numpy()

    print("Reference features shape: %s" % str(ref_features.shape))
    print("Evaluation features shape: %s" % str(eval_features.shape))

    # Calculate k-NN precision and recall.
    if realism_path is not None:
        state, realism = knn_precision_recall_features(ref_features, eval_features, realism=True)
        if ".npy" in realism_path:
            np.save(realism_path, realism)
            print(f"Realism score saved to {realism_path}")
        else:
            np.save(f"{realism_path}/realism.npy", realism)
            print(f"Realism score saved to {realism_path}/realism.npy")
    else:
        state = knn_precision_recall_features(ref_features, eval_features, realism=False)

    # Print precision and recall.
    print("Precision: %s" % state["precision"])
    print("Recall: %s" % state["recall"])


if __name__ == "__main__":

    main()
