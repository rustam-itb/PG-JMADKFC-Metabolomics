# -*- coding: utf-8 -*-
"""Isolated JMADKFC revision experiment worker.
Based on the supplied original submission notebook and revised notebook.
One invocation evaluates exactly one method × setting × seed.
No research results are silently skipped; successful runs are written atomically.
"""
from __future__ import annotations
import os
# Limit library thread pools BEFORE importing numpy/scipy/sklearn/torch.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_k] = "1"
from pathlib import Path
import re, sys, gc, json, argparse, traceback, time, hashlib, random, platform, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.cluster import KMeans

# Defaults are changed only by CLI arguments.
DATA_DIR = Path(__file__).resolve().parent
DEVICE = 'cpu'
ALPHA = 2.0
pretrain_epochs = 6
joint_epochs = 8
ABLATIONS = ['Full','NoMaskInput','GlobalKernel','NoClusterGradient']
ORIGINAL_BASELINES = ['FCM-Mean','FCM-Median','FCM-KNN','FCM-MICE','FCM-PDS','FCM-OCS',
                      'KFCM','VKFCM-K-LP-PDS','VKFCM-K-LP-OCS','DFCM','DEC']
baseline_names = ORIGINAL_BASELINES
DATASET_NAMES = ['MTBLS161','MTBLS404']
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass



# ========== Imported revised notebook Cell 4 ==========
class FuzzyCMeans:
    """
    Manual implementation of standard Fuzzy C-Means (FCM).

    Objective:
        J = sum_i sum_k u_ik^alpha ||x_i - v_k||^2

    Notes:
        - This model is fully unsupervised.
        - Class labels are not used during fitting.
        - Input data must be complete.
    """

    def __init__(self, n_clusters, alpha=2.0, max_iter=300, tol=1e-5,
                 random_state=None, eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers(self, X, U):
        U_alpha = U ** self.alpha
        numerator = U_alpha.T @ X
        denominator = U_alpha.sum(axis=0)[:, None] + self.eps
        return numerator / denominator

    def _squared_distances(self, X, centers):
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _update_membership(self, distances):
        distances = np.asarray(distances, dtype=float)
        U = np.zeros_like(distances, dtype=float)

        zero_mask = distances <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D = np.maximum(distances[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D[:, :, None] / D[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _objective(self, U, distances):
        return float(np.sum((U ** self.alpha) * distances))

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if np.isnan(X).any():
            raise ValueError("Standard FCM requires complete data.")

        n_samples = X.shape[0]
        U = self._initialize_membership(n_samples)

        for _ in range(self.max_iter):
            U_old = U.copy()
            centers = self._update_centers(X, U)
            distances = self._squared_distances(X, centers)
            U = self._update_membership(distances)

            obj = self._objective(U, distances)
            self.objective_history_.append(obj)

            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.centers_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X, dtype=float)
        distances = self._squared_distances(X, self.centers_)
        U = self._update_membership(distances)
        return np.argmax(U, axis=1)

def validate_complete_matrix(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix.")
    if np.isnan(X).any():
        raise ValueError("X must be complete before artificial missingness is introduced.")

    return X


def number_of_missing_entries(X, missing_rate):
    if not (0.0 < missing_rate < 1.0):
        raise ValueError("missing_rate must be between 0 and 1.")
    return int(round(X.size * missing_rate))


def apply_mask(X, mask):
    """
    Apply binary mask to complete matrix.

    mask:
        1 = observed
        0 = artificially missing
    """
    X = validate_complete_matrix(X)
    mask = np.asarray(mask, dtype=int)

    if mask.shape != X.shape:
        raise ValueError("mask must have the same shape as X.")

    X_missing = X.copy()
    X_missing[mask == 0] = np.nan
    return X_missing


def rank_normalize_columns(X):
    """Rank-normalize each feature column to [0, 1]."""
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    R = np.zeros_like(X, dtype=float)

    for j in range(d):
        order = np.argsort(X[:, j])
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        R[:, j] = ranks / max(n - 1, 1)

    return R


def sample_weighted_entries(weights, n_missing, rng):
    """Sample flat matrix indices without replacement using nonnegative weights."""
    w = np.asarray(weights, dtype=float).ravel()
    w = np.maximum(w, 0.0)

    if np.all(w == 0):
        p = None
    else:
        p = w / w.sum()

    return rng.choice(w.size, size=n_missing, replace=False, p=p)


def generate_mcar_mask(X, missing_rate, random_state=None):
    """Generate MCAR mask by uniformly removing entries at random."""
    X = validate_complete_matrix(X)
    rng = np.random.default_rng(random_state)

    n_missing = number_of_missing_entries(X, missing_rate)
    flat_idx = rng.choice(X.size, size=n_missing, replace=False)

    mask = np.ones(X.size, dtype=int)
    mask[flat_idx] = 0
    return mask.reshape(X.shape)


def generate_mar_mask(X, missing_rate, observed_fraction=0.30, target_fraction=0.70,
                      gamma=3.0, random_state=None):
    """
    Generate MAR mask.

    Missingness depends on observed auxiliary metabolites, not directly on the
    value being masked.
    """
    X = validate_complete_matrix(X)
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    if d < 2:
        raise ValueError("MAR simulation requires at least two features.")

    n_missing = number_of_missing_entries(X, missing_rate)

    n_aux = max(1, int(round(d * observed_fraction)))
    n_target = max(1, int(round(d * target_fraction)))

    cols = rng.permutation(d)
    aux_cols = cols[:n_aux]
    target_cols = cols[n_aux:n_aux + n_target]

    if target_cols.size == 0:
        target_cols = cols[n_aux:]

    R = rank_normalize_columns(X)
    a = R[:, aux_cols].mean(axis=1)

    weights = np.zeros_like(X, dtype=float)
    weights[:, target_cols] = np.exp(gamma * a)[:, None]

    flat_idx = sample_weighted_entries(weights, n_missing, rng)

    mask = np.ones(X.size, dtype=int)
    mask[flat_idx] = 0
    return mask.reshape(X.shape)


def generate_mnar_mask(X, missing_rate, gamma=3.0, low_intensity=True,
                       random_state=None):
    """
    Generate MNAR mask.

    Default setting:
        Low-intensity values have higher probability of being missing.
    """
    X = validate_complete_matrix(X)
    rng = np.random.default_rng(random_state)

    n_missing = number_of_missing_entries(X, missing_rate)
    R = rank_normalize_columns(X)

    if low_intensity:
        weights = np.exp(-gamma * R)
    else:
        weights = np.exp(gamma * R)

    flat_idx = sample_weighted_entries(weights, n_missing, rng)

    mask = np.ones(X.size, dtype=int)
    mask[flat_idx] = 0
    return mask.reshape(X.shape)


def generate_missingness(X, mechanism, missing_rate, random_state=None, **kwargs):
    """
    Generate incomplete matrix and binary mask.

    mechanism:
        "MCAR", "MAR", or "MNAR"

    Returns:
        X_missing : matrix with np.nan at artificially missing positions
        mask      : binary mask, 1 observed and 0 missing
    """
    mechanism = mechanism.upper()

    if mechanism == "MCAR":
        mask = generate_mcar_mask(X, missing_rate, random_state=random_state)
    elif mechanism == "MAR":
        mask = generate_mar_mask(X, missing_rate, random_state=random_state, **kwargs)
    elif mechanism == "MNAR":
        mask = generate_mnar_mask(X, missing_rate, random_state=random_state, **kwargs)
    else:
        raise ValueError("mechanism must be one of: MCAR, MAR, MNAR.")

    X_missing = apply_mask(X, mask)
    return X_missing, mask

def mean_impute(X_missing):
    X = np.asarray(X_missing, dtype=float).copy()
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X


def median_impute(X_missing):
    X = np.asarray(X_missing, dtype=float).copy()
    col_medians = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_medians, inds[1])
    return X


def knn_impute(X_missing, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors, weights="uniform")
    return imputer.fit_transform(X_missing)


def mice_impute(X_missing, random_state=None, max_iter=10):
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=False,
        initial_strategy="mean"
    )
    return imputer.fit_transform(X_missing)


class ImputationFCMBaseline:
    """
    Pipeline:
        incomplete matrix -> imputation -> manual FCM
    """

    def __init__(self, imputation, n_clusters, alpha=2.0, fcm_max_iter=300,
                 fcm_tol=1e-5, random_state=None, **imputer_kwargs):
        self.imputation = imputation.lower()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.fcm_max_iter = fcm_max_iter
        self.fcm_tol = fcm_tol
        self.random_state = random_state
        self.imputer_kwargs = imputer_kwargs

        self.X_completed_ = None
        self.fcm_ = None
        self.labels_ = None
        self.membership_ = None

    def _impute(self, X_missing):
        if self.imputation == "mean":
            return mean_impute(X_missing)

        if self.imputation == "median":
            return median_impute(X_missing)

        if self.imputation == "knn":
            return knn_impute(X_missing, **self.imputer_kwargs)

        if self.imputation == "mice":
            return mice_impute(
                X_missing,
                random_state=self.random_state,
                **self.imputer_kwargs
            )

        raise ValueError("imputation must be one of: mean, median, knn, mice.")

    def fit(self, X_missing):
        self.X_completed_ = self._impute(X_missing)

        self.fcm_ = FuzzyCMeans(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            max_iter=self.fcm_max_iter,
            tol=self.fcm_tol,
            random_state=self.random_state
        )
        self.fcm_.fit(self.X_completed_)

        self.labels_ = self.fcm_.labels_
        self.membership_ = self.fcm_.membership_
        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_

def clustering_accuracy(y_true, y_pred):
    """
    Clustering accuracy after optimal label matching using the Hungarian algorithm.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    true_map = {label: idx for idx, label in enumerate(true_labels)}
    pred_map = {label: idx for idx, label in enumerate(pred_labels)}

    y_true_idx = np.array([true_map[y] for y in y_true], dtype=int)
    y_pred_idx = np.array([pred_map[y] for y in y_pred], dtype=int)

    size = max(len(true_labels), len(pred_labels))
    contingency = np.zeros((size, size), dtype=int)

    for t, p in zip(y_true_idx, y_pred_idx):
        contingency[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def external_clustering_metrics(y_true, y_pred):
    """
    Compute ARI, NMI, and ACC.

    Labels are used only for post-clustering validation.
    """
    return {
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ACC": clustering_accuracy(y_true, y_pred)
    }


def rmse_on_masked_entries(X_full, X_reconstructed, mask):
    """
    RMSE computed only on artificially masked entries.

    mask:
        1 = observed
        0 = artificially missing
    """
    X_full = np.asarray(X_full, dtype=float)
    X_reconstructed = np.asarray(X_reconstructed, dtype=float)
    mask = np.asarray(mask, dtype=int)

    idx = mask == 0

    if idx.sum() == 0:
        raise ValueError("No masked entries found.")

    return float(np.sqrt(np.mean((X_full[idx] - X_reconstructed[idx]) ** 2)))


def mae_on_masked_entries(X_full, X_reconstructed, mask):
    """
    MAE computed only on artificially masked entries.

    mask:
        1 = observed
        0 = artificially missing
    """
    X_full = np.asarray(X_full, dtype=float)
    X_reconstructed = np.asarray(X_reconstructed, dtype=float)
    mask = np.asarray(mask, dtype=int)

    idx = mask == 0

    if idx.sum() == 0:
        raise ValueError("No masked entries found.")

    return float(np.mean(np.abs(X_full[idx] - X_reconstructed[idx])))


def degradation_score(reference_value, missing_value):
    """
    Performance degradation from complete-data reference to missing-data setting.
    """
    return float(reference_value - missing_value)


# ========== Imported revised notebook Cell 6 ==========
# =========================================================
# Utility functions for incomplete-data FCM
# =========================================================

def infer_observation_mask(X_missing):
    """
    Infer binary observation mask from a matrix containing np.nan.

    Returns:
        mask: 1 for observed entries, 0 for missing entries
    """
    X_missing = np.asarray(X_missing, dtype=float)
    return (~np.isnan(X_missing)).astype(int)


def validate_incomplete_matrix(X_missing, mask=None):
    """
    Validate incomplete matrix and observation mask.
    """
    X_missing = np.asarray(X_missing, dtype=float)

    if X_missing.ndim != 2:
        raise ValueError("X_missing must be a 2D matrix.")

    if mask is None:
        mask = infer_observation_mask(X_missing)
    else:
        mask = np.asarray(mask, dtype=int)

    if mask.shape != X_missing.shape:
        raise ValueError("mask must have the same shape as X_missing.")

    if not np.array_equal(mask, (~np.isnan(X_missing)).astype(int)):
        # This warning is useful when X_missing uses filled values.
        # In this notebook, X_missing is expected to contain np.nan at missing positions.
        pass

    observed_counts = mask.sum(axis=1)
    if np.any(observed_counts == 0):
        raise ValueError("Each sample must have at least one observed feature.")

    return X_missing, mask


def column_observed_means(X_missing, mask, eps=1e-10):
    """
    Compute feature-wise means using observed values only.
    """
    X_zero = np.nan_to_num(X_missing, nan=0.0)
    num = np.sum(mask * X_zero, axis=0)
    den = np.sum(mask, axis=0) + eps
    return num / den


def initial_completion(X_missing, mask, strategy="mean"):
    """
    Initial completion for OCS.

    strategy:
        "mean" or "median"
    """
    X_missing = np.asarray(X_missing, dtype=float)
    X_completed = X_missing.copy()

    if strategy == "mean":
        values = np.nanmean(X_missing, axis=0)
    elif strategy == "median":
        values = np.nanmedian(X_missing, axis=0)
    else:
        raise ValueError("strategy must be 'mean' or 'median'.")

    inds = np.where(mask == 0)
    X_completed[inds] = np.take(values, inds[1])

    if np.isnan(X_completed).any():
        raise ValueError("Initial completion still contains NaN. Check features with all missing values.")

    return X_completed

# =========================================================
# FCM-PDS: Fuzzy C-Means with Partial Distance Strategy
# =========================================================

class FCM_PDS:
    """
    Fuzzy C-Means with Partial Distance Strategy (PDS).

    Main idea:
        Distance between sample i and prototype k is computed only over
        observed features of sample i, then scaled by d / |O_i|.

    PDS distance:
        d_ik = (d / |O_i|) * sum_{j in O_i} (x_ij - v_kj)^2

    Prototype update:
        v_kj = sum_i u_ik^alpha m_ij x_ij / sum_i u_ik^alpha m_ij

    Notes:
        - Works directly with X_missing containing np.nan.
        - Class labels are not used.
    """

    def __init__(self, n_clusters, alpha=2.0, max_iter=300, tol=1e-5,
                 random_state=None, eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers(self, X_missing, mask, U):
        X_zero = np.nan_to_num(X_missing, nan=0.0)
        U_alpha = U ** self.alpha

        n, d = X_missing.shape
        centers = np.zeros((self.n_clusters, d), dtype=float)

        global_means = column_observed_means(X_missing, mask, eps=self.eps)

        for k in range(self.n_clusters):
            weights = U_alpha[:, k][:, None] * mask
            numerator = np.sum(weights * X_zero, axis=0)
            denominator = np.sum(weights, axis=0)

            centers[k] = numerator / (denominator + self.eps)

            # If a cluster-feature denominator is nearly zero, use observed global mean
            bad = denominator <= self.eps
            centers[k, bad] = global_means[bad]

        return centers

    def _partial_squared_distances(self, X_missing, mask, centers):
        X_zero = np.nan_to_num(X_missing, nan=0.0)
        n, d = X_missing.shape

        observed_counts = mask.sum(axis=1)
        scale = d / (observed_counts + self.eps)

        distances = np.zeros((n, self.n_clusters), dtype=float)

        for k in range(self.n_clusters):
            diff2 = ((X_zero - centers[k]) ** 2) * mask
            distances[:, k] = scale * np.sum(diff2, axis=1)

        return distances

    def _update_membership(self, distances):
        distances = np.asarray(distances, dtype=float)
        U = np.zeros_like(distances, dtype=float)

        zero_mask = distances <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D = np.maximum(distances[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D[:, :, None] / D[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _objective(self, U, distances):
        return float(np.sum((U ** self.alpha) * distances))

    def fit(self, X_missing, mask=None):
        X_missing, mask = validate_incomplete_matrix(X_missing, mask)
        n_samples = X_missing.shape[0]

        U = self._initialize_membership(n_samples)

        for _ in range(self.max_iter):
            U_old = U.copy()

            centers = self._update_centers(X_missing, mask, U)
            distances = self._partial_squared_distances(X_missing, mask, centers)
            U = self._update_membership(distances)

            obj = self._objective(U, distances)
            self.objective_history_.append(obj)

            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

# =========================================================
# FCM-OCS: Fuzzy C-Means with Optimal Completion Strategy
# =========================================================

class FCM_OCS:
    """
    Fuzzy C-Means with Optimal Completion Strategy (OCS).

    Main idea:
        Missing entries are optimally completed during clustering by minimizing
        the FCM objective with respect to missing values.

    Missing-value update:
        x_hat_ij = sum_k u_ik^alpha v_kj / sum_k u_ik^alpha, for missing entries.

    Then standard FCM updates are applied to the completed matrix.

    Notes:
        - Works directly with X_missing containing np.nan.
        - Class labels are not used.
    """

    def __init__(self, n_clusters, alpha=2.0, max_iter=300, tol=1e-5,
                 random_state=None, init_completion="mean", eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_completion = init_completion
        self.eps = eps

        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.X_completed_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers(self, X_completed, U):
        U_alpha = U ** self.alpha
        numerator = U_alpha.T @ X_completed
        denominator = U_alpha.sum(axis=0)[:, None] + self.eps
        return numerator / denominator

    def _squared_distances(self, X_completed, centers):
        diff = X_completed[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _update_membership(self, distances):
        distances = np.asarray(distances, dtype=float)
        U = np.zeros_like(distances, dtype=float)

        zero_mask = distances <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D = np.maximum(distances[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D[:, :, None] / D[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_missing_entries(self, X_missing, mask, U, centers):
        """
        OCS update for missing entries:
            x_hat_ij = sum_k u_ik^alpha v_kj / sum_k u_ik^alpha
        """
        X_completed = np.asarray(X_missing, dtype=float).copy()
        U_alpha = U ** self.alpha

        numerator = U_alpha @ centers
        denominator = U_alpha.sum(axis=1, keepdims=True) + self.eps
        X_hat = numerator / denominator

        X_completed[mask == 0] = X_hat[mask == 0]

        if np.isnan(X_completed).any():
            raise ValueError("OCS completion produced NaN values.")

        return X_completed

    def _objective(self, U, distances):
        return float(np.sum((U ** self.alpha) * distances))

    def fit(self, X_missing, mask=None):
        X_missing, mask = validate_incomplete_matrix(X_missing, mask)
        n_samples = X_missing.shape[0]

        U = self._initialize_membership(n_samples)
        X_completed = initial_completion(X_missing, mask, strategy=self.init_completion)

        for _ in range(self.max_iter):
            U_old = U.copy()
            X_old = X_completed.copy()

            centers = self._update_centers(X_completed, U)
            X_completed = self._update_missing_entries(X_missing, mask, U, centers)

            centers = self._update_centers(X_completed, U)
            distances = self._squared_distances(X_completed, centers)
            U = self._update_membership(distances)

            obj = self._objective(U, distances)
            self.objective_history_.append(obj)

            delta_U = np.linalg.norm(U - U_old)
            delta_X = np.linalg.norm(X_completed - X_old) / (np.linalg.norm(X_old) + self.eps)

            if delta_U < self.tol and delta_X < self.tol:
                break

        self.centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        self.X_completed_ = X_completed
        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

# =========================================================
# Unified wrapper for incomplete-data FCM baselines
# =========================================================

class IncompleteFCMBaseline:
    """
    Unified interface for incomplete-data FCM baselines.

    strategy:
        "pds" -> FCM with Partial Distance Strategy
        "ocs" -> FCM with Optimal Completion Strategy
    """

    def __init__(self, strategy, n_clusters, alpha=2.0, max_iter=300,
                 tol=1e-5, random_state=None, **kwargs):
        self.strategy = strategy.lower()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_ = None
        self.labels_ = None
        self.membership_ = None
        self.centers_ = None
        self.X_completed_ = None

    def fit(self, X_missing, mask=None):
        if self.strategy == "pds":
            self.model_ = FCM_PDS(
                n_clusters=self.n_clusters,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )

        elif self.strategy == "ocs":
            self.model_ = FCM_OCS(
                n_clusters=self.n_clusters,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                **self.kwargs
            )

        else:
            raise ValueError("strategy must be either 'pds' or 'ocs'.")

        self.model_.fit(X_missing, mask)

        self.labels_ = self.model_.labels_
        self.membership_ = self.model_.membership_
        self.centers_ = self.model_.centers_

        if hasattr(self.model_, "X_completed_"):
            self.X_completed_ = self.model_.X_completed_

        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_


# ========== Imported revised notebook Cell 8 ==========
# =========================================================
# KFCM: Kernel-based Fuzzy C-Means
# =========================================================

class KernelFuzzyCMeans:
    """
    Kernel-based Fuzzy C-Means using Gaussian kernel-induced dissimilarity.

    Kernel similarity:
        K(x_i, v_k) = exp(-||x_i - v_k||^2 / (2 sigma^2))

    Kernel-induced dissimilarity:
        D_ik = 1 - K(x_i, v_k)

    Objective:
        J = sum_i sum_k u_ik^alpha D_ik

    Prototype update:
        v_k = sum_i u_ik^alpha K(x_i,v_k) x_i /
              sum_i u_ik^alpha K(x_i,v_k)

    Notes:
        - Input X must be complete.
        - This implementation uses a global Gaussian bandwidth sigma.
        - If sigma is None, the median pairwise distance heuristic is used.
    """

    def __init__(self, n_clusters, alpha=2.0, sigma=None, max_iter=300,
                 tol=1e-5, random_state=None, eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

        self.sigma_ = None
        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _standard_fcm_centers(self, X, U):
        U_alpha = U ** self.alpha
        numerator = U_alpha.T @ X
        denominator = U_alpha.sum(axis=0)[:, None] + self.eps
        return numerator / denominator

    def _squared_distances(self, X, centers):
        diff = X[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _estimate_sigma(self, X):
        """
        Estimate global Gaussian bandwidth using median pairwise distance.
        """
        X = np.asarray(X, dtype=float)
        n = X.shape[0]

        if n < 2:
            return 1.0

        diff = X[:, None, :] - X[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        upper = dist2[np.triu_indices(n, k=1)]
        positive = upper[upper > self.eps]

        if positive.size == 0:
            return 1.0

        sigma = np.sqrt(np.median(positive))
        return float(max(sigma, self.eps))

    def _kernel_similarity(self, distances):
        """
        Gaussian kernel similarity from squared Euclidean distances.
        """
        return np.exp(-distances / (2.0 * self.sigma_ ** 2 + self.eps))

    def _kernel_dissimilarity(self, X, centers):
        distances = self._squared_distances(X, centers)
        K = self._kernel_similarity(distances)
        D = 1.0 - K
        return D, K

    def _update_membership(self, dissimilarities):
        D = np.asarray(dissimilarities, dtype=float)
        U = np.zeros_like(D, dtype=float)

        zero_mask = D <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D_safe = np.maximum(D[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D_safe[:, :, None] / D_safe[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers_kernel(self, X, U, centers_old):
        """
        Fixed-point prototype update for Gaussian KFCM.
        """
        distances = self._squared_distances(X, centers_old)
        K = self._kernel_similarity(distances)
        U_alpha = U ** self.alpha

        centers_new = np.zeros_like(centers_old)

        for k in range(self.n_clusters):
            weights = U_alpha[:, k] * K[:, k]
            numerator = weights @ X
            denominator = weights.sum() + self.eps
            centers_new[k] = numerator / denominator

        return centers_new

    def _objective(self, U, D):
        return float(np.sum((U ** self.alpha) * D))

    def fit(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if np.isnan(X).any():
            raise ValueError("KFCM requires complete data. Apply imputation first.")

        n_samples = X.shape[0]
        U = self._initialize_membership(n_samples)

        if self.sigma is None:
            self.sigma_ = self._estimate_sigma(X)
        else:
            if self.sigma <= 0:
                raise ValueError("sigma must be positive.")
            self.sigma_ = float(self.sigma)

        centers = self._standard_fcm_centers(X, U)

        for _ in range(self.max_iter):
            U_old = U.copy()
            centers_old = centers.copy()

            centers = self._update_centers_kernel(X, U, centers_old)
            D, K = self._kernel_dissimilarity(X, centers)
            U = self._update_membership(D)

            obj = self._objective(U, D)
            self.objective_history_.append(obj)

            delta_U = np.linalg.norm(U - U_old)
            delta_V = np.linalg.norm(centers - centers_old) / (np.linalg.norm(centers_old) + self.eps)

            if delta_U < self.tol and delta_V < self.tol:
                break

        self.centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.centers_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X = np.asarray(X, dtype=float)

        if np.isnan(X).any():
            raise ValueError("KFCM prediction requires complete data.")

        D, K = self._kernel_dissimilarity(X, self.centers_)
        U = self._update_membership(D)
        return np.argmax(U, axis=1)

# =========================================================
# KFCM baseline wrapper:
#   incomplete matrix -> imputation -> KFCM
# =========================================================

class KFCMBaseline:
    """
    Pipeline for Kernel Fuzzy C-Means under incomplete data:

        X_missing -> imputation -> Kernel Fuzzy C-Means

    imputation:
        "mean", "median", "knn", or "mice"
    """

    def __init__(self, imputation="mean", n_clusters=2, alpha=2.0, sigma=None,
                 max_iter=300, tol=1e-5, random_state=None, **imputer_kwargs):
        self.imputation = imputation.lower()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.imputer_kwargs = imputer_kwargs

        self.X_completed_ = None
        self.model_ = None
        self.labels_ = None
        self.membership_ = None
        self.centers_ = None
        self.sigma_ = None

    def _impute(self, X_missing):
        if self.imputation == "mean":
            return mean_impute(X_missing)

        if self.imputation == "median":
            return median_impute(X_missing)

        if self.imputation == "knn":
            return knn_impute(X_missing, **self.imputer_kwargs)

        if self.imputation == "mice":
            return mice_impute(
                X_missing,
                random_state=self.random_state,
                **self.imputer_kwargs
            )

        raise ValueError("imputation must be one of: mean, median, knn, mice.")

    def fit(self, X_missing):
        self.X_completed_ = self._impute(X_missing)

        self.model_ = KernelFuzzyCMeans(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            sigma=self.sigma,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

        self.model_.fit(self.X_completed_)

        self.labels_ = self.model_.labels_
        self.membership_ = self.model_.membership_
        self.centers_ = self.model_.centers_
        self.sigma_ = self.model_.sigma_

        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_

# =========================================================
# Utility functions for VKFCM-K-LP
# =========================================================

def estimate_feature_bandwidths(X, eps=1e-10):
    """
    Estimate one Gaussian bandwidth per feature.

    The bandwidth for feature j is estimated using the observed standard
    deviation. This is stable for metabolomics data after z-score scaling.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Complete matrix or currently completed matrix.
    """
    X = np.asarray(X, dtype=float)

    sigma = np.nanstd(X, axis=0)
    sigma = np.where(sigma <= eps, 1.0, sigma)

    return sigma + eps


def featurewise_gaussian_dissimilarity(X, centers, sigma, eps=1e-10):
    """
    Compute feature-wise Gaussian kernel dissimilarities.

    Returns
    -------
    D_feat : ndarray, shape (n_samples, n_clusters, n_features)
        D_ikj = 1 - exp(-(x_ij - v_kj)^2 / (2 sigma_j^2))
    K_feat : ndarray, shape (n_samples, n_clusters, n_features)
        K_ikj = exp(-(x_ij - v_kj)^2 / (2 sigma_j^2))
    """
    X = np.asarray(X, dtype=float)
    centers = np.asarray(centers, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    diff2 = (X[:, None, :] - centers[None, :, :]) ** 2
    K_feat = np.exp(-diff2 / (2.0 * sigma[None, None, :] ** 2 + eps))
    D_feat = 1.0 - K_feat

    return D_feat, K_feat


def initialize_feature_weights(n_clusters, n_features):
    """
    Initialize cluster-wise feature weights with product-to-one constraint.

    If all weights are 1, product_j lambda_kj = 1 for every cluster k.
    """
    return np.ones((n_clusters, n_features), dtype=float)


def update_product_constraint_weights(U, D_feat, mask=None, alpha=2.0, eps=1e-10):
    """
    Update cluster-wise variable weights under product-to-one constraint.

    The update is based on:
        lambda_kj = geometric_mean(A_k*) / A_kj

    where:
        A_kj = sum_i u_ik^alpha D_ikj

    If mask is given, only observed feature values contribute.
    """
    U = np.asarray(U, dtype=float)
    D_feat = np.asarray(D_feat, dtype=float)

    n_samples, n_clusters, n_features = D_feat.shape
    U_alpha = U ** alpha

    A = np.zeros((n_clusters, n_features), dtype=float)

    if mask is None:
        for k in range(n_clusters):
            A[k] = np.sum(U_alpha[:, k][:, None] * D_feat[:, k, :], axis=0)
    else:
        mask = np.asarray(mask, dtype=int)
        for k in range(n_clusters):
            A[k] = np.sum(U_alpha[:, k][:, None] * mask * D_feat[:, k, :], axis=0)

    A = np.maximum(A, eps)

    log_gm = np.mean(np.log(A), axis=1, keepdims=True)
    gm = np.exp(log_gm)

    weights = gm / A

    # Numerical correction to ensure product-to-one approximately.
    weights = np.maximum(weights, eps)
    log_prod = np.mean(np.log(weights), axis=1, keepdims=True)
    weights = weights / np.exp(log_prod)

    return weights


def aggregate_weighted_kernel_distance(D_feat, weights, mask=None, eps=1e-10):
    """
    Aggregate feature-wise kernel dissimilarities using cluster-wise weights.

    For complete data:
        D_ik = sum_j lambda_kj D_ikj

    For PDS:
        D_ik = (d / |O_i|) sum_j m_ij lambda_kj D_ikj
    """
    D_feat = np.asarray(D_feat, dtype=float)
    weights = np.asarray(weights, dtype=float)

    n_samples, n_clusters, n_features = D_feat.shape

    weighted = D_feat * weights[None, :, :]

    if mask is None:
        D = np.sum(weighted, axis=2)
    else:
        mask = np.asarray(mask, dtype=int)
        observed_counts = mask.sum(axis=1)
        scale = n_features / (observed_counts + eps)

        D = np.zeros((n_samples, n_clusters), dtype=float)

        for k in range(n_clusters):
            D[:, k] = scale * np.sum(weighted[:, k, :] * mask, axis=1)

    return np.maximum(D, eps)

# =========================================================
# VKFCM-K-LP-PDS
# Variable-weighted Kernel FCM with Local adaptive distances
# using Partial Distance Strategy
# =========================================================

class VKFCM_K_LP_PDS:
    """
    VKFCM-K-LP with Partial Distance Strategy.

    Main components:
        - Feature-wise Gaussian kernel dissimilarity
        - Cluster-wise feature weights with product-to-one constraint
        - Partial distance strategy for missing entries
        - Fuzzy membership update

    PDS distance:
        D_ik = (d / |O_i|) sum_j m_ij lambda_kj D_ikj

    Feature weights:
        lambda_kj = GM(A_k*) / A_kj
        A_kj = sum_i u_ik^alpha m_ij D_ikj
    """

    def __init__(self, n_clusters, alpha=2.0, max_iter=300, tol=1e-5,
                 random_state=None, eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eps = eps

        self.centers_ = None
        self.weights_ = None
        self.sigma_ = None
        self.membership_ = None
        self.labels_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _initialize_centers(self, X_missing, mask, U):
        """
        Initialize centers using observed-value weighted means.
        """
        X_zero = np.nan_to_num(X_missing, nan=0.0)
        U_alpha = U ** self.alpha
        n_samples, n_features = X_missing.shape

        centers = np.zeros((self.n_clusters, n_features), dtype=float)
        global_means = column_observed_means(X_missing, mask, eps=self.eps)

        for k in range(self.n_clusters):
            w = U_alpha[:, k][:, None] * mask
            numerator = np.sum(w * X_zero, axis=0)
            denominator = np.sum(w, axis=0)
            centers[k] = numerator / (denominator + self.eps)

            bad = denominator <= self.eps
            centers[k, bad] = global_means[bad]

        return centers

    def _update_membership(self, D):
        D = np.asarray(D, dtype=float)
        U = np.zeros_like(D, dtype=float)

        zero_mask = D <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D_safe = np.maximum(D[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D_safe[:, :, None] / D_safe[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers(self, X_missing, mask, U, centers_old, weights, sigma):
        """
        Kernel-weighted prototype update using observed entries only.

        v_kj = sum_i u_ik^alpha m_ij K_ikj x_ij /
               sum_i u_ik^alpha m_ij K_ikj
        """
        X_zero = np.nan_to_num(X_missing, nan=0.0)
        _, K_feat = featurewise_gaussian_dissimilarity(
            X_zero, centers_old, sigma, eps=self.eps
        )

        U_alpha = U ** self.alpha
        n_samples, n_features = X_missing.shape

        centers_new = np.zeros_like(centers_old)
        global_means = column_observed_means(X_missing, mask, eps=self.eps)

        for k in range(self.n_clusters):
            # lambda_kj is constant over i for feature j, so it does not change
            # the fixed-point ratio, but we keep it for numerical consistency.
            w = U_alpha[:, k][:, None] * mask * K_feat[:, k, :] * weights[k][None, :]
            numerator = np.sum(w * X_zero, axis=0)
            denominator = np.sum(w, axis=0)

            centers_new[k] = numerator / (denominator + self.eps)

            bad = denominator <= self.eps
            centers_new[k, bad] = global_means[bad]

        return centers_new

    def _objective(self, U, D):
        return float(np.sum((U ** self.alpha) * D))

    def fit(self, X_missing, mask=None):
        X_missing, mask = validate_incomplete_matrix(X_missing, mask)
        n_samples, n_features = X_missing.shape

        U = self._initialize_membership(n_samples)
        centers = self._initialize_centers(X_missing, mask, U)

        # Use observed mean completion only to estimate feature bandwidths.
        X_for_sigma = initial_completion(X_missing, mask, strategy="mean")
        sigma = estimate_feature_bandwidths(X_for_sigma, eps=self.eps)

        weights = initialize_feature_weights(self.n_clusters, n_features)

        X_zero = np.nan_to_num(X_missing, nan=0.0)

        for _ in range(self.max_iter):
            U_old = U.copy()
            centers_old = centers.copy()
            weights_old = weights.copy()

            D_feat, K_feat = featurewise_gaussian_dissimilarity(
                X_zero, centers, sigma, eps=self.eps
            )

            weights = update_product_constraint_weights(
                U, D_feat, mask=mask, alpha=self.alpha, eps=self.eps
            )

            D = aggregate_weighted_kernel_distance(
                D_feat, weights, mask=mask, eps=self.eps
            )

            U = self._update_membership(D)

            centers = self._update_centers(
                X_missing, mask, U, centers, weights, sigma
            )

            obj = self._objective(U, D)
            self.objective_history_.append(obj)

            delta_U = np.linalg.norm(U - U_old)
            delta_V = np.linalg.norm(centers - centers_old) / (np.linalg.norm(centers_old) + self.eps)
            delta_W = np.linalg.norm(weights - weights_old) / (np.linalg.norm(weights_old) + self.eps)

            if delta_U < self.tol and delta_V < self.tol and delta_W < self.tol:
                break

        self.centers_ = centers
        self.weights_ = weights
        self.sigma_ = sigma
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

# =========================================================
# VKFCM-K-LP-OCS
# Variable-weighted Kernel FCM with Local adaptive distances
# using Optimal Completion Strategy
# =========================================================

class VKFCM_K_LP_OCS:
    """
    VKFCM-K-LP with Optimal Completion Strategy.

    Main components:
        - Feature-wise Gaussian kernel dissimilarity
        - Cluster-wise feature weights with product-to-one constraint
        - Iterative missing-value completion
        - Fuzzy membership update

    Missing-value update is implemented as a fixed-point approximation:
        x_hat_ij = sum_k u_ik^alpha lambda_kj K_ikj v_kj /
                   sum_k u_ik^alpha lambda_kj K_ikj

    for missing entries only.
    """

    def __init__(self, n_clusters, alpha=2.0, max_iter=300, tol=1e-5,
                 random_state=None, init_completion="mean", eps=1e-10):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init_completion = init_completion
        self.eps = eps

        self.centers_ = None
        self.weights_ = None
        self.sigma_ = None
        self.membership_ = None
        self.labels_ = None
        self.X_completed_ = None
        self.objective_history_ = []

    def _initialize_membership(self, n_samples):
        rng = np.random.default_rng(self.random_state)
        U = rng.random((n_samples, self.n_clusters))
        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _initialize_centers_complete(self, X_completed, U):
        U_alpha = U ** self.alpha
        numerator = U_alpha.T @ X_completed
        denominator = U_alpha.sum(axis=0)[:, None] + self.eps
        return numerator / denominator

    def _update_membership(self, D):
        D = np.asarray(D, dtype=float)
        U = np.zeros_like(D, dtype=float)

        zero_mask = D <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D_safe = np.maximum(D[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D_safe[:, :, None] / D_safe[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)
        return U

    def _update_centers(self, X_completed, U, centers_old, weights, sigma):
        D_feat, K_feat = featurewise_gaussian_dissimilarity(
            X_completed, centers_old, sigma, eps=self.eps
        )

        U_alpha = U ** self.alpha
        centers_new = np.zeros_like(centers_old)

        for k in range(self.n_clusters):
            w = U_alpha[:, k][:, None] * K_feat[:, k, :] * weights[k][None, :]
            numerator = np.sum(w * X_completed, axis=0)
            denominator = np.sum(w, axis=0)
            centers_new[k] = numerator / (denominator + self.eps)

        return centers_new

    def _update_missing_entries(self, X_missing, mask, X_completed, U, centers, weights, sigma):
        """
        OCS fixed-point completion for missing entries.

        For missing x_ij:
            x_hat_ij = sum_k u_ik^alpha lambda_kj K_ikj v_kj /
                       sum_k u_ik^alpha lambda_kj K_ikj
        """
        _, K_feat = featurewise_gaussian_dissimilarity(
            X_completed, centers, sigma, eps=self.eps
        )

        U_alpha = U ** self.alpha
        n_samples, n_features = X_completed.shape

        X_hat = np.zeros_like(X_completed)

        for j in range(n_features):
            numerator = np.zeros(n_samples, dtype=float)
            denominator = np.zeros(n_samples, dtype=float)

            for k in range(self.n_clusters):
                w = U_alpha[:, k] * weights[k, j] * K_feat[:, k, j]
                numerator += w * centers[k, j]
                denominator += w

            X_hat[:, j] = numerator / (denominator + self.eps)

        X_new = X_completed.copy()
        X_new[mask == 0] = X_hat[mask == 0]
        X_new[mask == 1] = X_missing[mask == 1]

        if np.isnan(X_new).any():
            raise ValueError("OCS completion produced NaN values.")

        return X_new

    def _objective(self, U, D):
        return float(np.sum((U ** self.alpha) * D))

    def fit(self, X_missing, mask=None):
        X_missing, mask = validate_incomplete_matrix(X_missing, mask)
        n_samples, n_features = X_missing.shape

        U = self._initialize_membership(n_samples)
        X_completed = initial_completion(X_missing, mask, strategy=self.init_completion)

        sigma = estimate_feature_bandwidths(X_completed, eps=self.eps)
        weights = initialize_feature_weights(self.n_clusters, n_features)
        centers = self._initialize_centers_complete(X_completed, U)

        for _ in range(self.max_iter):
            U_old = U.copy()
            centers_old = centers.copy()
            weights_old = weights.copy()
            X_old = X_completed.copy()

            D_feat, K_feat = featurewise_gaussian_dissimilarity(
                X_completed, centers, sigma, eps=self.eps
            )

            weights = update_product_constraint_weights(
                U, D_feat, mask=None, alpha=self.alpha, eps=self.eps
            )

            D = aggregate_weighted_kernel_distance(
                D_feat, weights, mask=None, eps=self.eps
            )

            U = self._update_membership(D)

            centers = self._update_centers(
                X_completed, U, centers, weights, sigma
            )

            X_completed = self._update_missing_entries(
                X_missing, mask, X_completed, U, centers, weights, sigma
            )

            # Re-estimate sigma after OCS completion.
            sigma = estimate_feature_bandwidths(X_completed, eps=self.eps)

            obj = self._objective(U, D)
            self.objective_history_.append(obj)

            delta_U = np.linalg.norm(U - U_old)
            delta_V = np.linalg.norm(centers - centers_old) / (np.linalg.norm(centers_old) + self.eps)
            delta_W = np.linalg.norm(weights - weights_old) / (np.linalg.norm(weights_old) + self.eps)
            delta_X = np.linalg.norm(X_completed - X_old) / (np.linalg.norm(X_old) + self.eps)

            if delta_U < self.tol and delta_V < self.tol and delta_W < self.tol and delta_X < self.tol:
                break

        self.centers_ = centers
        self.weights_ = weights
        self.sigma_ = sigma
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        self.X_completed_ = X_completed
        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

# =========================================================
# Unified wrapper for VKFCM-K-LP baselines
# =========================================================

class VKFCM_K_LP_Baseline:
    """
    Unified interface for VKFCM-K-LP baselines.

    strategy:
        "pds" -> VKFCM-K-LP with Partial Distance Strategy
        "ocs" -> VKFCM-K-LP with Optimal Completion Strategy
    """

    def __init__(self, strategy, n_clusters, alpha=2.0, max_iter=300,
                 tol=1e-5, random_state=None, **kwargs):
        self.strategy = strategy.lower()
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kwargs = kwargs

        self.model_ = None
        self.labels_ = None
        self.membership_ = None
        self.centers_ = None
        self.weights_ = None
        self.sigma_ = None
        self.X_completed_ = None

    def fit(self, X_missing, mask=None):
        if self.strategy == "pds":
            self.model_ = VKFCM_K_LP_PDS(
                n_clusters=self.n_clusters,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state
            )

        elif self.strategy == "ocs":
            self.model_ = VKFCM_K_LP_OCS(
                n_clusters=self.n_clusters,
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                **self.kwargs
            )

        else:
            raise ValueError("strategy must be either 'pds' or 'ocs'.")

        self.model_.fit(X_missing, mask)

        self.labels_ = self.model_.labels_
        self.membership_ = self.model_.membership_
        self.centers_ = self.model_.centers_
        self.weights_ = self.model_.weights_
        self.sigma_ = self.model_.sigma_

        if hasattr(self.model_, "X_completed_"):
            self.X_completed_ = self.model_.X_completed_

        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_


# ========== Imported revised notebook Cell 10 ==========
# =========================================================
# PyTorch imports and reproducibility utilities
# =========================================================

import torch
import torch.nn as nn
import torch.optim as optim


def set_torch_seed(seed=42):
    """
    Set NumPy and PyTorch seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================================================
# Autoencoder architecture for DFCM
# =========================================================

class DFCMAutoencoder(nn.Module):
    """
    Fully connected autoencoder for deep fuzzy clustering.

    Encoder:
        x -> hidden layers -> latent z

    Decoder:
        z -> hidden layers -> reconstructed x
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 64)):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

# =========================================================
# DFCM: Deep Autoencoder-Based Fuzzy C-Means
# =========================================================

class DeepAutoencoderFCM:
    """
    Deep Autoencoder-Based Fuzzy C-Means (DFCM).

    Pipeline for incomplete metabolomics data:
        1. Impute missing values.
        2. Pretrain an autoencoder using reconstruction loss.
        3. Initialize fuzzy memberships and prototypes in latent space.
        4. Jointly optimize:
               reconstruction loss + lambda_cluster * latent fuzzy loss

    Latent fuzzy loss:
        L_fuzzy = sum_i sum_k u_ik^alpha ||z_i - v_k||^2

    Notes:
        - This is an unsupervised clustering method.
        - Labels are not used during training.
        - Input X_missing may contain np.nan.
    """

    def __init__(
        self,
        n_clusters,
        imputation="mean",
        alpha=2.0,
        latent_dim=10,
        hidden_dims=(128, 64),
        lambda_cluster=1.0,
        pretrain_epochs=100,
        joint_epochs=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        tol=1e-5,
        random_state=42,
        device=None,
        verbose=False,
        **imputer_kwargs
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.imputation = imputation.lower()
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lambda_cluster = lambda_cluster
        self.pretrain_epochs = pretrain_epochs
        self.joint_epochs = joint_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.imputer_kwargs = imputer_kwargs

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_ = None
        self.X_completed_ = None
        self.X_reconstructed_ = None
        self.latent_ = None
        self.centers_ = None
        self.membership_ = None
        self.labels_ = None
        self.pretrain_loss_history_ = []
        self.joint_loss_history_ = []
        self.reconstruction_loss_history_ = []
        self.fuzzy_loss_history_ = []

    def _impute(self, X_missing):
        """
        Complete the incomplete matrix before applying DFCM.
        """
        if self.imputation == "mean":
            return mean_impute(X_missing)

        if self.imputation == "median":
            return median_impute(X_missing)

        if self.imputation == "knn":
            return knn_impute(X_missing, **self.imputer_kwargs)

        if self.imputation == "mice":
            return mice_impute(
                X_missing,
                random_state=self.random_state,
                **self.imputer_kwargs
            )

        raise ValueError("imputation must be one of: mean, median, knn, mice.")

    def _squared_distances_np(self, Z, centers):
        diff = Z[:, None, :] - centers[None, :, :]
        return np.sum(diff * diff, axis=2)

    def _update_membership_np(self, distances):
        distances = np.asarray(distances, dtype=float)
        U = np.zeros_like(distances, dtype=float)
        eps = 1e-10

        zero_mask = distances <= eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero
        if np.any(rows):
            D = np.maximum(distances[rows], eps)
            power = 1.0 / (self.alpha - 1.0)
            ratio = (D[:, :, None] / D[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + eps)

        U = U / (U.sum(axis=1, keepdims=True) + eps)
        return U

    def _update_centers_np(self, Z, U):
        eps = 1e-10
        U_alpha = U ** self.alpha
        numerator = U_alpha.T @ Z
        denominator = U_alpha.sum(axis=0)[:, None] + eps
        return numerator / denominator

    def _pretrain_autoencoder(self, X_tensor):
        """
        Pretrain the autoencoder using only reconstruction loss.
        """
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        mse = nn.MSELoss()

        self.model_.train()

        for epoch in range(self.pretrain_epochs):
            optimizer.zero_grad()

            _, X_hat = self.model_(X_tensor)
            loss = mse(X_hat, X_tensor)

            loss.backward()
            optimizer.step()

            self.pretrain_loss_history_.append(float(loss.item()))

            if self.verbose and (epoch + 1) % 50 == 0:
                print(f"Pretrain epoch {epoch+1}/{self.pretrain_epochs}, loss={loss.item():.6f}")

    def _encode_numpy(self, X_tensor):
        """
        Return latent representation as NumPy array.
        """
        self.model_.eval()

        with torch.no_grad():
            Z = self.model_.encode(X_tensor).detach().cpu().numpy()

        return Z

    def _reconstruct_numpy(self, X_tensor):
        """
        Return reconstructed matrix as NumPy array.
        """
        self.model_.eval()

        with torch.no_grad():
            _, X_hat = self.model_(X_tensor)
            X_hat = X_hat.detach().cpu().numpy()

        return X_hat

    def _initialize_latent_fcm(self, Z):
        """
        Initialize memberships and latent prototypes using manual FCM.
        """
        fcm_init = FuzzyCMeans(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            max_iter=300,
            tol=1e-5,
            random_state=self.random_state
        )
        fcm_init.fit(Z)

        U = fcm_init.membership_
        centers = fcm_init.centers_

        return U, centers

    def fit(self, X_missing):
        """
        Fit DFCM to incomplete data after imputation.
        """
        set_torch_seed(self.random_state)

        X_missing = np.asarray(X_missing, dtype=float)
        self.X_completed_ = self._impute(X_missing)

        if np.isnan(self.X_completed_).any():
            raise ValueError("Imputed matrix still contains NaN.")

        X_completed = np.asarray(self.X_completed_, dtype=np.float32)
        n_samples, input_dim = X_completed.shape

        X_tensor = torch.tensor(X_completed, dtype=torch.float32, device=self.device)

        self.model_ = DFCMAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        # Step 1: pretrain autoencoder
        self._pretrain_autoencoder(X_tensor)

        # Step 2: initialize latent FCM
        Z = self._encode_numpy(X_tensor)
        U, centers = self._initialize_latent_fcm(Z)

        # Step 3: joint optimization
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        mse = nn.MSELoss()
        previous_loss = None

        for epoch in range(self.joint_epochs):
            self.model_.train()
            optimizer.zero_grad()

            Z_tensor, X_hat = self.model_(X_tensor)

            centers_tensor = torch.tensor(
                centers,
                dtype=torch.float32,
                device=self.device
            )

            U_tensor = torch.tensor(
                U,
                dtype=torch.float32,
                device=self.device
            )

            distances = torch.cdist(Z_tensor, centers_tensor, p=2) ** 2

            reconstruction_loss = mse(X_hat, X_tensor)
            fuzzy_loss = torch.sum((U_tensor ** self.alpha) * distances) / n_samples

            total_loss = reconstruction_loss + self.lambda_cluster * fuzzy_loss

            total_loss.backward()
            optimizer.step()

            total_loss_value = float(total_loss.item())
            rec_value = float(reconstruction_loss.item())
            fuzzy_value = float(fuzzy_loss.item())

            self.joint_loss_history_.append(total_loss_value)
            self.reconstruction_loss_history_.append(rec_value)
            self.fuzzy_loss_history_.append(fuzzy_value)

            # Alternating update of latent memberships and prototypes
            Z_np = self._encode_numpy(X_tensor)
            centers = self._update_centers_np(Z_np, U)
            distances_np = self._squared_distances_np(Z_np, centers)
            U = self._update_membership_np(distances_np)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Joint epoch {epoch+1}/{self.joint_epochs}, "
                    f"loss={total_loss_value:.6f}, "
                    f"rec={rec_value:.6f}, fuzzy={fuzzy_value:.6f}"
                )

            if previous_loss is not None:
                rel_change = abs(total_loss_value - previous_loss) / (abs(previous_loss) + 1e-10)
                if rel_change < self.tol:
                    break

            previous_loss = total_loss_value

        # Final update
        Z_final = self._encode_numpy(X_tensor)
        centers = self._update_centers_np(Z_final, U)
        distances_final = self._squared_distances_np(Z_final, centers)
        U = self._update_membership_np(distances_final)

        self.latent_ = Z_final
        self.centers_ = centers
        self.membership_ = U
        self.labels_ = np.argmax(U, axis=1)
        self.X_reconstructed_ = self._reconstruct_numpy(X_tensor)

        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_

# =========================================================
# Unified wrapper for DFCM baseline
# =========================================================

class DFCMBaseline:
    """
    Unified interface for DFCM baseline.

    Pipeline:
        X_missing -> imputation -> DFCM
    """

    def __init__(
        self,
        imputation="mean",
        n_clusters=2,
        alpha=2.0,
        latent_dim=10,
        hidden_dims=(128, 64),
        lambda_cluster=1.0,
        pretrain_epochs=100,
        joint_epochs=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        tol=1e-5,
        random_state=42,
        device=None,
        verbose=False,
        **imputer_kwargs
    ):
        self.model_ = DeepAutoencoderFCM(
            n_clusters=n_clusters,
            imputation=imputation,
            alpha=alpha,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            lambda_cluster=lambda_cluster,
            pretrain_epochs=pretrain_epochs,
            joint_epochs=joint_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            tol=tol,
            random_state=random_state,
            device=device,
            verbose=verbose,
            **imputer_kwargs
        )

        self.labels_ = None
        self.membership_ = None
        self.centers_ = None
        self.latent_ = None
        self.X_completed_ = None
        self.X_reconstructed_ = None

    def fit(self, X_missing):
        self.model_.fit(X_missing)

        self.labels_ = self.model_.labels_
        self.membership_ = self.model_.membership_
        self.centers_ = self.model_.centers_
        self.latent_ = self.model_.latent_
        self.X_completed_ = self.model_.X_completed_
        self.X_reconstructed_ = self.model_.X_reconstructed_

        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_

# =========================================================
# Additional imports for DEC
# =========================================================

from sklearn.cluster import KMeans
import torch.nn.functional as F

# =========================================================
# Autoencoder architecture for DEC
# =========================================================

class DECAutoencoder(nn.Module):
    """
    Fully connected autoencoder used for DEC pretraining.

    DEC first pretrains an autoencoder using reconstruction loss.
    Then the encoder is fine-tuned using clustering-oriented KL loss.
    """

    def __init__(self, input_dim, latent_dim=10, hidden_dims=(128, 64)):
        super().__init__()

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())
            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())
            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

# =========================================================
# DEC clustering network
# =========================================================

class DECNetwork(nn.Module):
    """
    DEC network consisting of:
        - pretrained encoder
        - trainable cluster centers in latent space

    Soft assignment uses Student-t distribution:
        q_ik = (1 + ||z_i - mu_k||^2 / a)^(-(a+1)/2)
               normalized over clusters
    """

    def __init__(self, encoder, initial_centers, student_alpha=1.0):
        super().__init__()

        self.encoder = encoder
        self.student_alpha = student_alpha

        initial_centers = torch.tensor(initial_centers, dtype=torch.float32)
        self.cluster_centers = nn.Parameter(initial_centers)

    def forward(self, x):
        z = self.encoder(x)
        q = self.soft_assign(z)
        return z, q

    def soft_assign(self, z):
        """
        Compute Student-t soft assignment q.
        """
        dist2 = torch.sum(
            (z[:, None, :] - self.cluster_centers[None, :, :]) ** 2,
            dim=2
        )

        numerator = (1.0 + dist2 / self.student_alpha)
        numerator = numerator ** (-(self.student_alpha + 1.0) / 2.0)

        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        return q


def dec_target_distribution(q):
    """
    Compute DEC target distribution.

    p_ik = (q_ik^2 / f_k) / sum_l (q_il^2 / f_l)
    where f_k = sum_i q_ik.
    """
    weight = (q ** 2) / torch.sum(q, dim=0, keepdim=True)
    p = weight / torch.sum(weight, dim=1, keepdim=True)
    return p

# =========================================================
# DEC: Deep Embedded Clustering
# =========================================================

class DeepEmbeddedClustering:
    """
    Deep Embedded Clustering (DEC).

    Pipeline:
        1. Impute missing values.
        2. Pretrain autoencoder using reconstruction loss.
        3. Initialize cluster centers using KMeans in latent space.
        4. Fine-tune encoder and cluster centers using KL divergence:
               KL(P || Q)

    Notes:
        - Labels are not used during training.
        - Input X_missing may contain np.nan.
        - DEC does not optimize fuzzy c-means objective.
    """

    def __init__(
        self,
        n_clusters,
        imputation="mean",
        latent_dim=10,
        hidden_dims=(128, 64),
        pretrain_epochs=100,
        dec_epochs=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        student_alpha=1.0,
        update_interval=10,
        tol=1e-4,
        random_state=42,
        device=None,
        verbose=False,
        **imputer_kwargs
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        self.n_clusters = n_clusters
        self.imputation = imputation.lower()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.pretrain_epochs = pretrain_epochs
        self.dec_epochs = dec_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.student_alpha = student_alpha
        self.update_interval = update_interval
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.imputer_kwargs = imputer_kwargs

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.autoencoder_ = None
        self.dec_model_ = None

        self.X_completed_ = None
        self.X_reconstructed_ = None
        self.latent_ = None
        self.q_ = None
        self.labels_ = None
        self.centers_ = None

        self.pretrain_loss_history_ = []
        self.dec_loss_history_ = []

    def _impute(self, X_missing):
        """
        Complete the incomplete matrix before applying DEC.
        """
        if self.imputation == "mean":
            return mean_impute(X_missing)

        if self.imputation == "median":
            return median_impute(X_missing)

        if self.imputation == "knn":
            return knn_impute(X_missing, **self.imputer_kwargs)

        if self.imputation == "mice":
            return mice_impute(
                X_missing,
                random_state=self.random_state,
                **self.imputer_kwargs
            )

        raise ValueError("imputation must be one of: mean, median, knn, mice.")

    def _pretrain_autoencoder(self, X_tensor):
        """
        Pretrain autoencoder using MSE reconstruction loss.
        """
        optimizer = optim.Adam(
            self.autoencoder_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        mse = nn.MSELoss()
        self.autoencoder_.train()

        for epoch in range(self.pretrain_epochs):
            optimizer.zero_grad()

            _, X_hat = self.autoencoder_(X_tensor)
            loss = mse(X_hat, X_tensor)

            loss.backward()
            optimizer.step()

            self.pretrain_loss_history_.append(float(loss.item()))

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"DEC pretrain epoch {epoch+1}/{self.pretrain_epochs}, "
                    f"loss={loss.item():.6f}"
                )

    def _encode_numpy(self, X_tensor):
        """
        Encode X into latent representation as NumPy array.
        """
        self.autoencoder_.eval()

        with torch.no_grad():
            Z = self.autoencoder_.encode(X_tensor)
            Z = Z.detach().cpu().numpy()

        return Z

    def _reconstruct_numpy(self, X_tensor):
        """
        Reconstruct X using the autoencoder decoder.
        """
        self.autoencoder_.eval()

        with torch.no_grad():
            _, X_hat = self.autoencoder_(X_tensor)
            X_hat = X_hat.detach().cpu().numpy()

        return X_hat

    def _initialize_centers_kmeans(self, Z):
        """
        Initialize DEC cluster centers using KMeans in latent space.
        """
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.random_state
        )
        labels = kmeans.fit_predict(Z)
        centers = kmeans.cluster_centers_

        return centers, labels

    def fit(self, X_missing):
        """
        Fit DEC to incomplete data after imputation.
        """
        set_torch_seed(self.random_state)

        X_missing = np.asarray(X_missing, dtype=float)
        self.X_completed_ = self._impute(X_missing)

        if np.isnan(self.X_completed_).any():
            raise ValueError("Imputed matrix still contains NaN.")

        X_completed = np.asarray(self.X_completed_, dtype=np.float32)
        n_samples, input_dim = X_completed.shape

        X_tensor = torch.tensor(
            X_completed,
            dtype=torch.float32,
            device=self.device
        )

        # Step 1: autoencoder pretraining
        self.autoencoder_ = DECAutoencoder(
            input_dim=input_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims
        ).to(self.device)

        self._pretrain_autoencoder(X_tensor)

        # Step 2: KMeans initialization in latent space
        Z_init = self._encode_numpy(X_tensor)
        initial_centers, initial_labels = self._initialize_centers_kmeans(Z_init)

        # Step 3: DEC model
        self.dec_model_ = DECNetwork(
            encoder=self.autoencoder_.encoder,
            initial_centers=initial_centers,
            student_alpha=self.student_alpha
        ).to(self.device)

        optimizer = optim.Adam(
            self.dec_model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        previous_labels = initial_labels.copy()

        # Initial target distribution
        self.dec_model_.eval()
        with torch.no_grad():
            _, q = self.dec_model_(X_tensor)
            p = dec_target_distribution(q).detach()

        # Step 4: DEC optimization
        for epoch in range(self.dec_epochs):
            self.dec_model_.train()

            if epoch % self.update_interval == 0:
                with torch.no_grad():
                    _, q_current = self.dec_model_(X_tensor)
                    p = dec_target_distribution(q_current).detach()
                    labels_current = torch.argmax(q_current, dim=1).cpu().numpy()

                label_change = np.mean(labels_current != previous_labels)

                if self.verbose:
                    print(
                        f"DEC epoch {epoch}/{self.dec_epochs}, "
                        f"label_change={label_change:.6f}"
                    )

                if epoch > 0 and label_change < self.tol:
                    break

                previous_labels = labels_current.copy()

            optimizer.zero_grad()

            _, q = self.dec_model_(X_tensor)

            # KL(P || Q)
            loss = F.kl_div(
                torch.log(q + 1e-10),
                p,
                reduction="batchmean"
            )

            loss.backward()
            optimizer.step()

            self.dec_loss_history_.append(float(loss.item()))

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"DEC epoch {epoch+1}/{self.dec_epochs}, "
                    f"KL loss={loss.item():.6f}"
                )

        # Final outputs
        self.dec_model_.eval()

        with torch.no_grad():
            Z_final, q_final = self.dec_model_(X_tensor)

        self.latent_ = Z_final.detach().cpu().numpy()
        self.q_ = q_final.detach().cpu().numpy()
        self.labels_ = np.argmax(self.q_, axis=1)
        self.centers_ = self.dec_model_.cluster_centers.detach().cpu().numpy()

        # Decoder is not optimized during DEC fine-tuning, but reconstruction
        # remains useful as a reference from the pretrained autoencoder.
        self.X_reconstructed_ = self._reconstruct_numpy(X_tensor)

        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_

# =========================================================
# Unified wrapper for DEC baseline
# =========================================================

class DECBaseline:
    """
    Unified interface for DEC baseline.

    Pipeline:
        X_missing -> imputation -> DEC
    """

    def __init__(
        self,
        imputation="mean",
        n_clusters=2,
        latent_dim=10,
        hidden_dims=(128, 64),
        pretrain_epochs=100,
        dec_epochs=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        student_alpha=1.0,
        update_interval=10,
        tol=1e-4,
        random_state=42,
        device=None,
        verbose=False,
        **imputer_kwargs
    ):
        self.model_ = DeepEmbeddedClustering(
            n_clusters=n_clusters,
            imputation=imputation,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            pretrain_epochs=pretrain_epochs,
            dec_epochs=dec_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            student_alpha=student_alpha,
            update_interval=update_interval,
            tol=tol,
            random_state=random_state,
            device=device,
            verbose=verbose,
            **imputer_kwargs
        )

        self.labels_ = None
        self.q_ = None
        self.centers_ = None
        self.latent_ = None
        self.X_completed_ = None
        self.X_reconstructed_ = None

    def fit(self, X_missing):
        self.model_.fit(X_missing)

        self.labels_ = self.model_.labels_
        self.q_ = self.model_.q_
        self.centers_ = self.model_.centers_
        self.latent_ = self.model_.latent_
        self.X_completed_ = self.model_.X_completed_
        self.X_reconstructed_ = self.model_.X_reconstructed_

        return self

    def fit_predict(self, X_missing):
        self.fit(X_missing)
        return self.labels_


# ========== Imported revised notebook Cell 12 ==========
# =========================================================
# Missingness-aware encoder-decoder for JMADKFC
# =========================================================

class JMADKFCAutoencoder(nn.Module):
    """
    Missingness-aware autoencoder.

    Encoder input:
        X_tilde = [X_zero_filled, M] with dimension 2d

    Decoder output:
        reconstructed metabolite matrix with dimension d
    """

    def __init__(self, metabolite_dim, latent_dim=10, hidden_dims=(128, 64), dropout=0.0):
        super().__init__()

        input_dim = 2 * metabolite_dim

        encoder_layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(nn.ReLU())

            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))

            prev_dim = h

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim

        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(nn.ReLU())

            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))

            prev_dim = h

        decoder_layers.append(nn.Linear(prev_dim, metabolite_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x_tilde):
        return self.encoder(x_tilde)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_tilde):
        z = self.encode(x_tilde)
        x_hat = self.decode(z)
        return z, x_hat

# =========================================================
# JMADKFC: Joint Missingness-Aware Deep Kernel Fuzzy Clustering
# =========================================================

class JMADKFC:
    """
    Joint Missingness-Aware Deep Kernel Fuzzy Clustering.

    Objective:
        J = L_rec + lambda_cluster * L_akf + beta_reg * R_reg

    where:
        L_rec = masked reconstruction loss
        L_akf = adaptive kernel fuzzy clustering loss

    Input:
        X_missing : incomplete matrix with np.nan at missing entries
        mask      : binary mask, 1 observed and 0 missing

    Output:
        labels_
        membership_
        centers_
        decoded_prototypes_
        membership-weighted metabolite profiles
        metabolite importance scores
    """

    def __init__(
        self,
        n_clusters,
        alpha=2.0,
        latent_dim=10,
        hidden_dims=(128, 64),
        lambda_cluster=1.0,
        beta_reg=1e-5,
        q_neighbors=5,
        pretrain_epochs=100,
        joint_epochs=200,
        learning_rate=1e-3,
        tol=1e-5,
        prototype_inner_iter=1,
        random_state=42,
        device=None,
        dropout=0.0,
        verbose=False,
        eps=1e-10
    ):
        if n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.lambda_cluster = lambda_cluster
        self.beta_reg = beta_reg
        self.q_neighbors = q_neighbors
        self.pretrain_epochs = pretrain_epochs
        self.joint_epochs = joint_epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.prototype_inner_iter = prototype_inner_iter
        self.random_state = random_state
        self.dropout = dropout
        self.verbose = verbose
        self.eps = eps

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_ = None

        self.X_zero_filled_ = None
        self.mask_ = None
        self.X_tilde_ = None

        self.latent_ = None
        self.membership_ = None
        self.centers_ = None
        self.labels_ = None

        self.sigma_i_ = None
        self.sigma_k_ = None

        self.X_reconstructed_ = None
        self.decoded_prototypes_ = None

        self.pretrain_loss_history_ = []
        self.joint_loss_history_ = []
        self.reconstruction_loss_history_ = []
        self.akf_loss_history_ = []

    # -----------------------------------------------------
    # Input preparation
    # -----------------------------------------------------

    def _prepare_inputs(self, X_missing, mask=None):
        X_missing = np.asarray(X_missing, dtype=float)

        if X_missing.ndim != 2:
            raise ValueError("X_missing must be a 2D matrix.")

        if mask is None:
            mask = (~np.isnan(X_missing)).astype(int)
        else:
            mask = np.asarray(mask, dtype=int)

        if mask.shape != X_missing.shape:
            raise ValueError("mask must have the same shape as X_missing.")

        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("Each sample must have at least one observed feature.")

        X_zero = np.nan_to_num(X_missing, nan=0.0)
        X_tilde = np.concatenate([X_zero, mask.astype(float)], axis=1)

        return (
            X_zero.astype(np.float32),
            mask.astype(np.float32),
            X_tilde.astype(np.float32)
        )

    # -----------------------------------------------------
    # Loss functions
    # -----------------------------------------------------

    def _masked_reconstruction_loss(self, X_zero_tensor, mask_tensor, X_hat_tensor):
        """
        L_rec = sum m_ij (x_ij - xhat_ij)^2 / sum m_ij
        """
        numerator = torch.sum(mask_tensor * (X_zero_tensor - X_hat_tensor) ** 2)
        denominator = torch.sum(mask_tensor) + self.eps
        return numerator / denominator

    def _regularization_loss(self):
        """
        R_reg = ||theta||_2^2 + ||phi||_2^2
        """
        reg = 0.0

        for param in self.model_.parameters():
            reg = reg + torch.sum(param ** 2)

        return reg

    # -----------------------------------------------------
    # Autoencoder utilities
    # -----------------------------------------------------

    def _pretrain_autoencoder(self, X_tilde_tensor, X_zero_tensor, mask_tensor):
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0
        )

        self.model_.train()

        for epoch in range(self.pretrain_epochs):
            optimizer.zero_grad()

            _, X_hat = self.model_(X_tilde_tensor)

            rec_loss = self._masked_reconstruction_loss(
                X_zero_tensor,
                mask_tensor,
                X_hat
            )

            reg_loss = self._regularization_loss()
            loss = rec_loss + self.beta_reg * reg_loss

            loss.backward()
            optimizer.step()

            self.pretrain_loss_history_.append(float(loss.item()))

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"Pretrain epoch {epoch+1}/{self.pretrain_epochs}, "
                    f"loss={loss.item():.6f}, rec={rec_loss.item():.6f}"
                )

    def _encode_numpy(self, X_tilde_tensor):
        self.model_.eval()

        with torch.no_grad():
            Z = self.model_.encode(X_tilde_tensor)
            Z = Z.detach().cpu().numpy()

        return Z

    def _decode_numpy(self, Z_numpy):
        self.model_.eval()

        with torch.no_grad():
            Z_tensor = torch.tensor(
                Z_numpy,
                dtype=torch.float32,
                device=self.device
            )
            X_decoded = self.model_.decode(Z_tensor)
            X_decoded = X_decoded.detach().cpu().numpy()

        return X_decoded

    def _reconstruct_numpy(self, X_tilde_tensor):
        self.model_.eval()

        with torch.no_grad():
            _, X_hat = self.model_(X_tilde_tensor)
            X_hat = X_hat.detach().cpu().numpy()

        return X_hat

    # -----------------------------------------------------
    # Adaptive bandwidths
    # -----------------------------------------------------

    def _sample_bandwidths_np(self, Z):
        """
        sigma_i = distance from z_i to its q-th nearest neighbor.
        """
        Z = np.asarray(Z, dtype=float)
        n = Z.shape[0]

        if n < 2:
            return np.ones(n, dtype=float)

        q = int(min(max(1, self.q_neighbors), n - 1))

        diff = Z[:, None, :] - Z[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        dist_sorted = np.sort(dist, axis=1)

        # dist_sorted[:, 0] is self-distance.
        sigma_i = dist_sorted[:, q] + self.eps
        sigma_i = np.maximum(sigma_i, self.eps)

        return sigma_i

    def _prototype_bandwidths_np(self, U, sigma_i):
        """
        sigma_k = sum_i u_ik^alpha sigma_i / sum_i u_ik^alpha
        """
        U_alpha = U ** self.alpha

        numerator = U_alpha.T @ sigma_i
        denominator = U_alpha.sum(axis=0) + self.eps

        sigma_k = numerator / denominator + self.eps
        sigma_k = np.maximum(sigma_k, self.eps)

        return sigma_k

    # -----------------------------------------------------
    # Adaptive kernel
    # -----------------------------------------------------

    def _adaptive_kernel_np(self, Z, V, sigma_i, sigma_k):
        """
        K(z_i, v_k) = exp(-||z_i-v_k||^2 / (2 sigma_i sigma_k))
        D_AK = 1 - K
        """
        Z = np.asarray(Z, dtype=float)
        V = np.asarray(V, dtype=float)

        dist2 = np.sum((Z[:, None, :] - V[None, :, :]) ** 2, axis=2)

        denominator = 2.0 * sigma_i[:, None] * sigma_k[None, :] + self.eps

        K = np.exp(-dist2 / denominator)
        D = 1.0 - K

        D = np.maximum(D, self.eps)

        return D, K

    def _adaptive_kernel_loss_torch(self, Z_tensor, U_np, V_np, sigma_i_np, sigma_k_np):
        """
        L_akf = sum_i sum_k u_ik^alpha D_AK(z_i, v_k)

        U, V, sigma_i, and sigma_k are treated as fixed in the current
        gradient step, while Z_tensor remains differentiable.
        """
        U_tensor = torch.tensor(
            U_np,
            dtype=torch.float32,
            device=self.device
        )

        V_tensor = torch.tensor(
            V_np,
            dtype=torch.float32,
            device=self.device
        )

        sigma_i_tensor = torch.tensor(
            sigma_i_np,
            dtype=torch.float32,
            device=self.device
        )

        sigma_k_tensor = torch.tensor(
            sigma_k_np,
            dtype=torch.float32,
            device=self.device
        )

        dist2 = torch.sum(
            (Z_tensor[:, None, :] - V_tensor[None, :, :]) ** 2,
            dim=2
        )

        denominator = (
            2.0 * sigma_i_tensor[:, None] * sigma_k_tensor[None, :]
            + self.eps
        )

        K = torch.exp(-dist2 / denominator)
        D = 1.0 - K

        loss = torch.sum((U_tensor ** self.alpha) * D) / Z_tensor.shape[0]

        return loss

    # -----------------------------------------------------
    # Fuzzy membership and prototype updates
    # -----------------------------------------------------

    def _update_membership_np(self, D):
        """
        u_ik = [sum_l ((D_ik+eps)/(D_il+eps))^(1/(alpha-1))]^-1
        """
        D = np.asarray(D, dtype=float)
        U = np.zeros_like(D, dtype=float)

        zero_mask = D <= self.eps
        rows_with_zero = np.any(zero_mask, axis=1)

        if np.any(rows_with_zero):
            for i in np.where(rows_with_zero)[0]:
                z = zero_mask[i]
                U[i, z] = 1.0 / z.sum()

        rows = ~rows_with_zero

        if np.any(rows):
            D_safe = np.maximum(D[rows], self.eps)
            power = 1.0 / (self.alpha - 1.0)

            ratio = (D_safe[:, :, None] / D_safe[:, None, :]) ** power
            U[rows] = 1.0 / (ratio.sum(axis=2) + self.eps)

        U = U / (U.sum(axis=1, keepdims=True) + self.eps)

        return U

    def _update_prototypes_np(self, Z, U, V, sigma_i, sigma_k):
        """
        Adaptive kernel fixed-point prototype update:

            v_k =
            sum_i u_ik^alpha omega_ik K(z_i,v_k) z_i /
            sum_i u_ik^alpha omega_ik K(z_i,v_k)

        where:
            omega_ik = 1 / (sigma_i sigma_k + eps)
        """
        Z = np.asarray(Z, dtype=float)
        V_new = np.asarray(V, dtype=float).copy()

        for _ in range(self.prototype_inner_iter):
            _, K = self._adaptive_kernel_np(Z, V_new, sigma_i, sigma_k)

            U_alpha = U ** self.alpha
            omega = 1.0 / (sigma_i[:, None] * sigma_k[None, :] + self.eps)

            for k in range(self.n_clusters):
                weights = U_alpha[:, k] * omega[:, k] * K[:, k]

                numerator = np.sum(weights[:, None] * Z, axis=0)
                denominator = np.sum(weights) + self.eps

                V_new[k] = numerator / denominator

        return V_new

    # -----------------------------------------------------
    # Fitting procedure
    # -----------------------------------------------------

    def fit(self, X_missing, mask=None):
        set_torch_seed(self.random_state)

        X_zero, mask_array, X_tilde = self._prepare_inputs(X_missing, mask)

        n_samples, metabolite_dim = X_zero.shape

        X_zero_tensor = torch.tensor(
            X_zero,
            dtype=torch.float32,
            device=self.device
        )

        mask_tensor = torch.tensor(
            mask_array,
            dtype=torch.float32,
            device=self.device
        )

        X_tilde_tensor = torch.tensor(
            X_tilde,
            dtype=torch.float32,
            device=self.device
        )

        self.model_ = JMADKFCAutoencoder(
            metabolite_dim=metabolite_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        # Step 1: pretrain encoder-decoder using masked reconstruction loss
        self._pretrain_autoencoder(
            X_tilde_tensor,
            X_zero_tensor,
            mask_tensor
        )

        # Step 2: initialize U and V in pretrained latent space
        Z_init = self._encode_numpy(X_tilde_tensor)

        fcm_init = FuzzyCMeans(
            n_clusters=self.n_clusters,
            alpha=self.alpha,
            max_iter=300,
            tol=1e-5,
            random_state=self.random_state
        )

        fcm_init.fit(Z_init)

        U = fcm_init.membership_
        V = fcm_init.centers_

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0
        )
        self._initial_proto_separation = float(np.linalg.norm(V[0]-V[1]))

        previous_total_loss = None

        # Step 3: alternating joint optimization
        for epoch in range(self.joint_epochs):
            self._current_epoch = epoch

            # 3.1 Encode current latent representation
            Z_np = self._encode_numpy(X_tilde_tensor)

            # 3.2 Estimate adaptive bandwidths
            sigma_i = self._sample_bandwidths_np(Z_np)
            sigma_k = self._prototype_bandwidths_np(U, sigma_i)

            # 3.3 Compute adaptive kernel distance and update memberships
            D, _ = self._adaptive_kernel_np(Z_np, V, sigma_i, sigma_k)
            U = self._update_membership_np(D)

            # 3.4 Recompute sigma_k using updated memberships
            sigma_k = self._prototype_bandwidths_np(U, sigma_i)

            # 3.5 Update latent prototypes using adaptive kernel fixed-point rule
            V = self._update_prototypes_np(
                Z=Z_np,
                U=U,
                V=V,
                sigma_i=sigma_i,
                sigma_k=sigma_k
            )
            if hasattr(self, "_stabilize_prototypes"):
                V = self._stabilize_prototypes(Z_np, U, V, epoch)

            # 3.6 Update encoder-decoder parameters by minimizing J
            self.model_.train()
            optimizer.zero_grad()

            Z_tensor, X_hat_tensor = self.model_(X_tilde_tensor)

            rec_loss = self._masked_reconstruction_loss(
                X_zero_tensor,
                mask_tensor,
                X_hat_tensor
            )

            akf_loss = self._adaptive_kernel_loss_torch(
                Z_tensor=Z_tensor,
                U_np=U,
                V_np=V,
                sigma_i_np=sigma_i,
                sigma_k_np=sigma_k
            )

            reg_loss = self._regularization_loss()

            effective_lambda = (self._effective_lambda(epoch)
                                if hasattr(self, "_effective_lambda") else self.lambda_cluster)
            total_loss = rec_loss + effective_lambda * akf_loss + self.beta_reg * reg_loss
            if hasattr(self, "_optimizer_step_diagnostic"):
                self._optimizer_step_diagnostic(optimizer, rec_loss, akf_loss, reg_loss,
                                                effective_lambda, Z_tensor, U, V, sigma_i, epoch)
            else:
                total_loss.backward()
                optimizer.step()

            total_value = float(total_loss.item())
            rec_value = float(rec_loss.item())
            akf_value = float(akf_loss.item())

            self.joint_loss_history_.append(total_value)
            self.reconstruction_loss_history_.append(rec_value)
            self.akf_loss_history_.append(akf_value)
            # CGPC extension: optional detached update of input completion for the NEXT epoch.
            # Unchanged behavior for all pre-existing model classes.
            if hasattr(self, "_update_dynamic_completion") and epoch + 1 < self.joint_epochs:
                self._update_dynamic_completion(X_tilde_tensor, mask_tensor, epoch, U, V)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"JMADKFC epoch {epoch+1}/{self.joint_epochs}, "
                    f"loss={total_value:.6f}, "
                    f"rec={rec_value:.6f}, "
                    f"akf={akf_value:.6f}"
                )

            if previous_total_loss is not None:
                rel_change = abs(total_value - previous_total_loss) / (
                    abs(previous_total_loss) + self.eps
                )

                if rel_change < self.tol:
                    if self.verbose:
                        print(
                            f"Converged at epoch {epoch+1}, "
                            f"relative change={rel_change:.8f}"
                        )
                    break

            previous_total_loss = total_value

        # Step 4: final fuzzy update after training
        Z_final = self._encode_numpy(X_tilde_tensor)

        sigma_i = self._sample_bandwidths_np(Z_final)
        sigma_k = self._prototype_bandwidths_np(U, sigma_i)

        D_final, _ = self._adaptive_kernel_np(Z_final, V, sigma_i, sigma_k)
        U = self._update_membership_np(D_final)

        sigma_k = self._prototype_bandwidths_np(U, sigma_i)

        V = self._update_prototypes_np(
            Z=Z_final,
            U=U,
            V=V,
            sigma_i=sigma_i,
            sigma_k=sigma_k
        )
        if hasattr(self, "_stabilize_prototypes"):
            V = self._stabilize_prototypes(Z_final, U, V, self.joint_epochs)
        self.X_zero_filled_ = X_zero
        self.mask_ = mask_array
        self.X_tilde_ = X_tilde

        self.latent_ = Z_final
        self.membership_ = U
        self.centers_ = V
        self.labels_ = np.argmax(U, axis=1)

        self.sigma_i_ = sigma_i
        self.sigma_k_ = sigma_k

        self.X_reconstructed_ = self._reconstruct_numpy(X_tilde_tensor)
        self.decoded_prototypes_ = self._decode_numpy(V)

        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

    # -----------------------------------------------------
    # Biological interpretation utilities
    # -----------------------------------------------------

    def membership_weighted_profiles(self):
        """
        Compute membership-weighted observed metabolite profiles:

            xbar_kj =
            sum_i u_ik^alpha m_ij x_ij /
            sum_i u_ik^alpha m_ij
        """
        if self.membership_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_zero = self.X_zero_filled_
        M = self.mask_
        U = self.membership_

        U_alpha = U ** self.alpha

        n_clusters = self.n_clusters
        n_features = X_zero.shape[1]

        profiles = np.zeros((n_clusters, n_features), dtype=float)

        for k in range(n_clusters):
            numerator = np.sum(
                U_alpha[:, k][:, None] * M * X_zero,
                axis=0
            )

            denominator = np.sum(
                U_alpha[:, k][:, None] * M,
                axis=0
            ) + self.eps

            profiles[k] = numerator / denominator

        return profiles

    def global_observed_profile(self):
        """
        Compute global observed metabolite profile.
        """
        if self.membership_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X_zero = self.X_zero_filled_
        M = self.mask_

        numerator = np.sum(M * X_zero, axis=0)
        denominator = np.sum(M, axis=0) + self.eps

        return numerator / denominator

    def observed_standard_deviation(self):
        """
        Compute observed standard deviation for each metabolite.
        """
        X_zero = self.X_zero_filled_
        M = self.mask_

        global_profile = self.global_observed_profile()

        numerator = np.sum(
            M * (X_zero - global_profile[None, :]) ** 2,
            axis=0
        )

        denominator = np.sum(M, axis=0) + self.eps

        return np.sqrt(numerator / denominator)

    def metabolite_importance_scores(self):
        """
        I_kj = |xbar_kj - xbar_.j| / (s_j + eps)
        """
        profiles = self.membership_weighted_profiles()
        global_profile = self.global_observed_profile()
        sd = self.observed_standard_deviation()

        scores = np.abs(profiles - global_profile[None, :]) / (
            sd[None, :] + self.eps
        )

        return scores

    def top_metabolites_per_cluster(self, feature_names=None, top_n=10):
        """
        Return top discriminative metabolites for each cluster.
        """
        scores = self.metabolite_importance_scores()
        n_features = scores.shape[1]

        if feature_names is None:
            feature_names = [f"Metabolite_{j+1}" for j in range(n_features)]

        result = {}

        for k in range(self.n_clusters):
            idx = np.argsort(scores[k])[::-1][:top_n]

            result[f"Cluster_{k}"] = pd.DataFrame({
                "feature": [feature_names[j] for j in idx],
                "importance": scores[k, idx],
                "membership_weighted_profile": self.membership_weighted_profiles()[k, idx],
                "decoded_prototype": self.decoded_prototypes_[k, idx]
            })

        return result

# =========================================================
# Unified wrapper for proposed method
# =========================================================

class JMADKFCProposed:
    """
    Unified interface for proposed JMADKFC model.
    """

    def __init__(
        self,
        n_clusters=2,
        alpha=2.0,
        latent_dim=10,
        hidden_dims=(128, 64),
        lambda_cluster=1.0,
        beta_reg=1e-5,
        q_neighbors=5,
        pretrain_epochs=100,
        joint_epochs=200,
        learning_rate=1e-3,
        tol=1e-5,
        prototype_inner_iter=1,
        random_state=42,
        device=None,
        dropout=0.0,
        verbose=False
    ):
        self.model_ = JMADKFC(
            n_clusters=n_clusters,
            alpha=alpha,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            lambda_cluster=lambda_cluster,
            beta_reg=beta_reg,
            q_neighbors=q_neighbors,
            pretrain_epochs=pretrain_epochs,
            joint_epochs=joint_epochs,
            learning_rate=learning_rate,
            tol=tol,
            prototype_inner_iter=prototype_inner_iter,
            random_state=random_state,
            device=device,
            dropout=dropout,
            verbose=verbose
        )

        self.labels_ = None
        self.membership_ = None
        self.centers_ = None
        self.latent_ = None

        self.X_reconstructed_ = None
        self.decoded_prototypes_ = None
        self.sigma_i_ = None
        self.sigma_k_ = None

    def fit(self, X_missing, mask=None):
        self.model_.fit(X_missing, mask)

        self.labels_ = self.model_.labels_
        self.membership_ = self.model_.membership_
        self.centers_ = self.model_.centers_
        self.latent_ = self.model_.latent_

        self.X_reconstructed_ = self.model_.X_reconstructed_
        self.decoded_prototypes_ = self.model_.decoded_prototypes_
        self.sigma_i_ = self.model_.sigma_i_
        self.sigma_k_ = self.model_.sigma_k_

        return self

    def fit_predict(self, X_missing, mask=None):
        self.fit(X_missing, mask)
        return self.labels_

    def membership_weighted_profiles(self):
        return self.model_.membership_weighted_profiles()

    def metabolite_importance_scores(self):
        return self.model_.metabolite_importance_scores()

    def top_metabolites_per_cluster(self, feature_names=None, top_n=10):
        return self.model_.top_metabolites_per_cluster(
            feature_names=feature_names,
            top_n=top_n
        )


# ========== Imported revised notebook Cell 14 ==========
class ReviewerJMADKFC(JMADKFC):
    """Controlled experimental variant of the submission model."""
    def __init__(self, *args, variant='Full', **kwargs):
        if variant not in ('Full', 'NoMaskInput', 'GlobalKernel', 'NoClusterGradient'):
            raise ValueError(f'Unknown variant {variant}')
        self.variant = variant
        if variant == 'NoClusterGradient':
            kwargs['lambda_cluster'] = 0.0
        super().__init__(*args, **kwargs)
        self.epochs_run_ = 0

    def _prepare_inputs(self, X_missing, mask=None):
        X_zero, M, Xt = super()._prepare_inputs(X_missing, mask)
        if not np.isin(M, [0.0, 1.0]).all():
            raise ValueError('Non-binary missingness mask')
        if self.variant == 'NoMaskInput':
            Xt = np.concatenate([X_zero, np.zeros_like(M)], axis=1)
        return X_zero, M, Xt

    def _sample_bandwidths_np(self, Z):
        sigmas = super()._sample_bandwidths_np(Z)
        if self.variant == 'GlobalKernel':
            global_sigma = max(float(np.median(sigmas)), self.eps)
            sigmas = np.full_like(sigmas, global_sigma)
        return sigmas

    def _prototype_bandwidths_np(self, U, sigma_i):
        if self.variant == 'GlobalKernel':
            return np.full(self.n_clusters, float(sigma_i[0]), dtype=float)
        return super()._prototype_bandwidths_np(U, sigma_i)

    def _update_prototypes_np(self, Z, U, V, sigma_i, sigma_k):
        Z = np.asarray(Z, dtype=float)
        V_new = np.asarray(V, dtype=float).copy()
        U_alpha = np.asarray(U, dtype=float) ** self.alpha
        for _ in range(self.prototype_inner_iter):
            _, K = self._adaptive_kernel_np(Z, V_new, sigma_i, sigma_k)
            omega = 1.0 / (2.0 * sigma_i[:, None] * sigma_k[None, :] + self.eps)
            for k in range(self.n_clusters):
                w = U_alpha[:, k] * omega[:, k] * K[:, k]
                total = w.sum()
                if total <= self.eps:
                    # Preserve previous prototype instead of turning it into zero.
                    continue
                V_new[k] = (w[:, None] * Z).sum(axis=0) / total
        return V_new

    def fit(self, X_missing, mask=None):
        super().fit(X_missing, mask)
        self.epochs_run_ = len(self.joint_loss_history_)
        if not np.isfinite(self.X_reconstructed_).all():
            raise FloatingPointError('Non-finite reconstruction')
        if not np.isfinite(self.membership_).all():
            raise FloatingPointError('Non-finite memberships')
        if not np.allclose(self.membership_.sum(axis=1), 1.0, atol=1e-6):
            raise ArithmeticError('Fuzzy memberships do not sum to one')
        if self.membership_.shape != (X_missing.shape[0], self.n_clusters):
            raise ArithmeticError('Unexpected membership shape')
        return self


def build_reviewer_model(dataset_dim, seed, variant, pre_epochs, joint_eps, device=DEVICE):
    return ReviewerJMADKFC(
        variant=variant, n_clusters=2, alpha=ALPHA,
        latent_dim=min(10, max(2, dataset_dim // 4)), hidden_dims=(128,64),
        lambda_cluster=1.0, beta_reg=1e-6, q_neighbors=5,
        pretrain_epochs=pre_epochs, joint_epochs=joint_eps,
        learning_rate=1e-3, tol=0.0, prototype_inner_iter=1,
        random_state=int(seed), dropout=0.0, device=device, verbose=False)


# ========== Imported revised notebook Cell 16 ==========
import re

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def load_reference_dataset(dataset_name, data_dir=DATA_DIR):
    path = Path(data_dir) / (dataset_name + '.xlsx')
    if not path.is_file():
        raise FileNotFoundError(f'Missing source file {path}; do not substitute a different dataset')
    frame = pd.read_excel(path, sheet_name='Data')
    if 'Class' not in frame:
        raise ValueError(f'Missing Class column: {path}')
    if frame['Class'].isna().any():
        raise ValueError(f'Unlabeled samples in {path}; define a documented exclusion rule')
    feats = sorted([str(col) for col in frame.columns if re.fullmatch(r'M[0-9]+', str(col))],
                   key=lambda name: int(name[1:]))
    if not feats:
        raise ValueError(f'No M-indexed metabolite features in {path}')
    Xraw = frame[feats].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float64)
    if Xraw.ndim != 2 or not np.isfinite(Xraw).all():
        raise ValueError(f'Incomplete/nonfinite reference data in {path}')
    if (Xraw < 0).any():
        raise ValueError('Cannot apply log1p to negative raw metabolite intensities')
    y, labels = pd.factorize(frame['Class'].astype(str), sort=True)
    if len(labels) != 2:
        raise ValueError('Revision benchmark expects exactly 2 reference classes')
    annotations = pd.read_excel(path, sheet_name='Peak')
    if len(annotations) != len(feats):
        raise ValueError(f'Peak annotation count mismatch in {path}')
    # We do NOT infer names from an unverified row order; verify identity indices.
    if not np.array_equal(pd.to_numeric(annotations['Idx']).to_numpy(),
                          np.arange(1, len(feats)+1)):
        raise ValueError('Peak/feature index order mismatch')
    return {'name':dataset_name, 'path':str(path), 'sha256':sha256_file(path),
            'X_log':np.log1p(Xraw), 'y':y.astype(int), 'features':feats,
            'class_labels':labels.tolist(), 'annotations':annotations}


def observed_normalize(X_log, M):
    X_log = np.asarray(X_log, dtype=np.float64)
    M = np.asarray(M, dtype=bool)
    if M.shape != X_log.shape or not np.isfinite(X_log).all():
        raise ValueError('Invalid input shapes / non-finite complete log-reference')
    counts = M.sum(axis=0)
    if np.any(counts < 2):
        raise ValueError('Each feature requires at least two observed entries')
    # IMPORTANT: only observed positions contribute to fitted parameters.
    mu = np.sum(np.where(M, X_log, 0.0), axis=0) / counts
    obs_diff = np.where(M, X_log - mu[None,:], 0.0)
    var = np.sum(obs_diff ** 2, axis=0) / counts
    sd = np.sqrt(var)
    constant = sd <= 1e-12
    safe_sd = np.where(constant, 1.0, sd)
    X_obs = np.full(X_log.shape, np.nan, dtype=np.float64)
    X_obs[M] = ((X_log - mu[None,:]) / safe_sd[None,:])[M]
    # Ground truth scaling ONLY for evaluation, never for model fitting.
    X_truth_scaled = (X_log - mu[None,:]) / safe_sd[None,:]
    if not np.isfinite(X_obs[M]).all() or not np.isfinite(X_truth_scaled).all():
        raise ArithmeticError('Normalization yielded non-finite values')
    if not np.isnan(X_obs[~M]).all():
        raise AssertionError('Masked values must remain NaN')
    return {'X_obs':X_obs, 'X_truth_scaled':X_truth_scaled,
            'mu':mu,'sd':safe_sd,'observed_counts':counts,
            'constant_feature_count':int(constant.sum())}


def prepare_setting(reference, mechanism, missing_rate, seed, gamma=3.0):
    Xlog = reference['X_log']
    missing_args = {'gamma':float(gamma)} if mechanism in ('MAR','MNAR') else {}
    _, M = generate_missingness(Xlog, mechanism, missing_rate,
                                random_state=int(seed), **missing_args)
    normalized = observed_normalize(Xlog, M)
    missing_actual = float((M == 0).mean())
    if abs(missing_actual - float(missing_rate)) > 1.0/Xlog.size + 1e-12:
        raise AssertionError('Missingness rate mismatch')
    if not np.array_equal(np.isnan(normalized['X_obs']), (M == 0)):
        raise AssertionError('Observed/missing mask inconsistency')
    return {'mask':M.astype(np.uint8),'actual_rate':missing_actual, **normalized}


# ========== Imported revised notebook Cell 20 ==========
def original_baseline_by_name(name, n_clusters, seed, d, pre_epochs, joint_eps):
    # These model definitions are taken from submission notebook wrappers.
    common = dict(n_clusters=n_clusters, alpha=ALPHA, random_state=int(seed))
    if name in ('FCM-Mean','FCM-Median','FCM-KNN','FCM-MICE'):
        settings = {'FCM-Mean':dict(imputation='mean'),
                    'FCM-Median':dict(imputation='median'),
                    'FCM-KNN':dict(imputation='knn',n_neighbors=5),
                    'FCM-MICE':dict(imputation='mice',max_iter=10)}
        return ImputationFCMBaseline(**common, fcm_max_iter=300, fcm_tol=1e-5,
                                     **settings[name])
    if name in ('FCM-PDS','FCM-OCS'):
        return IncompleteFCMBaseline(**common, strategy=name.split('-')[1].lower(),
                                     max_iter=300,tol=1e-5,
                                     **({'init_completion':'mean'} if name=='FCM-OCS' else {}))
    if name == 'KFCM':
        return KFCMBaseline(**common, imputation='mean',sigma=None,max_iter=300,tol=1e-5)
    if name in ('VKFCM-K-LP-PDS','VKFCM-K-LP-OCS'):
        return VKFCM_K_LP_Baseline(**common, strategy=name.split('-')[-1].lower(),
                                  max_iter=300,tol=1e-5,
                                  **({'init_completion':'mean'} if name.endswith('OCS') else {}))
    if name == 'DFCM':
        return DFCMBaseline(**common, imputation='mean',
                            latent_dim=min(10,max(2,d//4)),hidden_dims=(128,64),
                            lambda_cluster=0.5,pretrain_epochs=pre_epochs,
                            joint_epochs=joint_eps,learning_rate=1e-3,weight_decay=1e-5,
                            tol=1e-5,verbose=False)
    if name == 'DEC':
        return DECBaseline(imputation='mean',n_clusters=n_clusters,
                           latent_dim=min(10,max(2,d//4)),hidden_dims=(128,64),
                           pretrain_epochs=pre_epochs,dec_epochs=joint_eps,
                           learning_rate=1e-3,weight_decay=1e-5,
                           student_alpha=1.0,update_interval=10,tol=1e-4,
                           random_state=int(seed),verbose=False)
    raise ValueError(f'Unknown baseline {name}')

MASK_BASELINES = {'FCM-PDS','FCM-OCS','VKFCM-K-LP-PDS','VKFCM-K-LP-OCS'}

def run_setting(dataset, settings, mechanism, rate, seed,
                variants=ABLATIONS, baselines=baseline_names):
    Xobs = settings['X_obs']
    M = settings['mask']
    Xref = settings['X_truth_scaled']
    y = dataset['y']
    rows=[]
    for name in [*variants, *baselines]:
        start=time.perf_counter()
        # Explicitly seed all stochastic implementations before method construction.
        random.seed(int(seed)); np.random.seed(int(seed)); set_torch_seed(int(seed))
        if name in ABLATIONS:
            model=build_reviewer_model(Xobs.shape[1],seed,name,
                                       pretrain_epochs,joint_epochs)
            pred=model.fit_predict(Xobs,M)
            reconstructed = model.X_reconstructed_
            epoch_count = model.epochs_run_
            family='ablation'
        else:
            model=original_baseline_by_name(name,2,seed,Xobs.shape[1],
                                            pretrain_epochs,joint_epochs)
            pred=model.fit_predict(Xobs,M) if name in MASK_BASELINES else model.fit_predict(Xobs)
            reconstructed = getattr(model,'X_reconstructed_',None)
            if reconstructed is None:
                reconstructed = getattr(model,'X_completed_',None)
            epoch_count = np.nan
            family='baseline'
        elapsed=time.perf_counter()-start
        pred=np.asarray(pred)
        if pred.shape != y.shape:
            raise ValueError(f'Label shape mismatch for {name}')
        if reconstructed is None:
            rmse=mae=np.nan
        else:
            values = np.asarray(reconstructed, dtype=float)
            if values.shape != Xref.shape or not np.isfinite(values[~M.astype(bool)]).all():
                raise ValueError(f'Reconstruction invalid for {name}')
            diff = (values-Xref)[~M.astype(bool)]
            rmse=float(np.sqrt(np.mean(diff**2)))
            mae=float(np.mean(np.abs(diff)))
        scores=external_clustering_metrics(y,pred)
        rows.append({'dataset':dataset['name'],'dataset_sha256':dataset['sha256'],
                     'mechanism':mechanism,'rate':float(rate),'seed':int(seed),
                     'method':name,'family':family,'ARI':float(scores['ARI']),
                     'NMI':float(scores['NMI']),'ACC':float(scores['ACC']),
                     'RMSE':rmse,'MAE':mae,'runtime_seconds':elapsed,
                     'epochs_run':epoch_count,'missing_fraction_actual':settings['actual_rate'],
                     'constant_features':settings['constant_feature_count'],
                     'n_samples':Xobs.shape[0],'n_features':Xobs.shape[1],
                     'n_clusters':2,'normalization':'observed_only_post_mask',
                     'gamma':3.0,'model_status':'success'})
    return rows


def run_experiments():
    rows=[]
    run_tag=f'{RUN_MODE}_observedonly'
    out=OUTPUT_DIR / run_tag
    out.mkdir(parents=True,exist_ok=True)
    env={'mode':RUN_MODE,'dataset_files':{name:references[name]['sha256'] for name in datasets},
         'seeds':list(seeds),'mechanisms':list(mechanisms),'rates':list(rates),
         'ablations':ABLATIONS,'baselines':list(baseline_names),
         'torch':torch.__version__,'sklearn':__import__('sklearn').__version__,
         'numpy':np.__version__,'python':platform.python_version(),
         'device':DEVICE,'pretrain_epochs':pretrain_epochs,'joint_epochs':joint_epochs,
         'normalization':'log1p_pre_mask; zscore_observed_only_post_mask; ddof=0',
         'gamma':3.0,'note':'New revision experiments, not submission results.'}
    (out/'run_manifest.json').write_text(json.dumps(env,indent=2,default=str))
    for dataset_name in datasets:
        dataset=references[dataset_name]
        for mechanism in mechanisms:
            for rate in rates:
                for seed in seeds:
                    st=prepare_setting(dataset,mechanism,rate,seed)
                    print(f'RUN {dataset_name} {mechanism} {rate:.0%} seed={seed}')
                    results=run_setting(dataset,st,mechanism,rate,seed)
                    rows.extend(results)
                    pd.DataFrame(rows).to_csv(out/'results_incremental.csv',index=False)
    result=pd.DataFrame(rows)
    result.to_csv(out/'results_raw.csv',index=False)
    if result.duplicated(['dataset','mechanism','rate','seed','method']).any():
        raise AssertionError('Duplicate experimental key')
    assert len(result)==len(datasets)*len(mechanisms)*len(rates)*len(seeds)*(len(ABLATIONS)+len(baseline_names))
    print('Finished:',len(result),'rows in',out)
    return result, out


# =============== ISOLATED COMMAND-LINE JOB ===============
def _sanitize_json_value(value):
    # Unavailable reconstruction metrics and baseline epoch counts must be null,
    # not IEEE NaN (JSON forbids NaN and many baseline results require it).
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_sanitize_json_value(v) for v in value]
    return value


def _write_json_atomic(path, record):
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    temporary=p.with_name(p.name+'.tmp.'+str(os.getpid()))
    payload=json.dumps(_sanitize_json_value(record),allow_nan=False,indent=2,default=str)
    temporary.write_text(payload,encoding='utf-8')
    os.replace(temporary, p)


def _run_one(args):
    global DATA_DIR, pretrain_epochs, joint_epochs, DEVICE
    DATA_DIR=Path(args.data_dir).expanduser().resolve()
    pretrain_epochs=int(args.pretrain_epochs)
    joint_epochs=int(args.joint_epochs)
    DEVICE='cpu'
    if args.method not in ABLATIONS+ORIGINAL_BASELINES:
        raise ValueError('Unknown method: '+args.method)
    dataset=load_reference_dataset(args.dataset,DATA_DIR)
    setting=prepare_setting(dataset,args.mechanism,float(args.rate),int(args.seed),gamma=float(args.gamma))
    var=[args.method] if args.method in ABLATIONS else []
    bas=[args.method] if args.method in ORIGINAL_BASELINES else []
    rec=run_setting(dataset,setting,args.mechanism,float(args.rate),int(args.seed),variants=var,baselines=bas)
    if len(rec)!=1:
        raise AssertionError('Exactly one result row expected')
    row=rec[0]
    # Store whether this was a diagnostic with shortened epochs or full training.
    row.update({'run_mode':args.mode,'protocol_version':'memorysafe_v1_observed_only',
                'pretrain_epochs_config':pretrain_epochs,'joint_epochs_config':joint_epochs,
                'worker_python':platform.python_version(),
                'numpy_version':np.__version__,'pandas_version':pd.__version__,
                'torch_version':torch.__version__,
                'sklearn_version':__import__('sklearn').__version__,
                'scipy_version':__import__('scipy').__version__})
    _write_json_atomic(args.output,row)
    print('SUCCESS',args.dataset,args.mechanism,args.rate,args.seed,args.method,'ARI',row['ARI'],flush=True)


def _preflight(args):
    global DATA_DIR
    DATA_DIR=Path(args.data_dir).expanduser().resolve()
    for name,shape in [('MTBLS161',(59,29)),('MTBLS404',(184,120))]:
        reference=load_reference_dataset(name,DATA_DIR)
        assert reference['X_log'].shape == shape
        for mechanism in ('MCAR','MAR','MNAR'):
            setting=prepare_setting(reference,mechanism,0.2,42)
            M=setting['mask'].astype(bool)
            # Mutate ground truth only at hidden positions: input/statistics must not change.
            changed=reference['X_log'].copy()
            changed[~M]+=1000
            verify=observed_normalize(changed,M)
            assert np.allclose(verify['mu'],setting['mu'])
            assert np.allclose(verify['sd'],setting['sd'])
            assert np.array_equal(verify['X_obs'],setting['X_obs'],equal_nan=True)
            assert np.all(np.isnan(setting['X_obs'][~M]))
            assert np.isfinite(setting['X_obs'][M]).all()
            assert not np.isnan(setting['X_truth_scaled']).any()
        print(name,'PASS dimensions, mask isolation, observed-only normalization',flush=True)
    print('PREFLIGHT_PASS',flush=True)


def main():
    parser=argparse.ArgumentParser(description='Memory-isolated JMADKFC runner')
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--dataset',choices=DATASET_NAMES)
    parser.add_argument('--mechanism',choices=['MCAR','MAR','MNAR'])
    parser.add_argument('--rate',type=float)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--method')
    parser.add_argument('--output')
    parser.add_argument('--pretrain-epochs',type=int,default=6)
    parser.add_argument('--joint-epochs',type=int,default=8)
    parser.add_argument('--mode',choices=['diagnostic','full'],default='diagnostic')
    parser.add_argument('--gamma',type=float,default=3.0)
    args=parser.parse_args()
    if args.preflight:
        _preflight(args)
    else:
        needed=['dataset','mechanism','rate','seed','method','output']
        if any(getattr(args,key) is None for key in needed):
            parser.error('Missing required job parameters')
        _run_one(args)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
