""" This module contains the code used to calculate the frechet inception distance
and the inception score when the inception features and logits was retrieved.
 """

# %%
import os
import warnings

import numpy as np
import scipy.stats
import tensorflow as tf
from scipy import linalg
from tqdm import tqdm

# %%


def inception_score(logits):
    """ Given the logits from the inception classifier
    This function calculates the inception score. """

    # Remove outputs 1001-1008 as they are not used to calculate IS
    # see https://github.com/openai/improved-gan/issues/29
    logits = logits[:, :1000]

    # Get the probabilities
    p = tf.nn.softmax(logits, axis=1)

    marginal_dist = tf.reduce_mean(p, axis=0)

    kl_divergences = []
    for cond_prob in p:
        kl_divergence = scipy.stats.entropy(cond_prob, marginal_dist)
        kl_divergences.append(kl_divergence)
    kl_divergences = np.array(kl_divergences)

    is_score = np.mean(kl_divergences)
    is_score = np.exp(is_score)
    return np.mean(is_score)

# %%


def evaluate_is(eval_dir):
    inception_scores = np.array([])
    for fname in tqdm(os.listdir(eval_dir), unit="split"):
        path = os.path.join(eval_dir, fname)
        logits = np.genfromtxt(path)
        is_mean = inception_score(logits)
        inception_scores = np.append(inception_scores, is_mean)

    return np.mean(inception_scores), np.std(inception_scores)

# %%


def assemble_splits(eval_dir):
    features = []
    for fname in tqdm(os.listdir(eval_dir), unit="split"):
        path = os.path.join(eval_dir, fname)
        features_split = np.genfromtxt(path)
        features.append(features_split)
    features = np.concatenate(features, axis=0)
    return features

# %%


def calculate_fid(real_samps, fake_samps):
    """ Calculate the Frétchet Inception distance between two sets.
    Evaluate over all training samples and 50000 fake samples. 

    Take in numpy features from the inception pool_3 layer.

    Code from original FID paper: https://github.com/bioinf-jku/TTUR """

    # Calc statistics
    mu1 = np.mean(real_samps, axis=0)
    mu2 = np.mean(fake_samps, axis=0)

    sigma1 = np.cov(real_samps, rowvar=False)
    sigma2 = np.cov(fake_samps, rowvar=False)

    # Check shapes
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Calculate FID
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid




# %%

# Calculated the bias in the foraminifera distribution
real_features = np.genfromtxt("path")

fids = []
for i in tqdm(range(50)):
    # Shuffle rows and sample two sets of 1000 features
    np.random.shuffle(real_features)
    set1, set2 = real_features[:1318], real_features[1318:]

    # Measure the FID between these two sets
    fid = calculate_fid(set1, set2)
    fids.append(fid)

fids = np.array(fids)
print(np.mean(fids), np.std(fids))

# %%

# CIFAR-10 EMA
eval_dir = r"path"
is_mean, is_std = evaluate_is(eval_dir)
print(f"Inception scores model EMA: {is_mean}, std {is_std}")

# %%
# # FID
eval_dir_fake = r"path"
fake_features = assemble_splits(eval_dir_fake)

real_features = np.genfromtxt(r"path")

fid = calculate_fid(real_features, fake_features)
print("FID: ", fid)
