import logging
import pickle
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.cluster import KMeans
from umap import UMAP

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
id2label = {0: "noHate", 1: "hate", 2: "offensive"}


def open_pickles(file, tensors=False):
    """
    Opens pickles files with labels and hidden representations and returns
    np.array
    """
    l = []
    l2 = []
    with open(file, "rb") as pickle_hiddens:
        while True:
            try:
                if tensors:
                    h = np.array(pickle.load(pickle_hiddens).cpu())
                    l.append(h)
                else:
                    index, label = pickle.load(pickle_hiddens)
                    l.append(index)
                    l2.append(id2label[label.cpu().item()])

            except EOFError:
                break

    if tensors:
        return np.array(l)
    return np.array(l), np.array(l2)


def get_data_frame(hidden_files, hidden_labels, umap=False):
    """
    Converts hidden representations and labels to np.arrays 
    and reduces dimensionality of representations if umap is not False
    """
    hiddens = open_pickles(hidden_files, tensors=True)
    logging.info(
        f"{time.strftime('%H:%M:%S', time.localtime())}: Hiddens loaded from {hidden_files}!"
    )
    if umap:
        umap_50d = UMAP(
            n_components=umap,
            init="random",
            random_state=42,
            min_dist=0.0,
            metric="cosine",
            n_neighbors=15,
        )
        hiddens = umap_50d.fit_transform(hiddens)
        logging.info(
            f"{time.strftime('%H:%M:%S', time.localtime())}: UMAP Dim reduction to dim {umap}!"
        )

    indices, labels = open_pickles(hidden_labels)
    logging.info(
        f"{time.strftime('%H:%M:%S', time.localtime())}: Indices and labels loaded from {hidden_labels}!"
    )

    n_hiddens = len(hiddens)
    return hiddens, n_hiddens, indices, labels


def k_means(hiddens, indices, hate_labels, n_clusters, seed=42, n_classes=2):
    """
    Clusters the hidden representations using k-means
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(
        hiddens
    )
    clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    X_dist = kmeans.transform(hiddens) ** 2
    center_dists = np.array([X_dist[i][x] for i, x in enumerate(clusters)])

    assert len(clusters) == len(indices) == len(hate_labels) == len(center_dists)

    cluster_dict = defaultdict(list)

    if n_classes == 2:
        cluster_hate = [
            [0, 0] for cluster_label in range(n_clusters)
        ]  #  [n_hate, n_noHate]
        for cluster_label, index, hate_label, center_dist in zip(
            clusters, indices, hate_labels, center_dists
        ):
            cluster_dict[cluster_label].append((int(index), hate_label, center_dist))
            if hate_label == "hate":
                cluster_hate[cluster_label][0] += 1
            else:
                cluster_hate[cluster_label][1] += 1

    elif n_classes == 3:
        cluster_hate = [
            [0, 0, 0] for cluster_label in range(n_clusters)
        ]  #  [n_hate, n_offensive, n_noHate]
        for cluster_label, index, hate_label, center_dist in zip(
            clusters, indices, hate_labels, center_dists
        ):
            cluster_dict[cluster_label].append((int(index), hate_label, center_dist))
            if hate_label == "hate":
                cluster_hate[cluster_label][0] += 1
            elif hate_label == "offensive":
                cluster_hate[cluster_label][1] += 1
            elif hate_label == "noHate":
                cluster_hate[cluster_label][2] += 1
            else:
                raise Exception(f"Unknown cluster label {hate_label}")

    av_dists = []
    for cluster_label, members in cluster_dict.items():
        av_dist = np.mean([m[2] for m in members])
        av_dists.append(av_dist)
        cluster_dict[cluster_label] = {"av_dist_center": av_dist, "members": members}

    avg_dist_centers = np.mean(av_dist)
    return cluster_dict, avg_dist_centers, cluster_hate, cluster_centers
