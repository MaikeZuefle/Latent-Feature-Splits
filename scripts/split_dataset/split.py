import argparse
import logging
import os
import time
import warnings

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from closest_clusters import get_closest
from clustering import get_data_frame, k_means
from get_indices import get_cluster_indices, test_train_indices
from subsets import get_subsets


def split_dataset(
    hidden_files,
    hidden_labels,
    folder_to_save,
    desired_hate_ratio=0.5,
    desired_offensive_ratio=None,
    test_ratio=0.1,
    max_clusters=50,
    umap=False,
    seed=42,
    split="subset",
    n_classes=2,
):
    """ 
    Splits the dataset using either the SUBSET-SUM-SPLIT or the CLOSEST-SPLIT 
    given the hidden representations of the data.
    """

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Start: {time.strftime('%H:%M:%S', time.localtime())}")
    os.makedirs(folder_to_save)

    if n_classes == 2:
        if desired_offensive_ratio:
            raise Exception("Offensive ratio given but only 2 classes!")
        if desired_hate_ratio > 1:
            raise Exception("Hate ratio must be < 1")

    elif n_classes == 3:
        if not desired_offensive_ratio:
            raise Exception("3 classes given, but no desired offensive ratio!")
        if desired_hate_ratio > 1 or desired_offensive_ratio > 1:
            raise Exception("Hate/Offensive ratio must be < 1")

    else:
        raise Exception(f"Only implemented for 2 or 3 classes, not {n_classes}!")

    hiddens, n_hiddens, indices, labels = get_data_frame(
        hidden_files, hidden_labels, umap=umap
    )

    if n_classes == 2:
        hate_target_n = int(round(desired_hate_ratio * test_ratio * n_hiddens, 0))
        noHate_target_n = int(
            round((1 - desired_hate_ratio) * test_ratio * n_hiddens, 0)
        )
        target = [hate_target_n, noHate_target_n]

    elif n_classes == 3:
        hate_target_n = int(round(desired_hate_ratio * test_ratio * n_hiddens, 0))
        offensive_target_n = int(
            round(desired_offensive_ratio * test_ratio * n_hiddens, 0)
        )
        noHate_target_n = int(
            round(
                (1 - (desired_hate_ratio + desired_offensive_ratio))
                * test_ratio
                * n_hiddens,
                0,
            )
        )
        target = [hate_target_n, offensive_target_n, noHate_target_n]

    if split == "subset":
        cluster_range = [i for i in range(2, max_clusters + 1)][::3]
    elif split == "closest":
        cluster_range = [i for i in range(2, max_clusters + 1)]
    else:
        raise Exception(f"Unknown splitting method {split}!")

    results = []
    for n_clusters in cluster_range:
        logging.info(
            f"{time.strftime('%H:%M:%S', time.localtime())} N Clusters {n_clusters}"
        )
        # clustering
        cluster_dict, avg_dist, all_tuples, cluster_centers = k_means(
            hiddens, indices, labels, n_clusters, seed=seed, n_classes=n_classes
        )

        # find test_tuples for split with desired test ratio and hate ratio
        if split == "subset":
            test_tuples, closest_solution, difference = get_subsets(
                all_tuples, target, n_clusters, n_classes
            )
            results.append(
                [
                    n_clusters,
                    closest_solution,
                    difference,
                    test_tuples,
                    cluster_dict,
                    all_tuples,
                ]
            )

        elif split == "closest":
            test_clusters, closest_solution, difference = get_closest(
                all_tuples, target, cluster_centers, n_classes
            )
            results.append(
                [
                    n_clusters,
                    closest_solution,
                    difference,
                    test_clusters,
                    cluster_dict,
                    all_tuples,
                ]
            )

    best_n, closest_solution, dif, chosens, cluster_dict, all_tuples = min(
        results, key=lambda x: x[2]
    )

    test_indices, train_indices, test_clusters, train_clusters = test_train_indices(
        closest_solution,
        chosens,
        target,
        cluster_dict,
        all_tuples,
        n_hiddens,
        hiddens,
        labels,
        split=split,
    )

    test_cluster_indices, train_cluster_indices = get_cluster_indices(
        test_clusters, train_clusters, cluster_dict
    )

    name = f"{split}_split_seed_{seed}"
    if umap:
        name += f"_umap{umap}"

    with open(folder_to_save + "/" + name + "_indices.txt", "w+") as test_file:
        test_file.write(f"test: {test_indices}\n")
        test_file.write(f"train: {train_indices}")

    with open(folder_to_save + "/cluster_indices.txt", "w+") as cluster_indices_file:
        cluster_indices_file.write("Test Clusters:\n")
        for i, test_i in enumerate(test_cluster_indices):
            cluster_indices_file.write(f"Cluster {i+1}:\n")
            cluster_indices_file.write(f"{test_i}\n\n")

        cluster_indices_file.write("Train Clusters:\n")
        for i, train_i in enumerate(train_cluster_indices):
            cluster_indices_file.write(f"Cluster {i+1}:\n")
            cluster_indices_file.write(f"{train_i}\n\n")

    with open(folder_to_save + "/overview.txt", "w+") as overview:
        overview.write(
            f"filename: {name}, n clusters: {best_n}, diff: {dif}, umap: {umap}\n\n"
        )

    logging.info(
        f"{time.strftime('%H:%M:%S', time.localtime())} Done! Saved to {folder_to_save}"
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--hidden_files", type=str, help="path to the hidden files, saved as pkl"
)
parser.add_argument(
    "--hidden_labels",
    type=str,
    help="path to the labels for the data, corresponding to the hidden file, saved as pkl.",
)
parser.add_argument(
    "--folder_to_save",
    type=str,
    help="folder where to save the indices for the new data split",
)
parser.add_argument("--split", type=str, help="splitting method: closest or subset")
parser.add_argument(
    "--test_ratio", type=float, help="desired test set ratio", default=0.1
)
parser.add_argument(
    "--n_classes", type=int, help="Number of classes considered in the dataset"
)
parser.add_argument(
    "--desired_hate_ratio", type=float, help="desired hate ratio in test set"
)
parser.add_argument(
    "--desired_offensive_ratio",
    type=float,
    help="desired offensive ratio in test set, if applicable",
    default=None,
)
parser.add_argument(
    "--max_clusters",
    type=int,
    help="maximum number of clusters considered in the algorithm",
    default=50,
)
parser.add_argument("--seed", type=int, help="Cluster seed", default=42)
parser.add_argument(
    "--umap",
    type=int,
    help="desired dimension for dimensionality reduction by umap",
    default=False,
)
parser.add_argument(
    "--data_path",
    type=str,
    help="path to data to retrieve topics, if false, no topics are retrieved",
    default=False,
)

args = parser.parse_args()


if __name__ == "__main__":
    hidden_files = args.hidden_files
    hidden_labels = args.hidden_labels
    folder_to_save = args.folder_to_save
    split = args.split
    test_ratio = args.test_ratio
    n_classes = args.n_classes
    desired_hate_ratio = args.desired_hate_ratio
    desired_offensive_ratio = args.desired_offensive_ratio
    max_clusters = args.max_clusters
    seed = args.seed
    umap = args.umap

    split_dataset(
        hidden_files,
        hidden_labels,
        folder_to_save,
        desired_hate_ratio=desired_hate_ratio,
        desired_offensive_ratio=desired_offensive_ratio,
        test_ratio=test_ratio,
        max_clusters=max_clusters,
        umap=umap,
        seed=seed,
        split=split,
        n_classes=n_classes,
    )
