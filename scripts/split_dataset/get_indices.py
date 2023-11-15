import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def subset_indices(
    closest_solution, chosens, target, cluster_dict, all_tuples, n_hiddens
):
    """ 
    Retrieve indices of train and test set based on the subset-sum-split. 
    Also add more data to the test set if the subset-sum-split did not produce the exact
    test ratio. 
    """
    test_tuples = chosens
    test_clusters = []

    # add clusters found by test_tuples sum to test set
    test_indices = []
    for s in test_tuples:
        suitable_tuples = [
            i for i, c in enumerate(all_tuples) if c and np.all(np.equal(s, c))
        ]
        # if there are multiple clusters with the same hate ratio, use the cluster with the smallest avg. distance to center
        best_i = min(
            [(i, cluster_dict[i]["av_dist_center"]) for i in suitable_tuples],
            key=lambda x: x[1],
        )[0]
        test_clusters.append(best_i)
        test_indices += [m[0] for m in cluster_dict[best_i]["members"]]
        all_tuples[best_i] = None

    train_clusters = [c for c in cluster_dict.keys() if c not in test_clusters]

    # add from other clusters to test set so that the hate ratio is correct
    if len(target) == 2:
        needed_hate_for_test = target[0] - closest_solution[0]
        needed_no_hate_for_test = target[1] - closest_solution[1]

        # find smallest cluster with enough hate/no hate examples (if exists)
        to_test = []
        for i, t in enumerate(all_tuples):
            if t:
                if t[0] >= needed_hate_for_test and t[1] >= needed_no_hate_for_test:
                    to_test.append([i, t])

        if to_test:
            to_test = min(to_test, key=lambda x: x[1][0] + x[1][1])
            hates = [
                x[0] for x in cluster_dict[to_test[0]]["members"] if x[1] == "hate"
            ]
            nohates = [
                x[0] for x in cluster_dict[to_test[0]]["members"] if x[1] != "hate"
            ]

        # randomly from all clusters
        else:
            hates, nohates = [], []
            for cluster_id in cluster_dict.keys():
                if not all_tuples[cluster_id]:  # cluster is already in test
                    continue
                else:
                    for ex in cluster_dict[cluster_id]["members"]:
                        if ex[1] == "hate":
                            hates.append(ex[0])
                        else:
                            nohates.append(ex[0])

        # sample randomly
        np.random.seed(42)
        to_test_hate = list(
            np.random.choice(hates, size=needed_hate_for_test, replace=False)
        )
        to_test_nohate = list(
            np.random.choice(nohates, size=needed_no_hate_for_test, replace=False)
        )

        test_indices += to_test_hate
        test_indices += to_test_nohate
        train_indices = [x for x in range(n_hiddens) if x not in test_indices]

    elif len(target) == 3:
        needed_hate_for_test = target[0] - closest_solution[0]
        needed_offensive_for_test = target[1] - closest_solution[1]
        needed_no_hate_for_test = target[2] - closest_solution[2]

        # find smallest cluster with enough hate/no hate examples (if exists)
        to_test = []
        for i, t in enumerate(all_tuples):
            if t:
                if (
                    t[0] >= needed_hate_for_test
                    and t[1] >= needed_offensive_for_test
                    and t[2] >= needed_no_hate_for_test
                ):
                    to_test.append([i, t])

        if to_test:
            to_test = min(to_test, key=lambda x: x[1][0] + x[1][1])
            hates = [
                x[0] for x in cluster_dict[to_test[0]]["members"] if x[1] == "hate"
            ]
            offensives = [
                x[0] for x in cluster_dict[to_test[0]]["members"] if x[1] == "offensive"
            ]
            nohates = [
                x[0] for x in cluster_dict[to_test[0]]["members"] if x[1] == "noHate"
            ]

        # randomly from all clusters
        else:
            hates, offensives, nohates = [], [], []
            for cluster_id in cluster_dict.keys():
                if not all_tuples[cluster_id]:  # cluster is already in test
                    continue
                else:
                    for ex in cluster_dict[cluster_id]["members"]:
                        if ex[1] == "hate":
                            hates.append(ex[0])
                        elif ex[1] == "offensive":
                            offensives.append(ex[0])
                        else:
                            nohates.append(ex[0])

        # sample randomly
        np.random.seed(42)
        to_test_hate = list(
            np.random.choice(hates, size=needed_hate_for_test, replace=False)
        )
        to_test_offensive = list(
            np.random.choice(offensives, size=needed_offensive_for_test, replace=False)
        )
        to_test_nohate = list(
            np.random.choice(nohates, size=needed_no_hate_for_test, replace=False)
        )

        test_indices += to_test_hate
        test_indices += to_test_nohate
        test_indices += to_test_offensive
        train_indices = [x for x in range(n_hiddens) if x not in test_indices]

    return test_indices, train_indices, test_clusters, train_clusters


def closest_indices(
    closest_solution,
    chosens,
    target,
    cluster_dict,
    all_tuples,
    n_hiddens,
    hiddens,
    hidden_labels,
):
    """ 
    Retrieve indices of train and test set based on the closest-split. 
    Also add more data to the test set if the closest-split did not produce the exact
    test ratio. 
    """
    test_clusters = chosens
    test_indices = []

    for t in test_clusters:
        test_indices += [m[0] for m in cluster_dict[t]["members"]]

    test_center = np.mean([hiddens[i] for i in test_indices], axis=0)

    if len(target) == 2:
        hate_dists, no_hate_dists = [], []
        for i, h in enumerate(hiddens):
            if i in test_indices:
                continue
            center = np.expand_dims(test_center, axis=0)
            other = np.expand_dims(h, axis=0)
            dist = cosine_similarity(center, other)
            if hidden_labels[i] == "hate":
                hate_dists.append((i, dist))
            else:
                no_hate_dists.append((i, dist))

        needed_hate_for_test = target[0] - closest_solution[0]
        needed_no_hate_for_test = target[1] - closest_solution[1]

        new_hates = sorted(hate_dists, key=lambda x: x[1], reverse=True)[
            :needed_hate_for_test
        ]
        new_nohates = sorted(no_hate_dists, key=lambda x: x[1], reverse=True)[
            :needed_no_hate_for_test
        ]
        new_indices = [h[0] for h in new_hates + new_nohates]

    elif len(target) == 3:
        hate_dists, offensive_dists, no_hate_dists = [], [], []
        for i, h in enumerate(hiddens):
            if i in test_indices:
                continue
            center = np.expand_dims(test_center, axis=0)
            other = np.expand_dims(h, axis=0)
            dist = cosine_similarity(center, other)
            if hidden_labels[i] == "hate":
                hate_dists.append((i, dist))
            elif hidden_labels[i] == "offensive":
                offensive_dists.append((i, dist))
            elif hidden_labels[i] == "noHate":
                no_hate_dists.append((i, dist))
            else:
                raise Exception

        assert len(test_indices) == len(set(test_indices))

        needed_hate_for_test = target[0] - closest_solution[0]
        needed_offensive_for_test = target[1] - closest_solution[1]
        needed_no_hate_for_test = target[2] - closest_solution[2]

        new_hates = sorted(hate_dists, key=lambda x: x[1], reverse=True)[
            :needed_hate_for_test
        ]
        new_offensives = sorted(offensive_dists, key=lambda x: x[1], reverse=True)[
            :needed_offensive_for_test
        ]
        new_nohates = sorted(no_hate_dists, key=lambda x: x[1], reverse=True)[
            :needed_no_hate_for_test
        ]

        new_indices = [h[0] for h in new_hates + new_nohates + new_offensives]

        assert len(new_indices) == len(set(new_indices))

    else:
        raise Exception(f"Not implemented for {len(target)} classes!")

    test_indices += new_indices

    assert len(test_indices) == sum(target) == len(set(test_indices))

    train_indices = [x for x in range(n_hiddens) if x not in test_indices]

    train_clusters = [c for c in cluster_dict.keys() if c not in test_clusters]
    return test_indices, train_indices, test_clusters, train_clusters


def test_train_indices(
    closest_solution,
    chosens,
    target,
    cluster_dict,
    all_tuples,
    n_hiddens,
    hiddens,
    hidden_labels,
    split="subset",
):

    if split == "subset":
        test_indices, train_indices, test_clusters, train_clusters = subset_indices(
            closest_solution, chosens, target, cluster_dict, all_tuples, n_hiddens
        )

    elif split == "closest":
        test_indices, train_indices, test_clusters, train_clusters = closest_indices(
            closest_solution,
            chosens,
            target,
            cluster_dict,
            all_tuples,
            n_hiddens,
            hiddens,
            hidden_labels,
        )

    assert len(test_indices) == sum(target)
    assert len(test_indices) + len(train_indices) == n_hiddens

    return test_indices, train_indices, test_clusters, train_clusters


def get_cluster_indices(test_clusters, train_clusters, cluster_dict):
    test_cluster_indices = []
    train_cluster_indices = []

    for cluster in test_clusters:
        indices = [i[0] for i in cluster_dict[cluster]["members"]]
        test_cluster_indices.append(indices)

    for cluster in train_clusters:
        indices = [i[0] for i in cluster_dict[cluster]["members"]]
        train_cluster_indices.append(indices)

    return test_cluster_indices, train_cluster_indices
