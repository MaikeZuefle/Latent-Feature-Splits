import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_closest(all_tuples, target, cluster_centers, n_classes):
    """
    CLOSEST-SPLIT, nearest neighbor clustering with clusters that are far away from the the center 
    """
    ccc = np.mean(cluster_centers, axis=0)
    dist_to_centroid = [
        cosine_similarity(np.expand_dims(c, axis=0), np.expand_dims(ccc, axis=0))
        for c in cluster_centers
    ]
    dist_to_centroid = np.argsort(dist_to_centroid, axis=0)

    if n_classes == 2:
        i = 0
        while True:
            try:
                cluster_far_away = dist_to_centroid[i].item()
            except IndexError:  # all clusters are too big for test set
                return [], [0, 0], sum(target)

            hate = all_tuples[cluster_far_away][0]
            nohate = all_tuples[cluster_far_away][1]
            if hate <= target[0] and nohate <= target[1]:
                break
            else:  # cluster too big for test set
                i += 1

        test_clusters = [cluster_far_away]
        hate_ratio = np.array(all_tuples[cluster_far_away])
        dist_to_clusters = np.full([len(cluster_centers), len(cluster_centers)], None)

        while (
            True
        ):  # as long as there are close clusters with the right amount of hate/no hate examples
            closest_clusters = []
            for t in test_clusters:

                for i, center in enumerate(cluster_centers):
                    if i == t:
                        dist_to_clusters[t][i] = np.array([[-1]])
                        dist_to_clusters[i][t] = np.array([[-1]])
                    else:
                        if dist_to_clusters[i][t] == None:
                            center = np.expand_dims(center, axis=0)
                            other = np.expand_dims(cluster_centers[t], axis=0)
                            dist = cosine_similarity(center, other)
                            dist_to_clusters[i][t] = dist
                            dist_to_clusters[t][i] = dist

                closest_dist = np.max(dist_to_clusters[t]).item()
                closest_cluster = np.argmax(dist_to_clusters[t])
                closest_clusters.append((closest_cluster, closest_dist))

            # find cluster that is the closest to any of the test clusters

            test_cluster = max(closest_clusters, key=lambda x: x[1])[0]
            new_ratio = hate_ratio + np.array(all_tuples[test_cluster])

            if new_ratio[0] <= target[0] and new_ratio[1] <= target[1]:
                hate_ratio = new_ratio
                test_clusters.append(test_cluster)
                for test_c in test_clusters:
                    dist_to_clusters[test_c][test_cluster] = np.array([[-1]])
                    dist_to_clusters[test_cluster][test_c] = np.array([[-1]])

            else:
                break

    elif n_classes == 3:

        i = 0
        while True:
            try:
                cluster_far_away = dist_to_centroid[i].item()
            except IndexError:  # all clusters are too big for test set
                return [], [0, 0, 0], sum(target)

            hate = all_tuples[cluster_far_away][0]
            offensive = all_tuples[cluster_far_away][1]
            nohate = all_tuples[cluster_far_away][2]

            if hate <= target[0] and offensive <= target[1] and nohate <= target[2]:
                break
            else:  # cluster too big for test set
                i += 1

        test_clusters = [cluster_far_away]
        hate_ratio = np.array(all_tuples[cluster_far_away])
        dist_to_clusters = np.full([len(cluster_centers), len(cluster_centers)], None)

        while (
            True
        ):  # as long as there are close clusters with the right amount of hate/no hate examples
            closest_clusters = []
            for t in test_clusters:

                for i, center in enumerate(cluster_centers):
                    if i == t:
                        dist_to_clusters[t][i] = np.array([[-1]])
                        dist_to_clusters[i][t] = np.array([[-1]])
                    else:
                        if dist_to_clusters[i][t] == None:
                            center = np.expand_dims(center, axis=0)
                            other = np.expand_dims(cluster_centers[t], axis=0)
                            dist = cosine_similarity(center, other)
                            dist_to_clusters[i][t] = dist
                            dist_to_clusters[t][i] = dist

                closest_dist = np.max(dist_to_clusters[t]).item()
                closest_cluster = np.argmax(dist_to_clusters[t])
                closest_clusters.append((closest_cluster, closest_dist))

            # find cluster that is the closest to any of the test clusters

            test_cluster = max(closest_clusters, key=lambda x: x[1])[0]

            new_ratio = hate_ratio + np.array(all_tuples[test_cluster])

            if (
                new_ratio[0] <= target[0]
                and new_ratio[1] <= target[1]
                and new_ratio[2] <= target[2]
            ):
                hate_ratio = new_ratio
                test_clusters.append(test_cluster)
                for test_c in test_clusters:
                    dist_to_clusters[test_c][test_cluster] = np.array([[-1]])
                    dist_to_clusters[test_cluster][test_c] = np.array([[-1]])

            else:
                break

    else:
        raise Exception(
            f"Closest Clustering not implemented for {n_classes} classes, only for 2 or 3!"
        )

    return test_clusters, hate_ratio, sum(target - hate_ratio)
