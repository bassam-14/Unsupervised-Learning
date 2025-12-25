#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np


# In[ ]:


def calculate_euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


# In[ ]:


def pairwise_distances(X):
    n = X.shape[0]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distances[i, j] = calculate_euclidean(X[i], X[j])
    return distances


# In[ ]:


## functions expectes predicted_labels to be a vector/array with size equal
# to the number of points and each entry has the assigned cluster corresponding to this point


# In[ ]:


# silhouette score -- > How well each data point
# fits within its assigned cluster compared to other clusters.
# ranges from -1 to 1
# it measuers cohesion between points in same cluster and seperation between
# points in diff. clusters


# In[ ]:


# a is cohesion
# b is seperation


def silhouette_score(X, predicted_labels):
    D = pairwise_distances(X)
    unique_labels = np.unique(predicted_labels)
    n = len(predicted_labels)
    s = np.zeros(n)

    for i in range(n):
        same_cluster = predicted_labels == predicted_labels[i]
        # other_clusters = predicted_labels != predicted_labels[i]

        a = np.mean(D[i, same_cluster]) if np.sum(same_cluster) > 1 else 0

        b = np.inf

        # calculate seperation (mean sum of distances)
        # between our current cluster and all other cluster and take the min
        for label in unique_labels:
            if label != predicted_labels[i]:
                mask = predicted_labels == label
                b = min(b, np.mean(D[i, mask]))

        s[i] = (b - a) / max(a, b)

    return np.mean(s)


# In[ ]:


# Davies–Bouldin Index
# What it measures:
# Average similarity between each cluster and its most similar cluster.


# In[ ]:


# kol ma kant asghar kant ahsn
# because i am dividing within cluster scatter by between clusters seperation
# so i want the seperation (den.) to be high and scatter within cluster (num.) to be low


def davies_bouldin_index(X, predicted_labels):
    unique_labels = np.unique(predicted_labels)
    k = len(unique_labels)

    # mean of each diff. cluster
    centroids = np.zeros((k, X.shape[1]))
    for i, label in enumerate(unique_labels):
        centroids[i] = X[predicted_labels == label].mean(axis=0)

    # we calc. the spread of points around cluster mean (for each diff. cluster)
    S = np.zeros(k)
    for i, label in enumerate(unique_labels):
        cluster = X[predicted_labels == label]
        distances = np.linalg.norm(cluster - centroids[i], axis=1)
        # S_i is the average distance of points to their centroid
        S[i] = np.mean(distances)

    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                # Distance between centroids
                dist_centroids = np.linalg.norm(centroids[i] - centroids[j])
                R[i, j] = (S[i] + S[j]) / dist_centroids

    # calc. mean for highest overlap with other clusters
    return np.mean(np.max(R, axis=1))


# In[ ]:


# Calinski–Harabasz Index
# what it measures:
# ratio of between-cluster dispersion to within-cluster dispersion.


# In[ ]:


# higher is better as within cluster despersion is in denmo.
# its used for choosing k --> as u increase k, within cluster despersion decrease
# so higher chi
def calinski_harabasz_index(X, predicted_labels):
    n, d = X.shape
    unique_labels = np.unique(predicted_labels)
    k = len(unique_labels)

    overall_mean = X.mean(axis=0)

    B = 0  # between-cluster dispersion
    W = 0  # within-cluster dispersion

    for label in unique_labels:
        cluster = X[predicted_labels == label]
        cluster_mean = cluster.mean(axis=0)
        # one sum for all cluster and within each cluster sum, we sum over all points
        B += len(cluster) * np.sum((cluster_mean - overall_mean) ** 2)
        W += np.sum((cluster - cluster_mean) ** 2)

    return (B / (k - 1)) / (W / (n - k))


# In[ ]:


# within cluster sum of squares
def wcss(X, predicted_labels):
    total = 0
    for label in np.unique(predicted_labels):
        cluster = X[predicted_labels == label]
        centroid = cluster.mean(axis=0)
        total += np.sum((cluster - centroid) ** 2)
    return total


# In[ ]:


# ranges from -1 to 1 where higher is better
def adjusted_rand_index(y_true, y_pred):
    n = len(y_true)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    contingency = np.zeros((len(labels_true), len(labels_pred)))

    for i, lt in enumerate(labels_true):
        for j, lp in enumerate(labels_pred):
            contingency[i, j] = np.sum((y_true == lt) & (y_pred == lp))

    def comb2(x):
        return x * (x - 1) / 2

    sum_comb = np.sum(comb2(contingency))
    sum_rows = np.sum(comb2(contingency.sum(axis=1)))
    sum_cols = np.sum(comb2(contingency.sum(axis=0)))

    expected = sum_rows * sum_cols / comb2(n)
    max_index = 0.5 * (sum_rows + sum_cols)

    return (sum_comb - expected) / (max_index - expected)


# In[ ]:


def normalized_mutual_info(y_true, y_pred):
    n = len(y_true)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    MI = 0
    for lt in labels_true:
        for lp in labels_pred:
            p_xy = np.sum((y_true == lt) & (y_pred == lp)) / n
            if p_xy > 0:
                p_x = np.sum(y_true == lt) / n
                p_y = np.sum(y_pred == lp) / n
                MI += p_xy * np.log(p_xy / (p_x * p_y))

    def entropy(labels):
        H = 0
        for l in np.unique(labels):
            p = np.sum(labels == l) / n
            H -= p * np.log(p)
        return H

    return MI / ((entropy(y_true) + entropy(y_pred)) / 2)


# In[ ]:


# Purity
# What it measures:
# Extent to which each cluster contains points from a single class.
# but this means that if a cluster has smaller no. of points
# it will have higher prob. of better purity that is not necessary true


# In[ ]:


# measuers the count of points of the majority class in each cluster / total points
# y_true --> ground truth table (labels) for each point
# y_pred --> no. of cluster assigned for each point
def purity_score(y_true, y_pred):
    total = 0
    for cluster in np.unique(y_pred):
        labels_in_cluster = y_true[
            y_pred == cluster
        ]  # getting the actual labels of points assigned to same current cluster
        _, counts = np.unique(
            labels_in_cluster, return_counts=True
        )  # counts no. of apperances of each non-ve no.  (index==no., value == apperances)
        total += np.max(counts)  # total of majority contributions
    return total / len(y_true)


# In[ ]:


def confusion_matrix(y_true, y_pred):
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)

    cm = np.zeros((len(labels_true), len(labels_pred)), dtype=int)

    for i, t in enumerate(labels_true):
        for j, p in enumerate(labels_pred):
            cm[i, j] = np.sum((y_true == t) & (y_pred == p))

    return cm
