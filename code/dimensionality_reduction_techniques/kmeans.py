import numpy as np
import pymongo
import numpy as np
import os
import random

def cluster_calc(x: list, cent_no: int):
    cent_temp = centroids = initialize_kmeans_plusplus(np.array(x), cent_no).tolist()
    cluster_array = [[] for i in range(cent_no)]
    repeat_flag = True
    while repeat_flag:
        cluster_array = [[] for i in range(cent_no)]
        for i in x:
            min_dist, min_j = vector_euclid(i, centroids[0]), 0
            for j in range(len(centroids)):
                temp_dist = vector_euclid(i, centroids[j])
                if temp_dist < min_dist:
                    min_dist = temp_dist
                    min_j = j
            cluster_array[min_j].append(i)  
        centroids = [np.mean(i, axis=0).tolist() for i in cluster_array]
        if centroid_check(cent_temp, centroids):
            repeat_flag = False
        cent_temp = centroids
    return centroids, cluster_array

def initialize_kmeans_plusplus(points, K, c_dist=np.linalg.norm):
    centers = []
    centers.append(points[np.random.randint(points.shape[0])])
    for _ in range(1, K):
        dist_sq = np.array([min([c_dist(c - x)**2 for c in centers]) for x in points])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        centers.append(points[i])
    return np.array(centers)

def vector_euclid(x, y):
    return np.linalg.norm(np.array(x) - np.array(y))

def centroid_check(prev, new):
    return all(np.allclose(p, n) for p, n in zip(prev, new))

def kmeans(data_matrix, k):
    centroids, cluster_array = cluster_calc(data_matrix, k)
    print(f'shape of centroids: {np.array(centroids).shape}')
    data_matrix_ls = []
    for i in data_matrix:
        temp = []
        for j in centroids:
            temp.append(vector_euclid(i, j))
        data_matrix_ls.append(temp)
    return np.array(data_matrix_ls)

def weight_calc(vector: list)-> float:
    return np.linalg.norm(vector)

def calculateImageIDWeightPairs(latent_semantics, feature_descriptor, k, type):
    res_matrix = []
    for i in range(len(latent_semantics)):
        res_matrix.append((i*2, weight_calc(latent_semantics[i])))    
    res_matrix = sorted(res_matrix, key=lambda x: x[1], reverse=True)
    with open(f"{os.path.join(os.getcwd(), f'../outputs/weight_pairs/{4}_{feature_descriptor}_{k}_{type}.txt')}", "w") as file:
        for image_id, weight in res_matrix:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")

