import os
import numpy as np
from sklearn.decomposition import NMF

def calculateImageIDWeightPairs(latent_semantics, feature_descriptor, k, type): 
    image_id_weight_pairs = []
    ls_matrix = latent_semantics
    for i in range(len(latent_semantics)):
        image_id = i * 2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True)
    with open(f"{os.path.join(os.getcwd(), f'../outputs/weight_pairs/{2}_{feature_descriptor}_{k}_{type}.txt')}", "w") as file:
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")

def nmf(data_matrix, k):
    print(data_matrix.shape)
    data_matrix[data_matrix < 0] = 0  #Truncate negative values to zero
    model = NMF(n_components=k, init='nndsvda')
    W = model.fit_transform(data_matrix)
    H = model.components_
    latent_semantics = W[:, :k]
    print(latent_semantics.shape)
    return W, H