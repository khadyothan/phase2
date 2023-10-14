import csv
import json
import os
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

def lda(data_matrix, k):
    print(data_matrix.shape)
    data_matrix = np.array(data_matrix)
    data_matrix_min_val = data_matrix.min()
    if data_matrix_min_val<0:
        data_matrix = data_matrix - data_matrix_min_val
        
    lda = LatentDirichletAllocation(n_components=k,verbose=True)
    model = lda.fit(data_matrix)
    LDA_factor_matrix = lda.components_ /lda.components_.sum(axis=1)[:, np.newaxis]
    
    query_distribution = np.matmul(data_matrix,np.transpose(LDA_factor_matrix))
    return LDA_factor_matrix, query_distribution


def calculateImageIDWeightPairs(latent_semantics, feature_descriptor, k, type): 
    image_id_weight_pairs = []
    ls_matrix = latent_semantics
    for i in range(len(ls_matrix)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True)
    with open(f"{os.path.join(os.getcwd(), f'../outputs/weight_pairs/{3}_{feature_descriptor}_{k}_{type}.txt')}", 'w') as file:  
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")
    return True