import os
import numpy as np

def calculateImageIDWeightPairs(latent_semantics, feature_descriptor, k, type): 
    image_id_weight_pairs = []
    ls_matrix = latent_semantics
    for i in range(len(latent_semantics)):
        image_id = i*2
        weight = np.linalg.norm(ls_matrix[i, :])
        image_id_weight_pairs.append((image_id, weight))
    sorted_image_id_weight_pairs_cm = sorted(image_id_weight_pairs, key=lambda x: x[1], reverse=True)
    with open(f"{os.path.join(os.getcwd(), f'../outputs/weight_pairs/{1}_{feature_descriptor}_{k}_{type}.txt')}", "w") as file:
        for image_id, weight in sorted_image_id_weight_pairs_cm:
            file.write(f"Image ID: {image_id}, Weight: {weight}\n")

def svd(data_matrix, k):
    print(data_matrix.shape)
    cov_matrix = np.dot(data_matrix.T, data_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvectors_k = eigenvectors[:, :k]
    singular_values = np.sqrt(eigenvalues[:k])
    U_reduced = np.real(np.dot(data_matrix, eigenvectors_k))
    VT_reduced = np.real(eigenvectors_k.T)
    S_reduced = np.real(np.diag(singular_values))
    latent_semantics = np.dot(U_reduced, S_reduced)
    print(latent_semantics.shape)
    return U_reduced, S_reduced, VT_reduced