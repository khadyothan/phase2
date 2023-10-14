import numpy as np
import pymongo
import torch
import torchvision.datasets as datasets
import sys
import json
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import dimensionality_reduction_techniques.svd as svd
import dimensionality_reduction_techniques.nnmf as nnmf
import dimensionality_reduction_techniques.lda as lda
import dimensionality_reduction_techniques.kmeans as kmeans
import dimensionality_reduction_techniques.cp as cp

def task5_execution(query_feature_model, k, dimredtech):
    if dimredtech == 1:
        data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../label_label_sm_matrices/label_label_sm_{query_feature_model}.csv')}", delimiter=',')
        U_reduced, S_reduced, VT_reduced = svd.svd(data_matrix, k)
        latent_semantics = np.dot(U_reduced, S_reduced)
        ls_dict = {
            "latent_semantics": latent_semantics.tolist(),
            "VT_reduced": VT_reduced.tolist()
        }
        file_path_ls = f"{os.path.join(os.getcwd(), f'../outputs/latent_semantics/{dimredtech}_{query_feature_model}_{k}_llsm.json')}"
        with open(file_path_ls, 'w') as json_file:
            json.dump(ls_dict, json_file)
        svd.calculateImageIDWeightPairs(latent_semantics, query_feature_model, k, "llsm")
        return latent_semantics
        
    elif dimredtech == 2:
        data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../label_label_sm_matrices/label_label_sm_{query_feature_model}.csv')}", delimiter=',')
        W, H = nnmf.nmf(data_matrix, k)
        latent_semantics = W[:, :k]
        ls_dict = {
            "latent_semantics": latent_semantics.tolist(),
        }
        file_path_ls = f"{os.path.join(os.getcwd(), f'../outputs/latent_semantics/{dimredtech}_{query_feature_model}_{k}_llsm.json')}"
        with open(file_path_ls, 'w') as json_file:
            json.dump(ls_dict, json_file)
        nnmf.calculateImageIDWeightPairs(latent_semantics, query_feature_model, k, "llsm")
        return latent_semantics
        
    elif dimredtech == 3:
        data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../label_label_sm_matrices/label_label_sm_{query_feature_model}.csv')}", delimiter=',')
        latent_semantics, query_distribution = lda.lda(data_matrix, k)
        print(query_distribution)
        ls_dict = {
            "latent_semantics": latent_semantics.tolist(),
        }
        file_path_ls = f"{os.path.join(os.getcwd(), f'../outputs/latent_semantics/{dimredtech}_{query_feature_model}_{k}_llsm.json')}"
        with open(file_path_ls, 'w') as json_file:
            json.dump(ls_dict, json_file)
        lda.calculateImageIDWeightPairs(query_distribution, query_feature_model, k, "llsm")
        return query_distribution
        
    elif dimredtech == 4:
        data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../label_label_sm_matrices/label_label_sm_{query_feature_model}.csv')}", delimiter=',')
        latent_semantics = kmeans.kmeans(data_matrix, k)
        ls_dict = {
            "latent_semantics": latent_semantics.tolist(),
        }
        file_path_ls = f"{os.path.join(os.getcwd(), f'../outputs/latent_semantics/{dimredtech}_{query_feature_model}_{k}_llsm.json')}"
        with open(file_path_ls, 'w') as json_file:
            json.dump(ls_dict, json_file)
        kmeans.calculateImageIDWeightPairs(latent_semantics, query_feature_model, k, "llsm")
        return latent_semantics
    
    else:
        print("Enter a valid dim red tech number!!!")
        
    return True

def task5(): 
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]

    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    print("\nSelect a feature model(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_model = input("Enter input: ")
    query_feature_model = str(query_feature_model)
    
    k = int(input("Enter k value: "))
    
    print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
    print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
    dimredtech = int(input("Enter your choice: "))
    
    return task5_execution(query_feature_model, k, dimredtech)

if __name__ == "__main__":
    task5()