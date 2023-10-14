import numpy as np
import pymongo
import torch
import torchvision.datasets as datasets
import sys
import json
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import dimensionality_reduction_techniques.cp as cp

def task4_execution(query_feature_model, k, query_image_vector):
    data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../data_matrices/data_matrix_{query_feature_model}.csv')}", delimiter=',')
    if len(query_image_vector) > 1:
        data_matrix = np.vstack((query_image_vector, data_matrix))
    weights, latent_semantics = cp.cp(data_matrix, k)
    ls_dict = {
        "image_image": latent_semantics[0].tolist(),
        "feature_feature": latent_semantics[1].tolist(),
        "label_label": latent_semantics[2].tolist(),
    }
    file_path_ls = f"{os.path.join(os.getcwd(), f'../outputs/latent_semantics/cp_{query_feature_model}_{k}.json')}"
    with open(file_path_ls, 'w') as json_file:
        json.dump(ls_dict, json_file)
    if len(query_image_vector) == 0:
        cp.calculateImageIDWeightPairs(latent_semantics[0], query_feature_model, k, "")
    return latent_semantics[0]

def task4(): 
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

    return task4_execution(query_feature_model, k, [])
    
if __name__ == "__main__":
    task4()