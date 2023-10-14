import os
import numpy as np
import pymongo
import tensorly as tl

def cp(data_matrix, k):
    print(data_matrix.shape)
    X = data_matrix
    data_tensor = X.reshape(X.shape[0],X.shape[1],1)
    weights,factors = tl.decomposition.parafac(data_tensor, rank=k, init='random')
    return weights, factors

def calculateImageIDWeightPairs(cp_ls, feature_descriptor, k, type):
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    ls_matrix = cp_ls
    image_labels = []
    for doc in collection.find():
        label = doc["label"]
        image_labels.append(label)  
    lS_pairs = []
    sum = 0
    count = 0
    for i in range(ls_matrix.shape[1]):
        list_pair = []
        curr_label = 0
        for j in range(ls_matrix.shape[0]):
            if(image_labels[j] == curr_label):
                sum = sum + ls_matrix[j][i]
                count = count + 1
            else:
                list_pair.append([curr_label,sum/count])
                curr_label = curr_label + 1
                sum = 0
                count = 0
        list_pair.append([label,sum])
        sorted_data = sorted(list_pair, key=lambda x: x[1])
        lS_pairs.append(sorted_data)
        label_weights_path = f"{os.path.join(os.getcwd(), f'../outputs/weight_pairs/cp_{feature_descriptor}_{k}_{type}.txt')}"
        with open(label_weights_path, "w") as f:
            for i in range(len(lS_pairs)):
                f.write(f"Latent Semantic {i + 1}:\n")
                for j in range(len(lS_pairs[i])):
                    f.write(f"      Label {lS_pairs[i][j][0]}: {lS_pairs[i][j][1]}\n")