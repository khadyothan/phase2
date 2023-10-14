import pymongo 
import numpy as np
import torchvision.datasets as datasets
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()

def createMatrix(feature_descriptor):
    data_vector = collection.find_one({"image_id": 0, })[feature_descriptor]
    data_vector = np.array(data_vector).flatten()
    ncolumns = len(data_vector)
    data_matrix = np.empty((0, ncolumns))
    for i in range(8678):
        if i%2 == 0:
            data_vector = collection.find_one({"image_id": i, })[feature_descriptor]
            data_vector = np.array(data_vector).flatten()
            data_matrix = np.vstack((data_matrix, data_vector))
    print(feature_descriptor)
    return data_matrix

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    caltech101_directory = os.path.join(path, "../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    
    feature_models = ["color_moments", "hog", "resnet50_layer3", "resnet50_avgpool", "resnet50_fc",'resnet_softmax']
    for i in range(len(feature_models)):
        data_matrix = createMatrix(feature_models[i] + "_feature_descriptor")
        file_path = f"{os.path.join(os.getcwd(), f'data_matrices/data_matrix_{feature_models[i]}.csv')}"
        np.savetxt(file_path, data_matrix, delimiter=",")
