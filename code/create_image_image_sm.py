import pymongo 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.datasets as datasets
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()

def createMatrix(feature_descriptor):
    image_vectors = [doc[feature_descriptor] for doc in collection.find()]
    num_images = len(image_vectors)
    image_similarity_matrix = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(i, num_images):
            if i == j:
                image_similarity_matrix[i, j] = 1.0
            else:
                feature_i = np.array(image_vectors[i]).reshape(1, -1)
                feature_j = np.array(image_vectors[j]).reshape(1, -1)
                similarity_matrix  = cosine_similarity(feature_i, feature_j)
                similarity = similarity_matrix[0, 0]
                image_similarity_matrix[i, j] = similarity
                image_similarity_matrix[j, i] = similarity 
            print(i, j)
    return image_similarity_matrix
    
if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    caltech101_directory = os.path.join(path, "../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    
    feature_models = ["color_moments", "hog", "resnet50_layer3", "resnet50_avgpool", "resnet50_fc","resnet_softmax"]
    for i in range(len(feature_models)):
        data_matrix = createMatrix(feature_models[i] + "_feature_descriptor")
        file_path = f"{os.path.join(os.getcwd(), f'../image_image_sm_matrices/image_image_sm_{feature_models[i]}.csv')}"
        np.savetxt(file_path, data_matrix, delimiter=",")