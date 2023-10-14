import pymongo
import torch
import torchvision.datasets as datasets
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import phase_1.printing_images_dict as printing_images_dict
import numpy as np
path = os.getcwd()

def squared_l2_distance(input_vector, db_vector):
    return np.sqrt(np.sum(np.square(np.array(input_vector) - np.ravel(db_vector))))

def findKimages(representative_image, feature_space, k):
    distance_dict = {}
    for doc in collection.find():
        db_image_id = doc["image_id"]
        db_img_vector = doc[feature_space]
        distance_dict[db_image_id] = squared_l2_distance(representative_image, db_img_vector)
    
    sorted_distances = dict(sorted(distance_dict.items(), key=lambda x: x[1]))
    top_k_images = dict(list(sorted_distances.items())[:int(k)])
    images_display = {image_id: {'image': image, 'score': top_k_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_images}
    printing_images_dict.print_images(images_display, feature_space)

def task1():
    query_label = int(input("Enter a label L: "))
    print("\nSelect a feature space(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    feature_space = input("Enter input ")
    feature_space = str(feature_space) + "_feature_descriptor"
    k = input("Enter k: ")
    
    data_vector = collection.find_one({"label": query_label, })[feature_space]
    data_vector = np.array(data_vector).flatten()
    ncolumns = len(data_vector)
    
    data_matrix = np.empty((0, ncolumns))
    
    label_images = collection.find({"label": query_label})
    for document in label_images:
        data_vector = document.get(feature_space)
        data_vector = np.array(data_vector).flatten()
        data_matrix = np.vstack((data_matrix, data_vector))

    mean_vector = np.mean(data_matrix, axis=0)
    distances = np.linalg.norm(data_matrix - mean_vector, axis=1)
    representative_image_id = np.argmin(distances)
    representative_image = data_matrix[representative_image_id]
    
    findKimages(representative_image, feature_space, k)
    

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    task1()