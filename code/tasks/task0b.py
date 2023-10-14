import torch
import numpy as np
import pymongo
import torchvision.datasets as datasets
from PIL import Image
import sys
from sklearn.metrics.pairwise import cosine_similarity
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import extracting_feature_space.color_moments as color_moments
import extracting_feature_space.hog as HOG
import extracting_feature_space.resnet_features as resnet_features
import phase_1.printing_images_dict as printing_images_dict
path = os.getcwd()

def l2_distance(input_vector, db_vector):
    return np.linalg.norm(np.array(input_vector) - np.array(db_vector))

def squared_l2_distance(input_vector, db_vector):
    return np.sqrt(np.sum(np.square(np.array(input_vector) - np.array(db_vector))))

def similarkimages(query_image_data, query_feature_space, k):
    distance_cm_dict, distance_hog_dict, distance_avgpool_dict, distance_layer3_dict, distance_fc_dict = {}, {}, {}, {}, {}
    
    query_cm_vector = color_moments.color_moments(query_image_data)
    query_hog_vector = HOG.HOG(query_image_data)
    query_layer3_vector, query_avgpool_vector, query_fc_vector = resnet_features.resnet_features(query_image_data)
    query_layer3_vector =  query_layer3_vector
    query_avgpool_vector =  query_avgpool_vector
    query_fc_vector =  query_fc_vector
    
    for doc in collection.find():
        db_image_id = doc["image_id"]
        db_cm_vector = doc["color_moments_feature_descriptor"]
        db_hog_vector = doc["hog_feature_descriptor"]
        db_avgpool_vector = doc["resnet50_avgpool_feature_descriptor"]
        db_layer3_vector = doc["resnet50_layer3_feature_descriptor"]
        db_fc_vector = doc["resnet50_fc_feature_descriptor"]
        
        distance_layer3_dict[db_image_id] = squared_l2_distance([query_layer3_vector], [db_layer3_vector])
        distance_avgpool_dict[db_image_id] = squared_l2_distance([query_avgpool_vector], [db_avgpool_vector])
        distance_fc_dict[db_image_id] = squared_l2_distance([query_fc_vector], [db_fc_vector])
        distance_cm_dict[db_image_id] = squared_l2_distance(query_cm_vector, db_cm_vector)
        distance_hog_dict[db_image_id] = squared_l2_distance(query_hog_vector, db_hog_vector)
          
    # Sort the distances and select the top K images for each feature descriptor
    sorted_cm_distances = dict(sorted(distance_cm_dict.items(), key=lambda x: x[1]))
    top_k_cm_images = dict(list(sorted_cm_distances.items())[:k])
    
    sorted_hog_distances = dict(sorted(distance_hog_dict.items(), key=lambda x: x[1]))
    top_k_hog_images = dict(list(sorted_hog_distances.items())[:k])
    
    sorted_avgpool_distances = dict(sorted(distance_avgpool_dict.items(), key=lambda x: x[1]))
    top_k_avgpool_images = dict(list(sorted_avgpool_distances.items())[:k])
    
    sorted_layer3_distances = dict(sorted(distance_layer3_dict.items(), key=lambda x: x[1]))
    top_k_layer3_images = dict(list(sorted_layer3_distances.items())[:k])
    
    sorted_fc_distances = dict(sorted(distance_fc_dict.items(), key=lambda x: x[1]))
    top_k_fc_images = dict(list(sorted_fc_distances.items())[:k])
    
    # Create dictionaries to store top K images with their distances
    images_cm_display = {image_id: {'image': image, 'score': top_k_cm_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_cm_images}
    images_hog_display = {image_id: {'image': image, 'score': top_k_hog_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_hog_images}
    images_layer3_display = {image_id: {'image': image, 'score': top_k_layer3_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_layer3_images}
    images_avgpool_display = {image_id: {'image': image, 'score': top_k_avgpool_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_avgpool_images}
    images_fc_display = {image_id: {'image': image, 'score': top_k_fc_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_fc_images}
    
    if query_feature_space == "color_moments":
        printing_images_dict.print_images(images_cm_display, query_feature_space, target_size=(224, 224))
    elif query_feature_space == "hog":
        printing_images_dict.print_images(images_hog_display, query_feature_space, target_size=(224, 224))
    elif query_feature_space == "resnet50_layer3":
        printing_images_dict.print_images(images_layer3_display, query_feature_space, target_size=(224, 224))
    elif query_feature_space == "resnet50_avgpool":
        printing_images_dict.print_images(images_avgpool_display, query_feature_space, target_size=(224, 224))
    elif query_feature_space == "resnet50_fc":
        printing_images_dict.print_images(images_fc_display, query_feature_space, target_size=(224, 224))

def task0b(query_image_id, query_image_file, query_feature_space, k):
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                similarkimages(query_image_data, query_feature_space, k)
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
        similarkimages(query_image_data, query_feature_space, k)

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)

    print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
    query_type = int(input("Enter query type: "))
    query_image_id = None
    query_image_file = None
    if query_type == 1:
        query_image_id = int(input("Enter query image ID: "))
    elif query_type == 2:
        query_image_file = input("Give the query image file path: ")
    else: 
        print("Enter valid query type!")
        
    print("\nSelect input feature space(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_space = input("Enter your query feature space: ")
    k = int(input("Enter k value: "))
    
    task0b(query_image_id, query_image_file, query_feature_space, k)
    