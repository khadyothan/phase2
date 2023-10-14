import torch
import numpy as np
import pymongo
import torchvision.datasets as datasets
from PIL import Image
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import extracting_feature_space.color_moments as color_moments
import extracting_feature_space.hog as HOG
import extracting_feature_space.resnet_features as resnet_features
path = os.getcwd()

def task0a():
    for image_id, (image, label) in enumerate(dataset):  
        if image_id % 2 == 0:
        # Check if the image is grayscale and convert it to RGB if necessary
            if(len(np.array(image).shape) != 3):
                converted_img  = np.stack((np.array(image),) * 3, axis=-1)
                image = Image.fromarray(converted_img)
                
            # Extract color moments, HOG, and ResNet features for the image
            color_moments_feature_descriptor = color_moments.color_moments(image)
            hog_feature_descriptor = HOG.HOG(image)
            layer3_feature_descriptor, avgpool_feature_descriptor, fc_feature_descriptor = resnet_features.resnet_features(image)    
            
            # Insert the feature descriptors and related information into the MongoDB collection
            collection.insert_one({
                "image_id": image_id,
                "label": label,
                "color_moments_feature_descriptor": color_moments_feature_descriptor, 
                "hog_feature_descriptor": hog_feature_descriptor,
                "resnet50_layer3_feature_descriptor": layer3_feature_descriptor,
                "resnet50_avgpool_feature_descriptor": avgpool_feature_descriptor,
                "resnet50_fc_feature_descriptor": fc_feature_descriptor,
            })
            print(image_id)
            
if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"

    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    if collection_name not in db.list_collection_names():
        print("\nFeatures are being extracted and stored in the database. Wait patiently.\n")
        task0a()
        print("\nAll the feature descriptors of all the images are stored in MongoDB database in collection called caltech101collection !!\n")
    else:
        print("The database already has all the feature descriptors of all the images stored!!")