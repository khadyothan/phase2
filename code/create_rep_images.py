import pymongo
from sklearn.metrics import pairwise_distances_argmin_min
import torchvision.datasets as datasets
import numpy as np
import torch

def store_representative_images():

    for label in range(101):
        label_images = list(collection.find({'label': label}))  # Convert the cursor to a list
        color_moments_vectors = []
        hog_vectors = []
        layer3_vectors = []
        avgpool_vectors = []
        fc_vectors = []
        
        for image in label_images:
            color_moments_vectors.append(np.ravel(image['color_moments_feature_descriptor']))
            hog_vectors.append(np.ravel(image['hog_feature_descriptor']))
            layer3_vectors.append(np.ravel(image['resnet50_layer3_feature_descriptor']))
            avgpool_vectors.append(np.ravel(image['resnet50_avgpool_feature_descriptor']))
            fc_vectors.append(np.ravel(image['resnet50_fc_feature_descriptor']))

        mean_color_moments = np.mean(color_moments_vectors, axis=0)
        mean_hog = np.mean(hog_vectors, axis=0)
        mean_layer3 = np.mean(layer3_vectors, axis=0)
        mean_avgpool = np.mean(avgpool_vectors, axis=0)
        mean_fc = np.mean(fc_vectors, axis=0)
        
        # Find the image closest to the mean vector using Euclidean distance
        closest_image_idx_color_moments = pairwise_distances_argmin_min([mean_color_moments], color_moments_vectors, metric='euclidean')[0][0]
        closest_image_idx_hog = pairwise_distances_argmin_min([mean_hog], hog_vectors, metric='euclidean')[0][0]
        closest_image_idx_layer3 = pairwise_distances_argmin_min([mean_layer3], layer3_vectors, metric='euclidean')[0][0]
        closest_image_idx_avgpool = pairwise_distances_argmin_min([mean_avgpool], avgpool_vectors, metric='euclidean')[0][0]
        closest_image_idx_fc = pairwise_distances_argmin_min([mean_fc], fc_vectors, metric='euclidean')[0][0]
        
        # Get the Image vectors of the closest images
        closest_image_vector_color_moments = color_moments_vectors[closest_image_idx_color_moments]
        closest_image_vector_hog = hog_vectors[closest_image_idx_hog]
        closest_image_vector_layer3 = layer3_vectors[closest_image_idx_layer3]
        closest_image_vector_avgpool = avgpool_vectors[closest_image_idx_avgpool]
        closest_image_vector_fc = fc_vectors[closest_image_idx_fc]
        
        closest_image_id_color_moments = label_images[closest_image_idx_color_moments]['image_id']
        closest_image_id_hog = label_images[closest_image_idx_hog]['image_id']
        closest_image_id_layer3 = label_images[closest_image_idx_layer3]['image_id']
        closest_image_id_avgpool = label_images[closest_image_idx_avgpool]['image_id']
        closest_image_id_fc = label_images[closest_image_idx_fc]['image_id']
        
        # Store the results in the new collection
        new_collection.insert_one({
            'label': label,
            'color_moments_rep_image': closest_image_vector_color_moments.tolist(),
            'hog_rep_image': closest_image_vector_hog.tolist(),
            'resnet50_layer3_rep_image': closest_image_vector_layer3.tolist(),
            'resnet50_avgpool_rep_image': closest_image_vector_avgpool.tolist(),
            'resnet50_fc_rep_image': closest_image_vector_fc.tolist(),
            'color_moments_rep_image_id': closest_image_id_color_moments,
            'hog_rep_image_id': closest_image_id_hog,
            'resnet50_layer3_rep_image_id': closest_image_id_layer3,
            'resnet50_avgpool_rep_image_id': closest_image_id_avgpool,
            'resnet50_fc_rep_image_id': closest_image_id_fc,
        })
        
        print(label)
      
if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"
    new_collection = db["labelrepresentativeimages"]
    caltech101_directory = "C:/Khadyu/ASU/Fall 2023/Multimedia & Web Databases/Project/Phase1/data"
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    store_representative_images()
