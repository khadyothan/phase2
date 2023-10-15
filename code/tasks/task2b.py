import pymongo
import torchvision.datasets as datasets
import numpy as np
import torchvision
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.nn import functional as F
import torch.nn as nn
import sys
from sklearn.metrics import pairwise_distances_argmin_min
from torchvision import models
from torchvision.models import resnet50
from math import log2
from math import sqrt


cl = pymongo.MongoClient("mongodb://localhost:27017")
db = cl["caltech101db"]
collection = db["phase2trainingdataset"]
rep_collection = db["labelrepresentativeimages"]
caltech101_directory='/Users/lalitarvind/Downloads/MWD_Team_project_v1' 
data_transforms = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)])
basePath='/Users/lalitarvind/Downloads/MWD_Team_project_v1/cse515project/Code/'
dataset = datasets.Caltech101(caltech101_directory,transform = data_transforms ,download=True)
category_names = dataset.annotation_categories 

# calculate the kl divergence
def kl_divergence(p, q):
 return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence
def js_divergence(p, q):
 m = 0.5 * (p + q)
 return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def probabilistic_dist_plain(query, vectors):
    distances = []
    #print("calculating js divergence")
    for idx,vector in enumerate(vectors):
        temp = sqrt(js_divergence(vector,query))
        distances.append((idx,temp))
    distances = sorted(distances,key = lambda a:a[1])
    return distances


def extract_RESNET_features(dataset,collection):
    softmax = nn.Softmax(dim=1)
    resnet = resnet50(pretrained = True)
    for i,(tensor,label) in enumerate(dataset):
        if i%2==0:
            feature = softmax(resnet(tensor.reshape(-1,3,224,224)))
            collection.update_one({"image_id": i },{ "$set" : { "resnet_softmax_feature_descriptor" : feature.detach().tolist()[0]}})

def generate_resnet_softmax_representatives():

    for label in range(101):
        label_images = list(collection.find({'label': label}))
        resnet_softmax_vectors = []
        for image in label_images:
            resnet_softmax_vectors.append(np.ravel(image['resnet_softmax_feature_descriptor']))
        mean_resnet = np.mean(resnet_softmax_vectors, axis=0)
        closest_image_idx_resnet = probabilistic_dist_plain(mean_resnet,resnet_softmax_vectors)[0]
        #closest_image_idx_resnet = pairwise_distances_argmin_min([mean_resnet], resnet_softmax_vectors, metric='euclidean')[0][0]
        closest_image_vector_resnet = resnet_softmax_vectors[closest_image_idx_resnet[0]]
        closest_image_id_resnet = label_images[closest_image_idx_resnet[0]]['image_id']
        rep_collection.update_one({"label": label},{"$set":{"resnet_softmax_image_id": closest_image_id_resnet,"resnet_softmax_rep_image": closest_image_vector_resnet.tolist()}})

def task2b(query_features, k):
    feature_list = []
    feature_list=[i["resnet_softmax_feature_descriptor"] for i in collection.find({}, {"resnet_softmax_feature_descriptor": 1}) ]
    if len(feature_list) == 0:
        print("extracting resnet softmax features in mongodb")
        extract_RESNET_features()
        generate_resnet_softmax_representatives()
        feature_list=[i["resnet_softmax_feature_descriptor"] for i in collection.find({}, {"resnet_softmax_feature_descriptor": 1}) ]

    representatives = [list(rep_collection.find({'label': i}))[0] for i in range(101)]
    print("fetching representatives from DB")
    distances = []
    for rep in representatives:
        temp = 0
        temp = sqrt(js_divergence(np.array(rep["resnet_softmax_rep_image"]),np.array(query_features)))
        distances.append((rep["label"],category_names[rep["label"]],float(temp)**0.5))
    distances = sorted(distances,key = lambda a:a[2])
    return distances[:k]


if __name__ == "__main__":
    
    print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
    query_type = int(input("Enter query type: "))

    query_image_id = None
    query_image_file = None
    query_image_features = None

    softmax = nn.Softmax(dim=1)
    resnet = resnet50(pretrained = True)
    if query_type == 1:
        query_image_id = int(input("Enter query image ID: "))
        query_image_features = softmax(resnet(dataset[query_image_id][0].reshape(-1,3,224,224)))

    elif query_type == 2:
        query_image_file = input("Give the query image file path: ")
        image_dt = data_transforms(Image.open(query_image_file))
        query_image_features = softmax(resnet(image_dt.reshape(-1,3,224,224)))
    else: 
        print("Enter valid query type!")
    k = int(input("Enter k: "))
    query_image_features = query_image_features.detach().tolist()

    distances = task2b(query_image_features[0], k)
    print(f"Top {k} labels:\n")
    for item in distances:
        print(f"label id:{item[0]}, category: {item[1]}, distance: {item[2]}")