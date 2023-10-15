import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import extracting_feature_space.color_moments as color_moments
import extracting_feature_space.hog as hog
import extracting_feature_space.resnet_features as resnet_features
import dimensionality_reduction_techniques.svd as svd
import dimensionality_reduction_techniques.nnmf as nnmf
import dimensionality_reduction_techniques.lda as lda
import dimensionality_reduction_techniques.kmeans as kmeans
import dimensionality_reduction_techniques.cp as cp
import phase_1.printing_images_dict as printing_images_dict
import tasks.task3 as task3
import tasks.task4 as task4
import tasks.task5 as task5
import tasks.task6 as task6
from math import log2
from math import sqrt

caltech101_directory = os.path.join(path, "../../data")
dataset = datasets.Caltech101(caltech101_directory,download=False)
category_names = dataset.annotation_categories 
cl = pymongo.MongoClient("mongodb://localhost:27017")
db = cl["caltech101db"]
collection1 = db["phase2trainingdataset"]
collection2 = db["labelrepresentativeimages"]

# calculate the kl divergence
def kl_divergence(p, q):
 return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence
def js_divergence(p, q):
 m = 0.5 * (p + q)
 return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def probabilistic_dist(query, representatives):
    distances = []
    print("calculating js divergence")
    for label,rep in enumerate(representatives):
        temp = sqrt(js_divergence(rep,query))
        distances.append((label,category_names[label],temp))
    distances = sorted(distances,key = lambda a:a[2])
    return distances

def euclidian_dist(query, representatives):
    distances = []
    for j,rep in enumerate(representatives):
        temp = 0
        for i in range(len(rep)):
            temp = (rep[i]-float(query[i]))**2
        distances.append((j,category_names[j],temp**0.5))
    distances = sorted(distances,key = lambda a:a[2])
    return distances

def euclidian_dist_plain(query, datamatrix):
    distances = []
    for img_id,item in enumerate(datamatrix):
        dist = 0
        for i,val in enumerate(item):
            dist+=(val - query[i])**2
        distances.append((img_id,dist))
    distances = sorted(distances,key = lambda a:a[1])
    return distances

def task8_execution(query_image_data, query_latent_semantics, K):
    
    print("\nSelect a feature model(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n6. resnet_softmax\n")
    query_feature_model = str(input("Enter input: "))
    #query_feature_model = "hog"
    k = int(input("Enter k value: "))
    #k = 15
    if query_latent_semantics != 2:
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        dimredtech = int(input("Enter dimensionality reduction technique: "))
        #dimredtech = 1
    
    if query_feature_model == "color_moments":
            query_image_vector = color_moments.color_moments(query_image_data)
    elif query_feature_model == "hog":
        query_image_vector = hog.HOG(query_image_data)
    else :
        query_layer3_vector, query_avgpool_vector, query_fc_vector = resnet_features.resnet_features(query_image_data)
        if query_feature_model == "resnet50_layer3":
            query_image_vector = query_layer3_vector
        elif query_feature_model == "resnet50_avgpool":
            query_image_vector = query_avgpool_vector
        elif query_feature_model == "resnet50_fc":
            query_image_vector = query_fc_vector 
        else:
            query_image_vector = resnet_features.resnetSoftmax(query_fc_vector)
    query_image_vector = np.ravel(query_image_vector)
    
    rep_keyname = {
    "color_moments":["color_moments_rep_image","color_moments_image_id"],
    "hog":["hog_rep_image","hog_image_id"],
    "resnet50_layer3":["layer3_rep_image","layer3_image_id"],
    "resnet50_avgpool":["avgpool_rep_image","avgpool_image_id"],
    "resnet50_fc":["fc_rep_image","fc_image_id"],
    "resnet_softmax":["resnet_softmax_rep_image","resnet_softmax_image_id"]
    }
    
    representatives = list(collection2.find({},{rep_keyname[query_feature_model][0]:1,
                                                rep_keyname[query_feature_model][1]:1,}))
    if query_latent_semantics == 1:
        latent_semantics = task3.task3_execution(query_feature_model, k, dimredtech, query_image_vector)
        query_image_vector_ls = latent_semantics[0]
        database_vectors_ls = latent_semantics[1:]
        representatives_ls = [database_vectors_ls[int(label_dict[rep_keyname[query_feature_model][1]]/2)] for label_dict in representatives]
        if dimredtech!=3:
            distances = euclidian_dist(query_image_vector_ls,representatives_ls)
        else:
            distances = probabilistic_dist(query_image_vector_ls,representatives_ls)

        print(f"Top {K} labels:\n")
        for item in distances[:K]:
            print(f"label id:{item[0]}, category: {item[1]}, distance: {item[2]}")
        
    elif query_latent_semantics == 2:
        latent_semantics = task4.task4_execution(query_feature_model, k, query_image_vector)
        query_image_vector_ls = latent_semantics[0]
        database_vectors_ls = latent_semantics[1:]
        representatives_ls = [database_vectors_ls[int(label_dict[rep_keyname[query_feature_model][1]]/2)] for label_dict in representatives]
        distances = euclidian_dist(query_image_vector_ls,representatives_ls)
        print(f"Top {K} labels:\n")
        for item in distances[:K]:
            print(f"label id:{item[0]}, category: {item[1]}, distance: {item[2]}")
    
    elif query_latent_semantics == 3:
        latent_semantics = task5.task5_execution(query_feature_model,k,dimredtech)
        representatives_features = [label_dict[rep_keyname[query_feature_model][0]] for label_dict in representatives]
        distances_to_reps = euclidian_dist(query_image_vector,representatives_features)
        query_image_rep_label_no = distances_to_reps[0][0]
        query_rep_ls=latent_semantics[query_image_rep_label_no]
        if dimredtech!=3:
            distances = euclidian_dist(query_rep_ls,latent_semantics)
        else:
            distances = probabilistic_dist(query_rep_ls,latent_semantics)
        print(f"Top {K} labels:\n")
        for item in distances[:K]:
            print(f"label id:{item[0]}, category: {item[1]}, distance: {item[2]}")
        
    elif query_latent_semantics == 4:
        latent_semantics = task6.task6_execution(query_feature_model,k,dimredtech)
        data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../data_matrices/data_matrix_{query_feature_model}.csv')}", delimiter=',')
        distances = euclidian_dist_plain(query_image_vector,data_matrix)
        query_image_vector_ls = latent_semantics[distances[0][0]]
        representatives_ls = [latent_semantics[int(label_dict[rep_keyname[query_feature_model][1]]/2)] for label_dict in representatives]
        if dimredtech!=3:
            distances = euclidian_dist(query_image_vector_ls,representatives_ls)
        else:
            distances = probabilistic_dist(query_image_vector_ls,representatives_ls)
        print(f"Top {K} labels:\n")
        for item in distances[:K]:
            print(f"label id:{item[0]}, category: {item[1]}, distance: {item[2]}")
        

    return True

def task8():
    
    print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
    query_type = int(input("Enter query type: "))
    #query_type = 1
    query_image_id = None
    query_image_file = None
    if query_type == 1:
        query_image_id = int(input("Enter query image ID: "))
        #query_image_id = 2500
    elif query_type == 2:
        query_image_file = input("Give the query image file path: ")
    else: 
        print("Enter valid query type!")

    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
        
    print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")
    
    query_latent_semantics = int(input("Enter your choice number: "))
    #query_latent_semantics = 4
    K = int(input("Enter K value for finding K similar labels: "))
    #K = 10
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)    
    task8_execution(query_image_data, query_latent_semantics, K)


task8()