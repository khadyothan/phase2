import numpy as np
import networkx as nx
import pymongo
from sklearn.metrics.pairwise import cosine_similarity
import sys
import torch
import torchvision.datasets as datasets
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
import phase_1.printing_images_dict as printing_images_dict

def task11_2(G, m, l, collection):
    node_label = {}
    for document in collection.find():
        image_id = document["image_id"]
        label = document["label"]
        node_label[image_id/2] = label
    personalization = {}
    for node in G.nodes():
    # Set the personalized PageRank value to 1 for nodes with the desired label, 0 otherwise
        personalization[node] = 1 if node_label[node] == l else 0
    pagerank_scores = nx.pagerank(G, alpha=0.85, personalization=personalization)
    sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    top_m_image_ids = [node*2 for node, score in sorted_nodes[:m]]
    return top_m_image_ids

def createGraph(latent_semantics, n):
    similarity_matrix = cosine_similarity(latent_semantics)
    most_similar_images = np.argsort(-similarity_matrix, axis=1)[:, :n]
    G = nx.Graph()
    for i in range(len(latent_semantics)):
        G.add_node(i)
    for i, similar_indices in enumerate(most_similar_images):
        print(i)
        for j in similar_indices:
            if i!=j:
                G.add_edge(i, j)
    return G

def task11_execution(query_latent_semantics, n, m, l, collection, dataset):
    print("\nSelect a feature model(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n\n")
    query_feature_model = input("Enter input: ")
    query_feature_model = str(query_feature_model)
    k = int(input("Enter k value: "))
    if query_latent_semantics != 2:
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        dimredtech = int(input("Enter your choice: "))
        
    if query_latent_semantics == 1:
        latent_semantics = task3.task3_execution(query_feature_model, k, dimredtech, [])
    elif query_latent_semantics == 2:
        latent_semantics = task4.task4_execution(query_feature_model, k, [])
        
    G = createGraph(latent_semantics, n)
    top_m_image_ids = task11_2(G, m, l, collection)
    images_to_display = {image_id: {'image': image, 'score': 1} for image_id, (image, label) in enumerate(dataset) if image_id in top_m_image_ids}
    printing_images_dict.print_images(images_to_display, "Heading", target_size=(224, 224))
    
    return True

def task11():
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")
    query_latent_semantics = int(input("Enter your choice number: "))
    n = int(input("Enter n value: "))
    m = int(input("Enter m value: "))
    l = int(input("Enter label l value: ")) 
    task11_execution(query_latent_semantics, n, m, l, collection, dataset)
    
    return True

if __name__ == "__main__":
    task11()