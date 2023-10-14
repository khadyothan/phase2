import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.nn import functional as F
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
from torchvision import models
from torchvision.models import resnet50

def task2b(query_image_id, query_image_file, k,category_names):
    data_transforms = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)])
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                print(f'label:{label}')
                break
    elif query_image_file != None:
        query_image_data = data_transforms(Image.open(query_image_file))
    resize = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
   
    myTrainedmodel = resnet50(weights=models.ResNet50_Weights.DEFAULT)
    myTrainedmodel.fc = nn.Linear(2048, 101)
    myTrainedmodel.load_state_dict(torch.load(os.path.join(path, "../trained_models/resnetMod101.pkl") ,map_location=torch.device('cpu')))
    preds_tensor = myTrainedmodel(query_image_data.reshape(-1,3,224,224))
    pred_probs = F.softmax(preds_tensor, dim=1).to("cpu").data.numpy()
    probabilties=[ (idx,ele) for idx,ele in enumerate(pred_probs[0])]
    probabilties=sorted(probabilties,key= lambda a: a[1],reverse=True)
    for i in range(k):
        print(f'k:{str(i+1).ljust(2," ")} category name: {category_names[int(probabilties[i][0])].ljust(25," ")} label no : {str(int(probabilties[i][0])).ljust(3," ")} probability: {probabilties[i][1]}')


if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection = db["phase2trainingdataset"]
    collection_name = "phase2trainingdataset"
    caltech101_directory='C:\Khadyu\ASU\Fall 2023\Multimedia & Web Databases\Project\Phase2\cse515-project\data' 
    resize = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
    data_transforms = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)])
    dataset = datasets.Caltech101(caltech101_directory,transform = data_transforms ,download=False)
    category_names = dataset.annotation_categories 
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

    k = int(input("Enter k: "))
    task2b(query_image_id, query_image_file, k,category_names)