import os
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
from torchvision import models
from torchvision.models import resnet50
path = os.getcwd()

def resnet(image):
    
    data_transforms = T.Compose([T.Resize((224,224)), T.ToTensor(),T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0]==1 else x)])
    image = data_transforms(image)
    resize = T.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        
    myTrainedmodel = resnet50(weights=models.ResNet50_Weights.DEFAULT)
    myTrainedmodel.fc = nn.Linear(2048, 101)
    myTrainedmodel.load_state_dict(torch.load(path +'/../trained_models/resnetMod101.pkl' ,map_location=torch.device('cpu')))
    preds_tensor = myTrainedmodel(image.reshape(-1,3,224,224))
    pred_probs = F.softmax(preds_tensor, dim=1).to("cpu").data.numpy()
    probabilties=[ (idx,ele) for idx,ele in enumerate(pred_probs[0])]
    probabilties=sorted(probabilties,key= lambda a: a[1],reverse=True)
    
    return (pred_probs[0])

# caltech101_directory = os.path.join(path, "../../data")
# print(caltech101_directory)
# dataset = datasets.Caltech101(caltech101_directory, download=False)

# for image_id, (image, label) in enumerate(dataset):
#     result = resnet(image)
#     print(result)
#     break