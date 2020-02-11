from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import time
from numpy import genfromtxt

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
              'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
              }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



file_table = pd.read_csv('file_table.csv')
val_table = pd.read_csv('tiny-imagenet-200/val/val_annotations.txt', sep='\t',header = None)
val_table.drop(labels = [2,3,4,5],axis=1,inplace = True)
class_dict = dict(zip(file_table.loc[:,'class'],file_table.loc[:,'classid']))


#root_dir should contain until train folder
class ImageNetDataset(Dataset): #loads data from the csv file

    def __init__(self, csv_file, root_dir, transform=None):
        self.imagenet_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imagenet_frame)

    def __getitem__(self, idx):
        imgid = torch.tensor([int(self.imagenet_frame.iloc[idx, 0])])     
        query_name = os.path.join(self.root_dir,self.imagenet_frame.iloc[idx, 2])
        inclass_name = os.path.join(self.root_dir,self.imagenet_frame.iloc[idx, 3])
        outclass_name = os.path.join(self.root_dir,self.imagenet_frame.iloc[idx, 4])
        
        query_image = Image.open(query_name)
        query_image = query_image.convert('RGB')
        
        inclass_image = Image.open(inclass_name)
        inclass_image = inclass_image.convert('RGB')
        
        outclass_image = Image.open(outclass_name)
        outclass_image = outclass_image.convert('RGB')
               
        if self.transform:
            query_image = self.transform(query_image)
            inclass_image = self.transform(inclass_image)
            outclass_image = self.transform(outclass_image)
            
        sample = [imgid,query_name,query_image,inclass_image,outclass_image]
        return sample



class ValImages(Dataset): #loads data from the csv file

    def __init__(self, csv_file, root_dir, classdict, transform=None):
        self.val_frame = pd.read_csv(csv_file, sep='\t', header = None)
        self.val_frame.drop(labels = [2,3,4,5],axis=1,inplace = True)
        self.root_dir = root_dir
        self.transform = transform
        self.classdict = classdict

    def __len__(self):
        return len(self.val_frame)

    def __getitem__(self, idx):
        imgid = torch.tensor([int(self.classdict[self.val_frame.iloc[idx, 1]])])     
        val_name = os.path.join(self.root_dir,self.val_frame.iloc[idx, 0])

        val_image = Image.open(val_name)
        val_image = val_image.convert('RGB')
        
        if self.transform:
            val_image = self.transform(val_image)
            
        sample = [imgid,val_name,val_image]
        return sample

############################################################################################################
############################################################################################################

#Defining transformations 
transformations_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),   #crop to 32?
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transformations_val = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),   #crop to 32?
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#defining data loader objects    
trainset = ImageNetDataset(csv_file='file_table.csv',root_dir='tiny-imagenet-200/train/', transform = transformations_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,shuffle=True, num_workers=32)	

valset = ValImages(csv_file = 'tiny-imagenet-200/val/val_annotations.txt',
                   root_dir = 'tiny-imagenet-200/val/images/',
                   transform = transformations_val,
                   classdict = class_dict)
valloader = torch.utils.data.DataLoader(valset, batch_size = 1, shuffle = True, num_workers = 32)


#Defining pretarined models
def resnet101(**kwargs):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 23, 3])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
def resnet18(**kwargs):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


net = models.resnet101(pretrained = True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 4096)
net.load_state_dict(torch.load('checkpoint.ckpt'))

net.to(device)


#TESTING PHASE GENERATING TRAIN EMBEDDINGS AND SAVING IN CSV FILE
train_embeddings = np.zeros([100000,4096])
train_labels = np.zeros([100000,1])
train_imgpath = np.zeros([100000,1])
j=0
net.eval()
with torch.no_grad():   #doesn't calculcate gradient
    for data in trainloader:
        imgid, query_name, query, inclass, outclass = data  
        query = F.interpolate(query,scale_factor = 3.5)
        query = query.to(device)
        query_out = net(query)
        train_embeddings[j:j+10] = (query_out.data).cpu().numpy()      
        train_labels[j:j+10] = (imgid.data).cpu().numpy()
        train_imgpath[j:j+10] = (query_name.data).cpu().numpy()
        j+=10
        print(j)

precision_table = np.zeros([10000,1])
nlp = np.zeros([1260,3])
i=0
j=0
net.eval()
with torch.no_grad():
    for data in valloader:
        imgid, val_name, val = data
        val = F.interpolate(val, scale_factor = 3.5)
        val = val.to(device)
        val_out = net(val)
        val_embeddings = np.tile((val_out.data).cpu().numpy(),(100000,1))    #copying same embed to many rows
        imgid_int = int((imgid.data).cpu())  #this is label of validation image as an integer - tensor input
        
        
        diffs = train_embeddings-val_embeddings     #differnce bw both arrays
        norms = np.linalg.norm(diffs,2,axis = 1)    #l2 norm of each row
        norms = norms.reshape((-1,1))                #ensuring no vector dimension issues
        norms_labels = np.append(norms,train_labels,1)  #second column is labels
        norms_labels_path = np.append(norms_labels,train_imgpath)   
        norms_labels_path = norms_labels_path[norms_labels_path[:,0].argsort()]    #sort by first column
        
        nlp[j:j+30,:] = norms_labels_path[0:30,:]
        top30_list = list(norms_labels_path[0:30,1]) #labels of top 30 images
        correct = top30_list.count(imgid_int)   #counts number of occurences of validation image label
        precision = correct/30
        precision_table[i,:] = precision
        j+=30
        i+=1
        print('validation image {} name is: {}'.format(i,val_name))
        if i == 40:
            break
        
np.savetxt('precision.csv',precision_table,delimiter = ',')        
    