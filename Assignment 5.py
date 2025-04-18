#!/usr/bin/env python
# coding: utf-8

# In[2]:


from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import DataLoader as torch_dataloader
import torchvision
from torchvision import datasets, transforms
import torchvision.models as tv_models


# In[3]:


#Build Dataloader


# In[4]:


import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io as io
import glob
import pandas as pd
#%%
class MyDataset(torch_dataset):
    def __init__(self, path, filenamelist, labellist):
        self.path=path
        self.filenamelist=filenamelist
        self.labellist=labellist
    def __len__(self):
        #return the number of data points
        return len(self.filenamelist)
    def __getitem__(self, idx):
        I=io.imread(self.path+self.filenamelist[idx])
        I=skimage.util.img_as_float32(I)
        I = I.reshape(1,I.shape[0],I.shape[1])
        I = torch.tensor(I, dtype=torch.float32)
        I = I.expand(3, I.shape[1],I.shape[2])
        label=torch.tensor(self.labellist[idx], dtype=torch.int64)
        return I, label


# In[5]:


def get_dataloader(path='/Users/sushmitapersaud/Downloads/hw5s 4/S224/'):    
       df_train = pd.read_csv('/Users/sushmitapersaud/Downloads/hw5s 4/S224/train.csv')
       df_val = pd.read_csv('/Users/sushmitapersaud/Downloads/hw5s 4/S224/val.csv')
       df_test = pd.read_csv('/Users/sushmitapersaud/Downloads/hw5s 4/S224/test.csv')

       dataset_train = MyDataset(path, df_train['filename'].values, df_train['label'].values)
       dataset_val = MyDataset(path, df_val['filename'].values, df_val['label'].values)
       dataset_test = MyDataset(path, df_test['filename'].values, df_test['label'].values)

       loader_train = torch_dataloader(dataset_train, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
       loader_val = torch_dataloader(dataset_val, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
       loader_test = torch_dataloader(dataset_test, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
       
       print("Returning 6 values from get_dataloader")
       return loader_train, loader_val, loader_test, dataset_train, dataset_val, dataset_test


# In[6]:


get_dataloader()


# In[7]:


loader_train, loader_val, loader_test, dataset_train, dataset_val, dataset_test = get_dataloader()


# In[8]:


len(dataset_train)


# In[9]:


len(loader_train)


# In[10]:


len(dataset_val)


# In[11]:


len(loader_val)


# In[12]:


(x,label)=dataset_train[0]
print(x.size())
print(label)


# In[13]:


(x,label)=dataset_train[200]
print(x.size())
print(label)


# In[14]:


fig, ax = plt.subplots(figsize=(3, 3))
for n in range(2, 10, 1):
    x = dataset_train[n][0].detach().cpu().numpy()
    y = dataset_train[n][1]
    x = x.transpose(1,2,0)
    ax.imshow(x)
    ax.set_title('label: ' + str(y), fontsize=16)
    ax.axis('off')
    display.clear_output(wait=False)
    display.display(fig)
    plt.pause(0.5)


# In[15]:


#Define a CNN based on Resnet50
class Net(nn.Module):
    def __init__(self):
        super().__init__()        
        #use resnet50 as the base model
        self.resnet50 = tv_models.resnet50()
        #modified the last layer for binary classification  
        self.resnet50.fc=torch.nn.Linear(2048, 1)           
    
    def forward(self, x):
        z = self.resnet50(x)
        z = z.view(-1)
        return z


# In[16]:


def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)


# In[17]:


tv_models.resnet50()


# In[64]:


#The function to train the model in one epoch
def train(model, device, optimizer, dataloader, epoch):    
    model.train()#set model to training mode
    loss_train=0
    acc_train =0 
    for batch_idx, (X, Y) in enumerate(dataloader):
        #print(X.shape, Y.shape)
        #print(X.dtype, Y.dtype)
        Y = Y.to(X.dtype)
        X, Y = X.to(device), Y.to(device)
        Z = model(X)#forward pass
        loss = nnF.binary_cross_entropy_with_logits(Z, Y)
        optimizer.zero_grad()#clear grad of each parameter
        loss.backward()#backward pass
        optimizer.step()#update parameters
        loss_train+=loss.item()        
        Yp = (Z.data > 0).to(torch.int64)
        Y = Y.to(torch.int64)
        acc_train+= torch.sum(Yp==Y).item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 100. * batch_idx / len(dataloader), loss.item()))
    loss_train/=len(dataloader)
    acc_train/=len(dataloader.dataset) 
    return loss_train, acc_train


# In[65]:


#The Function to test the model
def test(model, device, dataloader):
    model.eval()#set model to evaluation mode
    loss_test=0
    acc_test =0
    Confusion=np.zeros((2,2))
    with torch.no_grad(): # tell Pytorch not to build graph in the with section
        for batch_idx, (X, Y) in enumerate(dataloader):     
            Y = Y.to(X.dtype)
            X, Y = X.to(device), Y.to(device)
            Z = model(X)#forward pass            
            loss = nnF.binary_cross_entropy_with_logits(Z, Y)
            loss_test+=loss.item()
            Yp = (Z.data > 0).to(torch.int64)
            Y = Y.to(torch.int64)
            acc_test+= torch.sum(Yp==Y).item()
            for i in range(0, 2):
                for j in range(0, 2):
                    Confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
    loss_test/=len(dataloader)        
    acc_test/=len(dataloader.dataset)
    Sens=np.zeros(2)
    Prec=np.zeros(2)   
    for n in range(0, 2):
        TP=Confusion[n,n]
        FN=np.sum(Confusion[n,:])-TP
        FP=np.sum(Confusion[:,n])-TP
        Sens[n]=TP/(TP+FN)
        Prec[n]=TP/(TP+FP)    
    Acc = Confusion.diagonal().sum()/Confusion.sum() # should be the same as acc_test
    return loss_test, acc_test, (Confusion, Acc, Sens, Prec)


# In[66]:


#Create a model, and start the training-validation process
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
model=Net()
model.to(device)
#optimizer = optim.SGD(model.get_trainable_parameters(), lr=0.0001, momentum=0.99) 
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.99) 
#---------------------------------------------------------
(x,label)=dataset_train[0]
x=x.view(1,3,224,224).to(device)
z=model(x)
#
#run this whenever creating a new model
loss_train_list=[]
acc_train_list=[]
loss_val_list=[]
acc_val_list=[]
epoch_save=-1


# In[21]:


z


# In[22]:


y_hat = torch.sigmoid(z)
y_hat


# In[23]:


z.shape


# In[24]:


def plot_result(loss_train_list, acc_train_list, 
                loss_val_list, acc_val_list):    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].set_title('loss v.s. epoch',fontsize=16)
    ax[0].plot(loss_train_list, '-b', label='training loss')
    ax[0].plot(loss_val_list, '-g', label='validation loss')
    ax[0].set_xlabel('epoch',fontsize=16)
    #ax[0].set_xticks(np.arange(len(loss_train_list)))
    ax[0].legend(fontsize=16)
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch',fontsize=16)
    ax[1].plot(acc_train_list, '-b', label='training accuracy')
    ax[1].plot(acc_val_list, '-g', label='validation accuracy')
    ax[1].set_xlabel('epoch',fontsize=16)
    #ax[1].set_xticks(np.arange(len(loss_train_list)))
    ax[1].legend(fontsize=16)
    ax[1].grid(True)
    return fig, ax


# In[25]:


#update learning reate
lr_new=0.0001
for g in optimizer.param_groups:
    g['lr']=lr_new


# In[26]:


for epoch in range(epoch_save+1, epoch_save+30):
    t0=time.time()
    #-------- training --------------------------------
    loss_train, acc_train =train(model, device, optimizer, loader_train, epoch)    
    loss_train_list.append(loss_train)
    acc_train_list.append(acc_train)
    print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
    t1=time.time()
    print("time cost", t1-t0)
    #-------- validation --------------------------------
    loss_val, acc_val, other_val = test(model, device, loader_val)
    loss_val_list.append(loss_val)
    acc_val_list.append(acc_val)
    print('epoch', epoch, 'validation loss:', loss_val, 'acc:', acc_val)   
    #--------save model-------------------------
    result = (loss_train_list, acc_train_list, 
              loss_val_list, acc_val_list, other_val)
    save_checkpoint('CNN_LS_Pytorch_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
    epoch_save=epoch
    #------- show result ----------------------
    display.clear_output(wait=False)
    plt.close('all')
    fig, ax = plot_result(loss_train_list, acc_train_list, 
                          loss_val_list, acc_val_list)
    display.display(fig)


# In[27]:


plot_result(loss_train_list, acc_train_list, 
            loss_val_list, acc_val_list)


# In[28]:


best_id= np.argmax(acc_val_list)
best_id


# In[29]:


epoch_save=best_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load('CNN_LS_Pytorch_epoch'+str(epoch_save)+'.pt', map_location=device)
model=Net()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
#
#optimizer = optim.SGD(model.get_trainable_parameters(), lr=0.0001, momentum=0.99) 
#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
(loss_train_list, acc_train_list, 
 loss_val_list, acc_val_list, other_val) = checkpoint['result']   


# In[30]:


loss_val, acc_val, (Confusion, Acc, Sens, Prec) = test(model, device, loader_val)
Confusion_sens=Confusion.copy()
for n in range(0, 2):
    Confusion_sens[n,:]/=np.sum(Confusion[n,:])
Confusion_prec=Confusion.copy()
for n in range(0, 2):
    Confusion_prec[:,n]/=np.sum(Confusion[:,n])
print('Accuracy (average)', acc_val)
print('Accuracy (average)', Acc)
print('Sensitivity', Sens)
print('Precision', Prec)
print('Confusion_sens \n', Confusion_sens)
print('Confusion_prec \n', Confusion_prec)


# In[18]:


#Transfer Learning
from torchvision.models import ResNet50_Weights
class Net(nn.Module):
    def __init__(self):
        super().__init__()        
        #use resnet50 as the base model
        #self.resnet50 = tv_models.resnet50(pretrained=True) #old Pytorch
        self.resnet50 = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
        #modified the last layer for binary classification  
        self.resnet50.fc=torch.nn.Linear(2048, 2)           
        #freeze all parameters
        for p in self.resnet50.parameters():
            p.requires_grad = False 
        #set the parameters of layer4 to be trainable       
        for p in self.resnet50.layer4.parameters():
            p.requires_grad = True       
        #set the parameters of fc to be trainable       
        for p in self.resnet50.fc.parameters():
            p.requires_grad = True       
        
    def get_trainable_parameters(self):
        pList=list(self.resnet50.layer4.parameters())+list(self.resnet50.fc.parameters())
        return pList
    
    def forward(self,x):
        z = self.resnet50(x)
        return z


# In[19]:


def save_checkpoint(filename, model, optimizer, result, epoch):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'result':result},
               filename)
    print('saved:', filename)


# In[20]:


tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)


# In[21]:


def train(model, device, optimizer, dataloader, epoch):    
    model.train()#set model to training mode
    loss_train=0
    acc_train =0 
    for batch_idx, (X, Y) in enumerate(dataloader):
        #print(X.shape, Y.shape)
        #print(X.dtype, Y.dtype)
        X, Y = X.to(device), Y.to(device)
        Z = model(X)#forward pass
        loss = nnF.cross_entropy(Z, Y)
        optimizer.zero_grad()#clear grad of each parameter
        loss.backward()#backward pass
        optimizer.step()#update parameters
        loss_train+=loss.item()        
        Yp = Z.max(dim=-1)[1]
        acc_train+= torch.sum(Yp==Y).item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:.0f}%]\tLoss: {:.6f}'.format(
                    epoch, 100. * batch_idx / len(dataloader), loss.item()))
    loss_train/=len(dataloader)
    acc_train/=len(dataloader.dataset) 
    return loss_train, acc_train


# In[22]:


def test(model, device, dataloader):
    model.eval()#set model to evaluation mode
    loss_test=0
    acc_test =0
    Confusion=np.zeros((2,2))
    with torch.no_grad(): # tell Pytorch not to build graph in the with section
        for batch_idx, (X, Y) in enumerate(dataloader):     
            X, Y = X.to(device), Y.to(device)
            Z = model(X)#forward pass            
            loss = nnF.cross_entropy(Z, Y)
            loss_test+=loss.item()
            Yp = Z.max(dim=-1)[1]
            acc_test+= torch.sum(Yp==Y).item()
            for i in range(0, 2):
                for j in range(0, 2):
                    Confusion[i,j]+=torch.sum((Y==i)&(Yp==j)).item()
    loss_test/=len(dataloader)        
    acc_test/=len(dataloader.dataset)
    Sens=np.zeros(2)
    Prec=np.zeros(2)   
    for n in range(0, 2):
        TP=Confusion[n,n]
        FN=np.sum(Confusion[n,:])-TP
        FP=np.sum(Confusion[:,n])-TP
        Sens[n]=TP/(TP+FN)
        Prec[n]=TP/(TP+FP)    
    Acc = Confusion.diagonal().sum()/Confusion.sum() # should be the same as acc_test
    return loss_test, acc_test, (Confusion, Acc, Sens, Prec)


# In[23]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device', device)
model=Net()
model.to(device)
optimizer = optim.SGD(model.get_trainable_parameters(), lr=0.0001, momentum=0.99) 
#---------------------------------------------------------
(x,label)=dataset_train[0]
x=x.view(1,3,224,224).to(device)
z=model(x)
#
#run this whenever creating a new model
loss_train_list=[]
acc_train_list=[]
loss_val_list=[]
acc_val_list=[]
epoch_save=-1


# In[24]:


z


# In[25]:


y_hat = torch.sigmoid(z)
y_hat


# In[26]:


z.shape


# In[27]:


def plot_result(loss_train_list, acc_train_list, 
                loss_val_list, acc_val_list):    
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].set_title('loss v.s. epoch',fontsize=16)
    ax[0].plot(loss_train_list, '-b', label='training loss')
    ax[0].plot(loss_val_list, '-g', label='validation loss')
    ax[0].set_xlabel('epoch',fontsize=16)
    #ax[0].set_xticks(np.arange(len(loss_train_list)))
    ax[0].legend(fontsize=16)
    ax[0].grid(True)
    ax[1].set_title('accuracy v.s. epoch',fontsize=16)
    ax[1].plot(acc_train_list, '-b', label='training accuracy')
    ax[1].plot(acc_val_list, '-g', label='validation accuracy')
    ax[1].set_xlabel('epoch',fontsize=16)
    #ax[1].set_xticks(np.arange(len(loss_train_list)))
    ax[1].legend(fontsize=16)
    ax[1].grid(True)
    return fig, ax


# In[28]:


#update learning reate
lr_new=0.0001
for g in optimizer.param_groups:
    g['lr']=lr_new


# In[32]:


for epoch in range(epoch_save+1, 10):
    t0=time.time()
    #-------- training --------------------------------
    loss_train, acc_train =train(model, device, optimizer, loader_train, epoch)    
    loss_train_list.append(loss_train)
    acc_train_list.append(acc_train)
    print('epoch', epoch, 'training loss:', loss_train, 'acc:', acc_train)
    #-------- validation --------------------------------
    loss_val, acc_val, other_val = test(model, device, loader_val)
    loss_val_list.append(loss_val)
    acc_val_list.append(acc_val)
    print('epoch', epoch, 'validation loss:', loss_val, 'acc:', acc_val)   
    t1=time.time()
    print("time cost", t1-t0)
    #--------save model-------------------------
    result = (loss_train_list, acc_train_list, 
              loss_val_list, acc_val_list, other_val)
    save_checkpoint('CNN_TL_Pytorch_ce_epoch'+str(epoch)+'.pt', model, optimizer, result, epoch)
    epoch_save=epoch
    #------- show result ----------------------
    display.clear_output(wait=False)
    plt.close('all')
    fig, ax = plot_result(loss_train_list, acc_train_list, 
                          loss_val_list, acc_val_list)
    display.display(fig)


# In[36]:


plot_result(loss_train_list, acc_train_list, 
            loss_val_list, acc_val_list)


# In[37]:


best_id= np.array(acc_val_list).argmax()
best_id


# In[38]:


epoch_save=best_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint=torch.load('CNN_TL_Pytorch_ce_epoch'+str(epoch_save)+'.pt', map_location=device)
model=Net()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
(loss_train_list, acc_train_list, 
 loss_val_list, acc_val_list, other_val) = checkpoint['result']  


# In[39]:


loss_val, acc_val, (Confusion, Acc, Sens, Prec) = test(model, device, loader_val)
Confusion_sens=Confusion.copy()
for n in range(0, 2):
    Confusion_sens[n,:]/=np.sum(Confusion[n,:])
Confusion_prec=Confusion.copy()
for n in range(0, 2):
    Confusion_prec[:,n]/=np.sum(Confusion[:,n])
print('Accuracy (average)', acc_val)
print('Accuracy (average)', Acc)
print('Sensitivity', Sens)
print('Precision', Prec)
print('Confusion_sens \n', Confusion_sens)
print('Confusion_prec \n', Confusion_prec)


# In[48]:


get_ipython().system('pip install grad-cam')


# In[38]:


#GradCAM
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage


# In[141]:


#choose the target layer(s)
target_layers = [model.layer4[-1]]


# In[142]:


#load a single rgb image from harddrive
image=skimage.io.imread("/Users/sushmitapersaud/Downloads/hw5s 4/S224/COVID/Covid (1).png")
image=image.astype("float32")
image=image/image.max()
image.shape
print(image.shape)
plt.imshow(image)


# In[146]:


image=image.reshape(224,224,1)
image=np.concatenate([image, image, image])
image_input = torch.tensor(image).permute(2, 0, 1) # (224,224,3) to (3,224,224)


# In[151]:


#convert numpy array to pytorch tensor
image_input = torch.tensor(image).permute(2, 0, 1) # (224,224,3) to (3,224,224)
image_input=image_input.reshape(1,3,224,224)# a batch that only has one image
image_input.shape


# In[152]:


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 0:
            return model_output
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# In[153]:


model_output=model(image_input)
model_output.shape


# In[154]:


# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
#category 281 is tabby, tabby cat

targets = [ClassifierOutputTarget(category=281)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# pay attention to the shape of each input
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam.shape


# In[155]:


grayscale_cam=grayscale_cam[0]
grayscale_cam.shape


# In[156]:


plt.imshow(grayscale_cam, cmap='gray')


# In[159]:


# In this example grayscale_cam has only one image in the batch:
heatmap_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, heatmap_resized, use_rgb=True)


# In[160]:


cam_image.shape


# In[161]:


plt.imshow(cam_image)


# In[163]:


cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
targets = [ClassifierOutputTarget(category=281)]
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam=grayscale_cam[0]
grayscale_cam_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, grayscale_cam_resized, use_rgb=True)
fig, ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(image)
ax[1].imshow(grayscale_cam, cmap='gray')
ax[2].imshow(cam_image)


# In[165]:


#EigenCAM
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)
targets = [ClassifierOutputTarget(category=281)]
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam=grayscale_cam[0]
grayscale_cam_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, grayscale_cam_resized, use_rgb=True)
fig, ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(image)
ax[1].imshow(grayscale_cam, cmap='gray')
ax[2].imshow(cam_image)


# In[31]:


target_layers = [model.resnet50.layer4[-1]]


# In[32]:


#load a single rgb image from harddrive
image=skimage.io.imread("/Users/sushmitapersaud/Downloads/hw5s 4/S224/COVID/Covid (1).png")
image=image.astype("float32")
image=image/image.max()
image.shape
print(image.shape)
plt.imshow(image)


# In[33]:


image=image.reshape(224,224,1)
image=np.concatenate([image, image, image])
image_input = torch.tensor(image).permute(2, 0, 1) # (224,224,3) to (3,224,224)


# In[34]:


#convert numpy array to pytorch tensor
image_input = torch.tensor(image).permute(2, 0, 1) # (224,224,3) to (3,224,224)
image_input=image_input.reshape(1,3,224,224)# a batch that only has one image
image_input.shape


# In[35]:


class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 0:
            return model_output
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# In[36]:


model_output=model(image_input)
model_output.shape


# In[41]:


# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

# https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
#category 281 is tabby, tabby cat

targets = [ClassifierOutputTarget(category=1)]

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
# pay attention to the shape of each input
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam.shape


# In[42]:


grayscale_cam=grayscale_cam[0]
grayscale_cam.shape


# In[43]:


plt.imshow(grayscale_cam, cmap='gray')


# In[49]:


import cv2
# In this example grayscale_cam has only one image in the batch:
heatmap_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, heatmap_resized, use_rgb=True)


# In[50]:


cam_image.shape


# In[51]:


plt.imshow(cam_image)


# In[53]:


cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
targets = [ClassifierOutputTarget(category=1)]
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam=grayscale_cam[0]
grayscale_cam_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, grayscale_cam_resized, use_rgb=True)
fig, ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(image)
ax[1].imshow(grayscale_cam, cmap='gray')
ax[2].imshow(cam_image)


# In[54]:


#EigenCAM
cam = EigenCAM(model=model, target_layers=target_layers, use_cuda=False)
targets = [ClassifierOutputTarget(category=281)]
grayscale_cam = cam(input_tensor=image_input, targets=targets)
grayscale_cam=grayscale_cam[0]
grayscale_cam_resized = cv2.resize(grayscale_cam, (image.shape[1], image.shape[0]))
cam_image = show_cam_on_image(image, grayscale_cam_resized, use_rgb=True)
fig, ax = plt.subplots(1,3, figsize=(10,10))
ax[0].imshow(image)
ax[1].imshow(grayscale_cam, cmap='gray')
ax[2].imshow(cam_image)


# In[ ]:




