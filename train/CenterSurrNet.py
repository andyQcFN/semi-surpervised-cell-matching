import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import os

imgSize = 50 #how large the center crop will be


def main():
    cwd = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training in", device)


    norm_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ])


    dataset = CentralSurroundDataset(cwd, 'trainSet.txt', transform = norm_transform)
    batchSize = 1024
    trainloader = DataLoader(dataset, shuffle=True, batch_size=batchSize)
    
    valSet = CentralSurroundDataset(cwd, 'trainSet.txt', transform = norm_transform) #input a different validation set
    batchSize = 1024
    valLoader = DataLoader(valSet, shuffle=True, batch_size=batchSize)
    

    model = CenterSurrNet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = 6e-4, weight_decay=1e-5)

    '''
    #For loading a model checkpoint
    
    model_file = ''
    loadModel(model, optimizer, model_file)
    print("Loaded " + model_file)
    '''
    

    criterion = nn.BCEWithLogitsLoss()
    num_epochs = 10

    for epoch in range(num_epochs):
    
        correctVal = 0
        correctTrain = 0
       
        for i, (img1_set, img2_set, labels) in enumerate(trainloader):

            if device:
                img1_set = img1_set.to(device=device, dtype=torch.float)
                img2_set = img2_set.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.float)
           
            labels = Variable(labels.view(-1, 1).float())

         
            output_labels_prob = model(img1_set,img2_set)
            
            loss = criterion(output_labels_prob, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            output = (output_labels_prob>0.5).float()
            correctTrain += (output == labels).float().sum()
            
        accuracy = 100 * correctTrain / len(dataset)
      
        torch.save(model.state_dict(), str(num_epochs) + "_2Stream")
        torch.save(optimizer.state_dict(), str(num_epochs) + "_2StreamOptimizer")
        print(f'Epoch {epoch+1}, Iter {i+1} Loss: {loss}')
        
        
        with torch.no_grad():
            for i, (img1_set,img2_set,labels) in enumerate(valLoader):

                if device:
                    img1_set = img1_set.to(device=device, dtype=torch.float)
                    img2_set = img2_set.to(device=device, dtype=torch.float)
                    labels = labels.to(device=device, dtype=torch.float)
                

                labels = Variable(labels.view(-1, 1).float())
                
                output_labels_prob = model(img1_set,img2_set)

                loss = criterion(output_labels_prob, labels)
  
                
                
                output2 = (output_labels_prob>0.5).float()
                correctVal += (output2 == labels).float().sum()
        
        
        valAccuracy = 100 * correctVal / len(valSet)
        print(f'Train Accuracy: {accuracy} --- Validation Accuracy: {valAccuracy}')
            
        


    torch.save(model.state_dict(), str(num_epochs) + "_2Channel")
    torch.save(optimizer.state_dict(), str(num_epochs) + "_2ChannelOptimizer")

class CenterSurrNet(nn.Module):
    def __init__(self):
        super(CenterSurrNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, 10) 
        self.conv2 = nn.Conv2d(64, 128, 7)  
        self.conv3 = nn.Conv2d(128, 128, 4)
        self.conv4 = nn.Conv2d(128, 256, 4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        
        #32 image size
        # self.fc1 = nn.Linear(1024, 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fcOut = nn.Linear(1024, 1)
        
        #50 image size
        self.fc1 = nn.Linear(12544, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fcOut = nn.Linear(1024, 1)
        
        self.sigmoid = nn.Sigmoid()

    def convs(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, (2,2)) 
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, (2,2))
   
        return x

    def forward(self, x1,x2):
        x1 = self.convs(x1)
        x2 = self.convs(x2)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0],-1)
        x = torch.cat((x1,x2),1)

        #SingleLayer1Output
        #x = self.fcOut(x)
        
        #3Layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fcOut(x)
   
        return x
        
class CentralSurroundDataset(Dataset):
    def __init__(self, root_dir, path_file_dir, transform=None, random_aug=False):
        self.root_dir = root_dir
        path_file = open(path_file_dir, 'r')
        data = []

        for line in path_file:
            line = line.strip()
            img1, img2, label = line.split()
            label = int(label)
            data.append((img1, img2, label))
        

        self.data = data
        self.transform = transform
        self.random_aug = random_aug
        self.random_aug_prob = 0.7
        path_file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        resize_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(imgSize, imgSize)),
            ])

        img1, img2, label = self.data[idx]
        img1_file = torch.load(os.path.join(self.root_dir, img1))
        img1_file = img1_file.unsqueeze(0) #unsqueeze is [400,400] --> [1,400,400]
        img2_file = torch.load(os.path.join(self.root_dir, img2))
        img2_file = img2_file.unsqueeze(0)
      
        downSample1 = resize_transform(img1_file)
        downSample2 = resize_transform(img2_file)

        if self.random_aug:
            img1_file = self.random_augmentation(img1_file, self.random_aug_prob)
            img2_file = self.random_augmentation(img2_file, self.random_aug_prob)

        if self.transform:
            img1_file = self.transform(img1_file)
            img2_file = self.transform(img2_file)
        # return (img1_file, img2_file, label)
        
        lower = int(50 - imgSize/2)
        upper = int(50 + imgSize/2)
        
        
        return (torch.stack([img1_file[0][lower:upper, lower:upper],img2_file[0][lower:upper, lower:upper]]),torch.stack([downSample1[0],downSample2[0]]), label)


def loadModel(model,optim,filename):
    model.load_state_dict(torch.load(filename))
    optim.load_state_dict(torch.load(filename+'Optimizer'))
    model.train()



if __name__ == '__main__':
    main()
