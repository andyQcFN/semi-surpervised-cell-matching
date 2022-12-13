import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn.functional as F


imgSize = 50

input2 = 'example_input2.txt'
input1 = 'example_input1.txt'

#threshold distance to check for neuron matches to lower runtime
searchRadius = 10



# here we are matching neurons real data obtained from CNMFE, the matching is based on the 'input1' and 'input2' 
def main():
    cwd = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training in", device)

    dataset = MatchingSet(cwd, input1, input2)
    batchSize = 1
    testloader = DataLoader(dataset, shuffle=False, batch_size=batchSize)
    
    #load the name of the trained model
    modelPath = 'trained_models/220_2Stream'

    model = CenterSurrNet()

    loadModel(model, modelPath)

    model.to(device)
    model.eval()

    eval(model, testloader, dataset.numWay, device)
    
    
    
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


class MatchingSet(Dataset):
    def __init__(self, root_dir, path_file_dir, path_file_dir2, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        path_file = open(path_file_dir, 'r')
        path_file2 = open(path_file_dir2, 'r')

        data = []
        for line in path_file:
            line = line.strip()
            img1, idx, x, y = line.split()  # idx is the neuron ID
            data.append((img1, idx, x, y))

        data2 = []
        for line in path_file2:
            line = line.strip()
            img1, idx, x, y = line.split()
            data2.append((img1, idx, x, y))

        self.data = data
        self.data2 = data2
        self.numWay = len(data2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        [img1, neuronID, x1, y1] = self.data[idx]  # img1 will be the main image and img2 will be the label

        img1_file = torch.load(os.path.join(self.root_dir, img1))
        img1_file = img1_file.unsqueeze(0)  

        # find n numbers of distinct images, 1 in the same set as the main
        testSet = []
 
        for i in range(self.numWay):
            [img2, neuronID2, x2, y2] = self.data2[i]
            img2_file = torch.load(os.path.join(self.root_dir, img2))
            img2_file = img2_file.unsqueeze(0)

            testSet.append((img2_file, neuronID2, x2, y2))
        return img1_file, neuronID, x1, y1, testSet


def eval(model, test_loader, numWay, device):

    resize_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(imgSize, imgSize)),
            ])

    globalTracker = [] #this will keep the predValue and the match indexes as a tuple (sorted later)
  
    with torch.no_grad():
        model.eval()

        print('Matching started')
        count = 0
        for mainImg, (neuronID,), (x1,), (y1,), imgSets in test_loader:

            neuronLoc1 = np.array((int(x1), int(y1))) #location of neuron1

            mainImg = mainImg.to(device)
            threshold = 0.5
   
            

            for testImg, (neuronID2,), (x2,), (y2,) in imgSets:
                neuronLoc2 = np.array((int(x2), int(y2)))

                eucDistance = np.linalg.norm(neuronLoc1-neuronLoc2) #euclidean distance

                if eucDistance > searchRadius: #if the neuron is too far away just ignore it
                    continue
        
                    
                testImg = testImg.to(device)
               
                lower = int(imgSize - imgSize/2)
                upper = int(imgSize + imgSize/2)
                
                downSample1 = resize_transform(mainImg)
                downSample2 = resize_transform(testImg)
                    
            
                 
                x1 = torch.stack([mainImg[0][0][lower:upper, lower:upper],testImg[0][0][lower:upper, lower:upper]])
                x1 = x1[None,:]
                
                x2 = torch.stack([downSample1[0][0],downSample2[0][0]])
                x2 = x2[None,:]
           
                
                output = model(x1,x2)
  

                if output > threshold:
                    globalTracker.append((output,int(neuronID),int(neuronID2)))
      
            count += 1
        
        globalTracker.sort(reverse=True)
        print(globalTracker)

        matchesMap1 = {} #typical first session maps to second session x->y
        matchesMap2 = {} #the reverse map direction y->x

        for i in range(len(test_loader)):
            matchesMap1[i+1] = (-1, -1)

        for i in range(numWay):
            matchesMap2[i+1] = -1

        for i in range(len(globalTracker)):

            if globalTracker[i][1] == 0:
                break

            idOne = globalTracker[i][1]
            idTwo = globalTracker[i][2]

            if matchesMap1[idOne][0] == -1 and matchesMap2[idTwo] == -1:
                matchesMap1[idOne] = (idTwo, float(globalTracker[i][0]))
                matchesMap2[idTwo] = idOne

        print(matchesMap1)

        for i in range(len(test_loader)):
            #output file is here, can change the name of the file to any preference
            append_new_line(f"{input1[0:9]}_{input2[0:9]}MatchesRad{searchRadius}2Stream.csv", f'{i+1},{matchesMap1[i+1][0]},{float(matchesMap1[i+1][1])}')




def loadModel(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()


def append_new_line(file_name, text_to_append):
   
    with open(file_name, "a+") as file:
        file.seek(0)
        data = file.read(100)
        if len(data) > 0:
            file.write("\n")

        file.write(text_to_append)


if __name__ == '__main__':
    main()
