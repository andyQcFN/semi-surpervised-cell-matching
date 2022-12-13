#from matplotlib.image import imsave
#from tensorboard.compat.proto import summary_pb2
import torch
#from torch import tensor
import torch.nn as nn
import torch.optim as optim
import torchvision
# import torchvision.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision.transforms as transforms
#from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
#import cv2
import os
from datetime import datetime

import numpy as np
from numpy import random
from torchvision.utils import save_image
from torchvision.utils import make_grid
import PIL
import uuid
import argparse

#this code generates the dataset, where we can vary the number of neurons and number of sessions to create

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_size", type=int, default=10000, help="number of training instances to be generated")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--folder_name", type=str, default='TrainImages', help="name of the folder containing dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each neuron")
parser.add_argument("--overall_size", type=int, default=150, help="size of overall image of each training instance")
parser.add_argument("--neuron_dropout", type=int, default=0.3, help="the chance that a neuron will dropout of a session")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")


opt = parser.parse_args()
print(opt)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def loadModel(gen, gen_optimizer, filename): #load model

    gen.load_state_dict(torch.load(filename + "_gen"))
    #gen_optimizer.load_state_dict(torch.load(filename + "_gen_optimizer"))

def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)

def boundingBox(image, coord):
    ##this function makes a bounding box for a pytorch image
    #coord [x_min,x_max,y_min, y_max]
    image[coord[0]:coord[0]+1, coord[2]:coord[3]] = 1
    image[coord[1]:coord[1]+1, coord[2]:coord[3]] = 1
    image[coord[0]:coord[1], coord[2]:coord[2]+1] = 1
    image[coord[0]:coord[1], coord[3]:coord[3]+1] = 1

    return image

if __name__ == '__main__':

    LOADMODEL = "100000"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lr = 3e-4
    z_dim = 100
    neuron_dim = 31 * 31 * 1  # 784
    batch_size = 1

    image_dim = opt.overall_size #width and height of generated image

    neuronImageSize = 31

    ##Number of NEURONS PARAMETERS #####

    min_neurons = 30 #number of the actual neurons that there will be
    max_neurons = 50

    min_nbFakeNeurons = 0 #number of fake neurons that you want added
    max_nbFakeNeurons = 35
    
    ####################################
    
    dropoutNeuronChance = opt.neuron_dropout

    datasetSize = opt.dataset_size

    gen = Generator().to(device)
    # fixed_noise = torch.randn((batch_size, z_dim)).to(device)
    fixed_noise = torch.zeros((batch_size, z_dim)).to(device) #fix the random
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    loadModel(gen,opt_gen,LOADMODEL)
    
    
    
    try: 
        os.mkdir(opt.folder_name)
    except OSError as error: 
        print(f"Folder {opt.folder_name} is already made, you may need to reset it to avoid conflicts")  

    for count in range(datasetSize):
        
        nb_neurons = np.random.randint(min_neurons,max_neurons)
        nb_fakeNeurons = np.random.randint(min_nbFakeNeurons,max_nbFakeNeurons)

        baseImage = torch.zeros((image_dim,image_dim),dtype=torch.float).to(device)
        baseImage2 = torch.zeros((image_dim,image_dim),dtype=torch.float).to(device)
        
        
        training_transforms = torchvision.transforms.Compose([ #transforms may cause artifacts. If so, remove everything except normalize
        torchvision.transforms.Resize(size=(31, 31)), 
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.RandomRotation(degrees=30, interpolation=transforms.InterpolationMode.BILINEAR,fill=-1),
        torchvision.transforms.RandomAffine(degrees=(0,0),
                                            translate=(0.1, 0.3),
                                            scale=(0.5, 0.75),
                                            fill = -1),
        torchvision.transforms.RandomPerspective(distortion_scale=0.2, p=.5,fill=-1,interpolation=transforms.InterpolationMode.BILINEAR,),
        torchvision.transforms.Normalize(mean=(0.5,), std=(0.5,)), 
        ])

        secondImage_transform = torchvision.transforms.Compose([
    
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=transforms.InterpolationMode.BILINEAR,fill=-1)
        ])

        neuronPositions = []

        cropSize = 50 #how large each crop is going to be *2
        pad = nn.ZeroPad2d(cropSize) #padding

        #build the session

        for i in range(nb_neurons):
            fixed_noise = torch.randn((batch_size, z_dim)).to(device)
            

            fake = gen(fixed_noise)
            fake = training_transforms(fake)
            #fake[fake < 0] = 0

            # change range of image from [-1,1] to [0,1]

            fake = fake + 1  # [0,2]
            fake = fake - fake.min()
            fake = fake / (fake.max() - fake.min())

            #fake2 = fake.detach().clone()
            fake2 = fake[:,:,:31,:31]
            fake = fake.type(torch.int64)

            coordx = random.randint(image_dim-neuronImageSize)
            coordy = random.randint(image_dim-neuronImageSize)



            randNum = random.rand()

            #if neuron is dropped out do not include in the neuron set
            if randNum > dropoutNeuronChance:
                baseImage[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize] = baseImage[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize]+fake2
                baseImage2[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize] = baseImage2[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize]+fake2
                neuronPositions.append([coordx+16,coordy+16]) #+16 because we want to get the centroid of the neuron +crop size because we add padding later

            else:
                baseImage[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize] = baseImage[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize]+fake2



        for i in range(nb_fakeNeurons):
            fixed_noise = torch.randn((batch_size, z_dim)).to(device)
            

            fake = gen(fixed_noise)
            fake = training_transforms(fake)
            #fake [fake < 0] = 0 #convert negative to 0

            #change range of image from [-1,1] to [0,1]

            fake = fake + 1 #[0,2]
            fake = fake - fake.min()
            fake = fake / (fake.max()-fake.min())

            #fake2 = fake.detach().clone()
            fake2 = fake[:, :, :31, :31]
            fake = fake.type(torch.int64)

            coordx = random.randint(image_dim-neuronImageSize)
            coordy = random.randint(image_dim-neuronImageSize)
            
            baseImage2[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize] = baseImage2[coordx:coordx+neuronImageSize,coordy:coordy+neuronImageSize]+fake2



        #create the files

        for i in range(len(neuronPositions)):

            sameNeuron = random.randint(2)
            randomNeuronIdx = i

            if sameNeuron == 0:
                
                randomNeuronIdx = random.randint(len(neuronPositions))

            randomNeuron1 = neuronPositions[i]
            randomNeuron2 = neuronPositions[randomNeuronIdx]


            neuronImage = baseImage.detach().clone()
            neuronImage = pad(neuronImage)

            neuronImage2 = baseImage2.detach().clone()
            neuronImage2 = pad(neuronImage2)

            '''
            #for visualizations
            
            boundingBox(neuronImage, [randomNeuron1[0] + cropSize, randomNeuron1[0] + cropSize + 16,
                                 randomNeuron1[1] + cropSize, randomNeuron1[1] + cropSize + 16])
            
            boundingBox(neuronImage2, [randomNeuron2[0] + cropSize, randomNeuron2[0] + cropSize + 16,
                                 randomNeuron2[1] + cropSize, randomNeuron2[1] + cropSize + 16])
                                 
            '''

            image1 = neuronImage[max(0,randomNeuron1[0]):randomNeuron1[0]+ cropSize * 2,max(0,randomNeuron1[1]):randomNeuron1[1]+cropSize*2]

            image2 = neuronImage2[max(0,randomNeuron2[0]):randomNeuron2[0]+ cropSize * 2,max(0,randomNeuron2[1]):randomNeuron2[1]+cropSize*2]
            fileName1 = 'TrainImages/'+uuid.uuid4().hex
            fileName2 = 'TrainImages/'+uuid.uuid4().hex

            if randomNeuronIdx == i:
                append_new_line("trainSet.txt",f'{fileName1} \t {fileName2} \t {1}')
            else:
                append_new_line("trainSet.txt",f'{fileName1} \t {fileName2} \t {0}')


            image2 = image2.float()

            image2 = image2.reshape(-1, 1, cropSize*2, cropSize*2)


            image2 = secondImage_transform(image2) #apply distortion
            image2 = image2.float() 


            flip = random.randint(2) #coin flip to decide if the order of the two images (since one has dropout)
            
           
            if flip == 0:

                torch.save(image1, fileName1)
                torch.save(image2[0][0],fileName2)
            else: 
                torch.save(image1, fileName2)
                torch.save(image2[0][0],fileName1)
                
            '''
            #for visualizations
            
            save_image(image1,f"{fileName1}.png")
            save_image(image2[0][0],f"{fileName2}.png")
            '''
        
            
      
       

        


    


