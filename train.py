# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser (description = 'train.py')

parser.add_argument ('--data_dir', help = 'Data Directory: Basic Usage', type = str, default="./flowers/")
parser.add_argument ('--save_dir', help = 'Saving Directory', type = str, default='./checkpoint.pth')
parser.add_argument ('--arch', help = 'Select between three architectures: vgg16,densenet121,alexnet ?(default: vgg16)', type = str, default = 'vgg16')
parser.add_argument ('--dropout', help = 'Dropout', type = float, default=.2)
parser.add_argument ('--learning_rate', help = 'Learning rate', type = float, default=.001)
parser.add_argument ('--hidden_layer1', help = 'First Hidden Layer', type = int, default = 4096)
parser.add_argument ('--hidden_layer2', help = 'Second Hidden Layer', type = int, default = 512)
parser.add_argument ('--epochs', help = 'Epochs', type = int, default = 4)
parser.add_argument ('--GPU', help = 'Using GPU', type = str, default = 'GPU')

args = parser.parse_args ()

if args.data_dir:
    data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

 

if args.dropout:
    dropout = args.dropout
if args.hidden_layer1:
    hidden_layer1 = args.hidden_layer1
if args.hidden_layer2:
    hidden_layer2 = args.hidden_layer2
if args.learning_rate:
    learning_rate = args.learning_rate
if args.epochs:
    epochs = args.epochs
if args.arch:
    arch = args.arch


# Define your transforms for the training, validation, and testing sets
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                                      
validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.RandomResizedCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                                      
testing_transforms = transforms.Compose([transforms.Resize(255),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
                                     
# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=training_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
test_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
                       
    
    
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    


def nn_network(arch, hidden_layer1,hidden_layer2,learning_rate,dropout):
    '''This function is used to build the network.
    Input - Number of hidden layers, learning rate and drop out value
    Output - Returns model, optimizer ,criterion
    '''

    if arch == 'vgg16':
        # Build and train your network vgg16
        model = models.vgg16(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            features = 25088
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(features, hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
        
    elif arch == 'alexnet':
        # Build and train your network alexnet
        model = models.alexnet(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            features = 9216
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(features, hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
     
    elif arch == 'densenet121':
        # TODO: Build and train your network densenet121
        model = models.densenet121(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
            features = 9216
            classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(features, hidden_layer1)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(hidden_layer2, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
    
     # TODO: Create the network, define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    return model, optimizer ,criterion

model, optimizer, criterion = nn_network(arch, hidden_layer1, hidden_layer2, learning_rate,dropout)
# The parameters of the feedforward classifier are appropriately getting trained
model.to (device)
steps = 0
print_every = 50

for e in range(epochs):
    # Make sure training is back on
    model.train()
    
    for input_label in iter(trainloader):
        running_loss = 0
        inputs, labels = input_label
        inputs, labels = inputs.to(device), labels.to(device)
        steps += 1
        
        optimizer.zero_grad()
        output = model.forward(inputs)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        

        if steps % print_every == 0:
            # For Dropout to be turned off
            model.eval () 
            accuracy = 0
            
            for input_label2 in iter(validationloader):
                inputs2, labels2 = input_label2
                inputs2, labels2 = inputs2.to(device), labels2.to(device)

                optimizer.zero_grad()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():    
                    output2 = model.forward(inputs2)
                    loss2 = criterion(output2,labels2)
                
                
                    # Calculate accuracy
                    ps = torch.exp(output2)
                    equality = (labels2.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()
                    
                    #Printing losses and accuracy
                    print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {loss2/len(validationloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validationloader):.3f}")

            

# TODO: Do validation on the test set
correct_all = 0
total_all = 0
i = 1
#Similar to the validation set
with torch.no_grad ():
    for input_label_test in testloader:
        inputs_test, labels_test = input_label_test
        inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
        output_test = model (inputs_test)
        _, predicted = torch.max (output_test.data,1)
        # Calculating the accuracy of test, and finally average of all tests.
        total = labels_test.size (0)
        correct = (predicted == labels_test).sum().item()
        print('Test image accuracy %d: %d %%' % (i, 100 * correct / total))  
        i = i+1
        total_all += labels_test.size (0)
        correct_all += (predicted == labels_test).sum().item()
        

print('Average Accuracy on test images: %d %%' % (100 * correct_all / total_all))

model.to (device) 

model.class_to_idx = train_data.class_to_idx

# Save the checkpoint 
checkpoint = {'classifier': model,
              'state_dict': model.state_dict(),
              'optimizer' : optimizer.state_dict(),
              'class_to_idx' : model.class_to_idx,
              'epochs': epochs
              
}

if args.save_dir:
        save_dir = args.save_dir
else:
        save_dir = './checkpoint.pth'

torch.save(checkpoint, save_dir)
    

    
    
    
    
    