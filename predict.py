# Imports here
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
import pandas as pd
import argparse

parser = argparse.ArgumentParser (description = 'predict.py')

parser.add_argument ('--checkpoint', help = 'Loading Model from checkout', type = str, default='checkpoint.pth')
parser.add_argument ('--path_to_image', help = 'Image Directory', type = str, default="flowers/test/1/image_06764.jpg")
parser.add_argument ('--topk', help = 'Top K probabilities', type = int, default = 5)
parser.add_argument ('--GPU', help = "Using GPU", type = str, default = 'gpu')
parser.add_argument('--category_names', help = 'Change to category names',type = str)

args = parser.parse_args ()

if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'
if args.checkpoint:
    checkpoint = args.checkpoint
if args.path_to_image:
    path_to_image = args.path_to_image
if args.topk:
    topk = args.topk

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass


checkpoint = torch.load('checkpoint.pth')
model = checkpoint['classifier']
#print(model)

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

model = load_checkpoint('checkpoint.pth')
#print(model)

def process_image(path_to_image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        
    '''
    # Process a PIL image for use in a PyTorch model
    # Will use a similar technique as above using transform.Compose for resizing, cropping and normalizing.
    pil_image = Image.open(path_to_image)
   
    transformation_pil_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = transformation_pil_image(pil_image)
    
    return img

data_dir = './flowers/'
img = (data_dir + '/test' + '/1/' + 'image_06764.jpg')
img = process_image(img)
#print(img.shape)



def predict(path_to_image, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    image_data = process_image(path_to_image)
    image_data = image_data.unsqueeze_(0)
    image_data = image_data.to(device)
    
    model.eval()
    with torch.no_grad ():
        output = model.forward (image_data)
    predictions = torch.exp (output)
    
    # Identify top predictions and top labels
    probs, classes = predictions.topk(topk)
    return probs, classes

# Implement the code to predict the class from an image file
image_path = (data_dir + '/test' + '/1/' + 'image_06764.jpg')
probs, classes = predict(image_path, model, topk)
predict(path_to_image, model, topk)

#converting into numpy arrays
probs = probs.to(device)
classes = classes.to(device)
probs_array = np.array(probs.detach().numpy().tolist())
classes_array = np.array(classes.detach().numpy().tolist())
probs_array = probs_array.reshape(topk)
classes_array = classes_array.reshape(topk)
#creating a DF with flower class and preditions 
class_index = pd.Series(model.class_to_idx)
class_flower = pd.Series(cat_to_name)
df = pd.DataFrame({'classes': class_index,'flower':class_flower})
df = df.set_index('classes')
df = df.iloc[classes_array]
df['predictions'] = probs_array
print(df)
