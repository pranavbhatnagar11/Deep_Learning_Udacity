## Image Classifier 

### Part 1: Development Notebook

Training data augmentation: Torchvision transforms are used to augment the training data with random scaling, rotations, mirroring, and/or cropping

Data normalization: The training, validation, and testing data is appropriately cropped and normalized

Data batching: The data for each set is loaded with torchvision's DataLoader

Data loading: The data for each set (train, validation, test) is loaded with torchvision's ImageFolder

Pretrained Network: A pretrained network such as VGG16 is loaded from torchvision.models and the parameters are frozen

Feedforward Classifier: A new feedforward network is defined for use as a classifier using the features as input

Training the network: The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature network are left static

Testing Accuracy: The network's accuracy is measured on the test data

Validation Loss and Accuracy: During training, the validation loss and accuracy are displayed

Loading checkpoints: There is a function that successfully loads a checkpoint and rebuilds the model

Saving the model: The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary

Image Processing: The process_image function successfully converts a PIL image into an object that can be used as input to a trained model

Class Prediction: The predict function successfully takes the path to an image and a checkpoint, then returns the top K most probably classes for that image

Sanity Checking with matplotlib: A matplotlib figure is created displaying an image and its associated top 5 most probable classes with actual flower names

### Part 2 - Command Line Application

train.py - Training a network, Training validation log, Model architecture, Model hyperparameters, Training with GPU
predict.py - Predicting classes, Top K classes, Displaying class names, Predicting with GPU

