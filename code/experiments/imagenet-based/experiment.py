# TODO: create standalone script that takes experiment hyperparameters as inputs
# and logs relevant output to logfile & mlflow

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from torch.optim import lr_scheduler
import mlflow
import json

# does not find the file
parameters = json.loads('./experiments.json')
print(parameters)

# hyperparameters
# NUM_EPOCHS = 25
# TODO: update to 50 in final experiments
NUM_EPOCHS = 1 # new, converged in this time
# TODO: update to 64 once larger dataset is available
BATCH_SIZE = 4
LAYERS_TRAINED = 2
LEARNING_RATE = 0.001
MOMENTUM = 0.9
RANDOM_SEED = 15

# use balanced torch imagefolder dataset
experiment = 'upperlower'
target_num_of_classes = 3

# add seconds to run training to mlflow!
runtime = 0

# set random seed for both CUDA and CPU with manual seed
torch.manual_seed(RANDOM_SEED)

# data_transforms = {
#     'train': transforms.Compose([
#         transforms.Grayscale(3),
#         transforms.GaussianBlur(3),
#         transforms.ColorJitter(brightness=[0.95,1.05], contrast=[0.8,1.2]),
#         transforms.RandomAffine(degrees=[-5,5], shear=(1,10,1,10), fill=255),
#         transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
#     ]),
#     'val': transforms.Compose([
#         transforms.Grayscale(3),
#         transforms.Resize((224,224)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
#     ]),
# }


# data_dir = f'/home/riikoro/fossil_data/tooth_samples/torch_imagefolder_0/{experiment}'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                           data_transforms[x])
#                   for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
#                                              shuffle=True, num_workers=0) # seed this random
#               for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # model = torchvision.models.vit_b_16(weights='DEFAULT')
# # model = torchvision.models.alexnet(weights='DEFAULT')
# model = torchvision.models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
# for param in list(model.parameters())[:-1*(LAYERS_TRAINED+1)]:
#     param.requires_grad = False

# # Parameters of newly constructed modules have requires_grad=True by default
# # alexnet
# # num_ftrs = model.classifier[6].in_features
# # model.classifier[6] = nn.Linear(num_ftrs, target_num_of_classes)

# # vit
# # num_ftrs = model.heads[0].in_features
# # model.heads[0] = nn.Linear(num_ftrs, target_num_of_classes)

# # efficientnet
# num_ftrs = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_ftrs, target_num_of_classes)

# model = model.to(device)
# criterion = nn.CrossEntropyLoss()


# # vit 
# # optimizer = optim.SGD(model.heads[0].parameters(), lr=0.001, momentum=0.9)
# # alexnet
# # optimizer = optim.SGD(model.classifier[6].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
# # efficientnet
# optimizer = optim.SGD(model.classifier[1].parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# # Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
#     best_train_acc = 0.0
#     best_test_acc = 0.0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             running_loss = 0.0
#             running_corrects = 0

#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             if phase == 'train':
#                 scheduler.step()
#             epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
#             if phase == 'val':
#                 best_test_acc = epoch_acc.item()
#             if phase == 'train':
#                 best_train_acc = epoch_acc.item()

#         print()

#     print(f'Best val Acc: {best_test_acc:4f}')

#     return model, best_train_acc, best_test_acc

# model, train_acc, val_acc = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# model_pt_filename = f'{experiment}.pt'
# torch.save(model, model_pt_filename)

# os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///../../mlflow.db'
# mlflow.set_experiment(experiment)

# params = {
#     'data_v': [3,4,5],
#     'train_size': dataset_sizes['train'],
#     'test_size': dataset_sizes['val'],
#     'batch_size': BATCH_SIZE,
#     'num_epochs': NUM_EPOCHS,
#     'base_model_path': type(model),
#     'layers_trained': LAYERS_TRAINED,
#     'learning_rate': LEARNING_RATE,
#     'momentum': MOMENTUM
# }

# # Start an MLflow run
# with mlflow.start_run():
#     # Log the hyperparameters
#     mlflow.log_params(params)

#     # Log the loss metric
#     mlflow.log_metric("training accuracy", train_acc)
#     mlflow.log_metric("test accuracy", val_acc)

#     mlflow.log_artifact(model_pt_filename)

#     # Set a tag that we can use to remind ourselves what this run was for
#     mlflow.set_tag("info", "EfficientNet with new dataset, one epoch to test script")