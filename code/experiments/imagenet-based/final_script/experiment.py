import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, transforms
import os
from torch.optim import lr_scheduler
import mlflow
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    with open(f'{dir_path}/experiments.json', 'r') as file:
        experiment_data = json.load(file)
except json.decoder.JSONDecodeError:
    print('Invalid experiments.json, aborting')
    exit()

for parameters in experiment_data['experiments']:
    # hyperparameters, tag
    try:
        NUM_EPOCHS = parameters['NUM_EPOCHS']
        BATCH_SIZE = parameters['BATCH_SIZE']
        LAYERS_TRAINED = parameters['LAYERS_TRAINED']
        LEARNING_RATE = parameters['LEARNING_RATE']
        MOMENTUM = parameters['MOMENTUM']
        RANDOM_SEED = parameters['RANDOM_SEED']
        BASE_MODEL = parameters['BASE_MODEL']
        TAG = parameters['TAG']
    except (AttributeError, KeyError) as e:
        print(f'Invalid data in experiments.json: {e}')
        exit()

    # do only upper/lower first
    experiment = 'upperlower'
    target_num_of_classes = 2

    # set random seed for both CUDA and CPU with manual seed
    torch.manual_seed(RANDOM_SEED)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(3),
            transforms.GaussianBlur(3),
            transforms.ColorJitter(brightness=[0.95,1.05], contrast=[0.8,1.2]),
            transforms.RandomAffine(degrees=[-5,5], shear=(1,10,1,10), fill=255),
            transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])
        ]),
    }

    data_dir = f'/home/riikoro/fossil_data/tooth_samples/torch_imagefolder_0/{experiment}'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=0) # seed this random
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model setup: load base model, replace last layer
    # syntax depends on base model submodule structure & naming
    if (BASE_MODEL == 'vit_b_16'):
        model = torchvision.models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')
        num_ftrs = model.heads[0].in_features
        model.heads[0] = nn.Linear(num_ftrs, target_num_of_classes)

    elif (BASE_MODEL == 'alexnet'):
        model = torchvision.models.alexnet(weights='AlexNet_Weights.IMAGENET1K_V1')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, target_num_of_classes)

    elif (BASE_MODEL == 'efficientnet_v2_s'):
        model = torchvision.models.efficientnet_v2_s(weights='EfficientNet_V2_S_Weights.IMAGENET1K_V1')
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, target_num_of_classes)

    elif (BASE_MODEL == 'vgg16'):
        model = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, target_num_of_classes)

    elif (BASE_MODEL == 'densenet121'):
        model = torchvision.models.densenet121(weights='DenseNet121_Weights.IMAGENET1K_V1')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, target_num_of_classes)
    
    elif (BASE_MODEL == 'resnet101'):
        model = torchvision.models.resnet101(weights='ResNet101_Weights.IMAGENET1K_V2')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, target_num_of_classes)
    
    else: 
        print('ERROR: unsupported base model, options: vit_b_16, alexnet, efficientnet_v2_s, vgg16, densenet121, resnet101')

    # Freeze layers
    if LAYERS_TRAINED != 'all':
        for param in list(model.parameters())[:-1*(LAYERS_TRAINED)]:
            param.requires_grad = False
    
    # Add softmax to end of net to get valid probabilities as outputs for confidence scores
    model = nn.Sequential(
        model,
        nn.Softmax(1)
    )
    model = model.to(device)

    # loss & lr scheduling
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        train_accuracies = []
        val_accuracies = []
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                if phase == 'val':
                    val_accuracies.append(epoch_acc.item())
                    val_losses.append(epoch_loss)
                if phase == 'train':
                    train_accuracies.append(epoch_acc.item())
                    train_losses.append(epoch_loss)

            print()

        print(f'Final validation accuracy: {val_accuracies[-1]:4f}')

        return model, train_accuracies, val_accuracies, train_losses, val_losses

    model, train_accuracies, val_accuracies, train_losses, val_losses = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

    model_pt_filename = f'{experiment}.pt'
    torch.save(model, model_pt_filename)

    os.environ['MLFLOW_TRACKING_URI'] = 'sqlite:///../../../mlflow.db'
    mlflow.set_experiment(experiment)

    params = {
        'train_size': dataset_sizes['train'],
        'val_size': dataset_sizes['val'],
        'batch_size': BATCH_SIZE,
        'num_epochs': NUM_EPOCHS,
        'base_model_path': type(model),
        'layers_trained': LAYERS_TRAINED,
        'learning_rate': LEARNING_RATE,
        'momentum': MOMENTUM
    }

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log accuracy
        mlflow.log_metric("training accuracy", train_accuracies[-1])
        mlflow.log_metric("validation accuracy", val_accuracies[-1])

        # Log the model
        mlflow.log_artifact(model_pt_filename)

        # Set a tag to remind what this run was for
        # Save accuracies & losses as str for easier viewing in the UI
        mlflow.set_tag("info", TAG)
        mlflow.set_tag("train_accuracies", str(train_accuracies))
        mlflow.set_tag("val_accuracies", str(val_accuracies))
        mlflow.set_tag("train_losses", str(train_losses))
        mlflow.set_tag("val_losses", str(val_losses))

