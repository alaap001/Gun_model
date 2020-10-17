# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib
from focal_loss_pytorch.focalloss import FocalLoss
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import time
import os
import copy

# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

BATCH_SIZE = 64
data_dir = '../final_data2/final_data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=1) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print("dataset_sizes: ", dataset_sizes)
class_names = image_datasets['train'].classes
print("class_names: ", class_names)
# exit(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
# out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])
# exit(0)

def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("{} of {} ; loss: {}".format(i, dataset_sizes[phase] // BATCH_SIZE, loss))
            # print("{} of {}".format(i, dataset_sizes[phase] // 256))
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '../weights/focal_loss/best_epoch'+str(epoch)+'.pth') 
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# def visualize_model(model, num_images=6):
#     was_training = model.training
#     model.eval()
#     images_so_far = 0
#     fig = plt.figure()
#
#     with torch.no_grad():
#         for i, (inputs, labels) in enumerate(dataloaders['val']):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 ax = plt.subplot(num_images//2, 2, images_so_far)
#                 ax.axis('off')
#                 ax.set_title('predicted: {}'.format(class_names[preds[j]]))
#                 imshow(inputs.cpu().data[j])
#
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return
#         model.train(mode=was_training)




model_ft = models.resnet18(pretrained=True)


for i, param in enumerate(model_ft.named_parameters()):
    param[1].requires_grad = False
    #print(i)
    if i >= 30: #"layer3.0.conv1.weight":
        param[1].requires_grad = True
    # print(param[0], param[1].requires_grad)

num_ftrs = model_ft.fc.in_features
print("num_ftrs: ", num_ftrs)
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).


###alternate model defining but not working properly
class FineTuneModel(nn.Module):
    def __init__(self, original_model, num_filters, num_classes):
        super().__init__()
        self.body = nn.Sequential(*(list(original_model.children())[:-1]))
        self.head = nn.Sequential(
                        nn.BatchNorm1d(num_filters),
                        nn.Dropout(0.5),
                        nn.Linear(num_filters, 512),
                        nn.ReLU(True),
                        nn.BatchNorm1d(num_filters),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes),
                    )

    def forward(self, x):
        f = self.body(x)
        # f = torch.flatten(f)
        f = f.view(f.size(0), -1)
        y = self.head(f)
        return y


model_ft = FineTuneModel(model_ft, num_ftrs, 2)
weights = torch.load('../weights/transfer_weights/best_epoch19.pth')
model_ft.load_state_dict(weights)
# print(model_ft)
# exit(0)

# ### different method for defining model
# model_ft = nn.Sequential(*(list(model_ft.children())[:-1]))
# model_ft.head = nn.Sequential(
#     # torch.flatten(),
#     nn.BatchNorm1d(num_ftrs),
#     nn.Dropout(0.25),
#     nn.Linear(num_ftrs, 512),
#     nn.ReLU(True),
#     nn.BatchNorm1d(num_ftrs),
#     nn.Dropout(0.25),
#     nn.Linear(512, 2),
#     nn.ReLU(True)
#     )
# print(model_ft)
# exit(0)

# model_ft.head_bn1 = nn.BatchNorm1d(num_ftrs)
# model_ft.head_dp1 = nn.Dropout(p=0.25)
# model_ft.head_fc1 = nn.Linear(num_ftrs, 512)
# # model_ft.head_relu = nn.functional.relu(model_ft.head_fc1, inplace=True)
# model_ft.head_bn2 = nn.BatchNorm1d(512)
# model_ft.head_dp2 = nn.Dropout(p=0.25)
# model_ft.head_fc2 = nn.Linear(512, 2)
# print(model_ft)
# exit(0)

#model_ft.load_state_dict(torch.load("../models/alcohol_ft_resnet_18_v3.pth"))

print("model_ft")
for i, param in enumerate(model_ft.named_parameters()):
    pass
    print(i, param[0], param[1].requires_grad)
    # param.requires_grad = False
# exit(0)
model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# Decay LR by a factor of 0.5 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.5)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)

torch.save(model_ft.state_dict(), "../weights/focal_loss/weapon_tf_ce_v1.pth")
# visualize_model(model_ft)