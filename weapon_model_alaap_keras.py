# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:33:23 2019

@author: wwech
"""

#from DenseNet import DenseNet
#
#dense_block_size = 50
#layers_in_block = 4
#growth_rate = 12
#classes = 2
#i_shape = (256,256,3)
#model = DenseNet().dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block,i_shape)
#model.summary()


# General Libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle 
import cv2
import matplotlib.pyplot as plt

# Keras libs
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # to generate more training data by augmentation
from tensorflow.keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Activation, \
                      BatchNormalization ,GlobalAveragePooling2D, concatenate, AveragePooling2D, Input
                      
from tensorflow.keras.models import Sequential , Model 
from tensorflow.keras.optimizers import Adam, SGD #For Optimizing the Neural Network
from sklearn.metrics import confusion_matrix # confusion matrix to carry out error analysis
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History # import callback functions for model
from tensorflow.keras.applications import DenseNet121
from tensorflow.python.keras.applications.resnet50 import ResNet50, preprocess_input


traiing_path = r'weapon_dataset_alaap/train'
valid_path = r'weapon_dataset_alaap/valid'
FREEZE_LAYERS = 20  # freeze the first this many layers for training

data_gen = ImageDataGenerator(rotation_range=30,
                              rescale=1./255,
                              validation_split = 0.1,
                              shear_range=0.2,
                              zoom_range=0.3,
                              width_shift_range=0.2,
                              height_shift_range=0.2, 
                              horizontal_flip=True)

Classes = ['Safe','Unsafe']

train_batches = data_gen.flow_from_directory(traiing_path,target_size = (224,224),
                                             classes =Classes,class_mode = "categorical",
                                             subset='training',
                                             shuffle = True,
                                             batch_size = 32)

validation_batch = data_gen.flow_from_directory(traiing_path,target_size = (224,224),
                                             classes = Classes,class_mode = "categorical",
                                             subset='validation',
                                             shuffle = True,
                                             batch_size = 32)

train_batches.image_shape

#testgen = ImageDataGenerator(rescale = 1./255)
#test_batch = testgen.flow_from_directory(path_test,classes=['999'],target_size = (128,128))


# show class indices
print('****************')
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')


model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=train_batches.image_shape, pooling='max')

model.layers

x = model.output
x
x = Flatten()(x)
x = Dropout(0.4)(x)

output_layer = ResNet50(2, activation='softmax', name='softmax')(x)

output_layer

net_final = Model(inputs=model.input, outputs=output_layer)

for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True


STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size # define train and valid batch size
STEP_SIZE_VALID = validation_batch.n//validation_batch.batch_size

history = History()

optimizer = Adam(lr = .001,decay = 1e-5)

net_final.compile(optimizer,loss="binary_crossentropy",metrics=["accuracy"])

# define callbakcs like early stopping and Learning rate decay on plateaus.
callbacks = [history, 
             EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=0, min_lr=1e-7, verbose=1)]

# fit the model
history = net_final.fit_generator(generator=train_batches,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_batch,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25, callbacks=callbacks , verbose = 1)

net_final.save('Resnet_alcohol_filter.h5')

# Plot training & validation accuracy values
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()


