# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 10:49:49 2019

@author: wwech
"""

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


traiing_path = r'WeaponS/train'
valid_path = r'weapon_dataset_alaap/valid'
test_path = r'weapon_dataset_alaap\test'

data_gen = ImageDataGenerator(rotation_range=30,
                              rescale=1./255,
                              validation_split = 0.20,  
                              shear_range=0.2,
                              zoom_range=0.3,
                              width_shift_range=0.2,
                              height_shift_range=0.2, 
                              horizontal_flip=True)

"""
change the validation split parameter and crete new valid genertor if needed.
"""

test_gen = ImageDataGenerator(rescale=1./255)


Classes = ['Safe','Unsafe']

train_batches = data_gen.flow_from_directory(traiing_path,target_size = (128,128),
                                             classes =Classes,class_mode = "categorical",
                                             subset='training',
                                             shuffle = True,
                                             batch_size = 64)

validation_batch = data_gen.flow_from_directory(traiing_path,target_size = (128,128),
                                             classes = Classes,class_mode = "categorical",
                                             subset='validation',
                                             shuffle = True,
                                             batch_size = 64)

tet_batch = test_gen.flow_from_directory(test_path,target_size = (128,128),
                                             classes = Classes,
                                             class_mode = "categorical",
                                             shuffle = False,
                                             batch_size = 64)

tet_batch.image_shape

model = Sequential()

model.add(ZeroPadding2D((1,1), input_shape = train_batches.image_shape))
model.add(Conv2D(64, kernel_size = 4, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.35))

model.add(Conv2D(128, kernel_size = 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(256, kernel_size = 3, activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.35))

model.add(Conv2D(256, kernel_size = 2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


STEP_SIZE_TRAIN = train_batches.n//train_batches.batch_size # define train and valid batch size
STEP_SIZE_VALID = validation_batch.n//validation_batch.batch_size

history = History()

optimizer = Adam(lr = .001,decay = 1e-5)

model.compile(optimizer,loss="binary_crossentropy",metrics=["accuracy"])

# define callbakcs like early stopping and Learning rate decay on plateaus.
callbacks = [history, 
             EarlyStopping(monitor='val_loss', patience=10, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, cooldown=0, min_lr=1e-7, verbose=1)]

# fit the model
history = model.fit_generator(generator=train_batches,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_batch,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=20, callbacks=callbacks , verbose = 1)

model.save('cnn_weapon22.h5')

from tensorflow.keras.models import load_model

# load model
model = load_model('cnn_weapon.h5')

STEP_SIZE_TEST=tet_batch.n//tet_batch.batch_size
try:
    pred=model.evaluate_generator(tet_batch,STEP_SIZE_TEST)
except Exception as e:
    print(e)
pred
model.metrics_names

tet_batch.reset() # Necessary to force it to start from beginning
Y_pred = model.predict_generator(tet_batch)
y_pred = np.argmax(Y_pred, axis=-1)
sum(y_pred==tet_batch.classes)/10000

y_pred

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
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





