#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 16:15:27 2020

@author: sshekhar
"""
# data visualisation and manipulation
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline 
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.metrics import homogeneity_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model

# specifically for cnn
from keras.layers import Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
 
import tensorflow as tf
import random as rn
import pickle

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
#import numpy as np  
from tqdm import tqdm
import os                   


X=[]
Z=[]
IMG_SIZE=150

imagePaths = "./images"
classNames = os.listdir(imagePaths)
classNames = [str(x) for x in np.unique(classNames)]

def assign_label(img,flower_type):
    return flower_type

def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))

for name in classNames:
    IMG_DIR=imagePaths+"/"+name
    make_train_data(name, IMG_DIR)
    print(len(X))

le_labels=LabelEncoder()
Y=le_labels.fit_transform(Z)
Y=to_categorical(Y,len(classNames))
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
print("shapes x_train,x_test,y_train,y_test: ", x_train.shape,x_test.shape,y_train.shape,y_test.shape)
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)
# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(len(classNames), activation = "softmax"))
batch_size=128
epochs=100

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params
        
datagen.fit(x_train)
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['acc'])

model.summary()
modelpath="17flowermodel"
if os.path.isfile(modelpath +'/'+'saved_model.pb'):
    model = load_model(modelpath)
    with open(modelpath+'/history', 'rb') as file:
        History=pickle.load(file)
else:
    #Fitting on the Training set and making predcitons on the Validation set
    History = model.fit_generator(datagen.flow(x_train,y_train, 
                                               batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, 
                              steps_per_epoch=x_train.shape[0] // batch_size, shuffle=False, callbacks=[red_lr])
    model.save(modelpath)

    with open(modelpath+'/history', 'wb') as file:
        model_history= History_trained_model(History.history, History.epoch, History.params)
        pickle.dump(model_history, file, pickle.HIGHEST_PROTOCOL)


# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))
# Evaluating the Model Performance

# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), History.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), History.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), History.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), History.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("loss_acc_plot.jpg")

# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)
test_digits=np.argmax(y_test,axis=1)
print("Homoginity score: ", homogeneity_score(pred_digits,test_digits))
# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]
for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Flower : "+str(le_labels.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual Flower : "+str(le_labels.inverse_transform([test_digits[prop_class[count]]])))
#        plt.tight_layout()
        count+=1
plt.savefig("correctly_predicted.jpg")
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Flower : "+str(le_labels.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Flower : "+str(le_labels.inverse_transform([test_digits[mis_class[count]]])))
#        plt.tight_layout()
        count+=1
plt.savefig("incorrectly_predicted.jpg")