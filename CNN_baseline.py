#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import os
import random
import gc #garbage collector for cleaning deleted data from memory


# In[2]:


train_dir = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/Small_train_set/'
test_dir = 'C:/Users/eitn35/Documents/EITN35/video_files/frames/Small_test_set/'

#rain_person = [train_dir+'{}'.format(i) for i in os.listdir(train_dir) if 'persons_1' in i]  #get person images
#rain_dogs = [train_dir+'{}'.format(i) for i in os.listdir(train_dir) if 'dogs_1' in i]  #get dog images
#rain_bikes = [train_dir+'{}'.format(i) for i in os.listdir(train_dir) if 'bikes_1' in i]  #get bike images
#rain_empty = [train_dir+'{}'.format(i) for i in os.listdir(train_dir) if ('persons_0'and 'dogs_0' and 'bikes_0')in i]  #get bike images


test_imgs = [test_dir+'{}'.format(i) for i in os.listdir(test_dir)] #get test images


#train_imgs = train_person + train_dogs + train_bikes + train_empty   # slice the dataset and use 3 persons
train_imgs = [train_dir+'{}'.format(i) for i in os.listdir(train_dir)] #get test images
random.shuffle(train_imgs)  # shuffle it randomly
random.shuffle(test_imgs)


gc.collect()   #collect garbage to save memory


# In[3]:


import matplotlib.image as mpimg
for ima in train_imgs[0:4]:
     img=mpimg.imread(ima)
     imgplot = plt.imshow(img)
     plt.show()


# In[4]:


#Lets declare our image dimensions
#we are using coloured images. 
nrows = 640
ncolumns = 360
channels = 3  #change to 1 if you want to use grayscale image

#A function to read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Returns two arrays: 
        X is an array of resized images
        y is an array of labels
    """
    X = [] # images
    y = []# labels
    i = 0
    for image in list_of_images:
        #ändra här mellan COLOR och GRAYSCALE beroende på antal channels
        X.append(cv2.resize(cv2.imread(image,cv2.IMREAD_COLOR), (nrows,ncolumns), interpolation=cv2.INTER_CUBIC))  #Read the image
        #get the labels
        if 'persons_1' in image:
            y.append(1)
        elif 'dogs_1' in image:
            y.append(2)
        elif 'bikes_1' in image:
            y.append(3)
        else:
            y.append(0)
        i += 1
    return X, y


# In[5]:


class_names = ['empty', 'person', 'dogs', 'bikes']


# In[6]:

print('reading dataset...')
X, y = read_and_process_image(train_imgs)
X_test, y_test = read_and_process_image(test_imgs)


# In[7]:


y[0]


# In[8]:


#Lets view some of the pics
plt.figure(figsize=(20,10))
columns = 4
for i in range(columns):
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.imshow(X[i])


# In[9]:


import seaborn as sns

gc.collect()

#Convert list to numpy array
X = np.array(X)
y = np.array(y)
X_test = np.array(X_test)
y_test = np.array(y_test)
#Lets plot the label to be sure we just have two class
#sns.countplot(y)
#plt.title('Labels for Cats and Dogs')


# In[10]:


print("Shape of train images is:", X.shape)
print("Shape of labels is:", y.shape)


# In[11]:


#Lets split the data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=2)

print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)


# In[12]:


#clear memory
#del X
#del y
gc.collect()

#get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)

#We will use a batch size of 32. Note: batch size should be a factor of 2.***4,8,16,32,64...***
batch_size = 32


# In[13]:



#from keras import models
#from keras import optimizers
from tensorflow.keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(ncolumns, nrows, 3))) #input ska var (150, 150, 3)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.4))  #Dropout for regularization
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(4))  #Sigmoid function at the end because we have just two classes


# In[14]:


#Lets see our model
model.summary()


# In[15]:


#We'll use the RMSprop optimizer with a learning rate of 0.0001
#We'll use binary_crossentropy loss because its a binary classification
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])


# In[16]:


#Lets create the augmentation configuration
#This helps prevent overfitting, since we are using a small dataset
#train_datagen = ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
 #                                   rotation_range=40,
  #                                  width_shift_range=0.2,
   #                                 height_shift_range=0.2,
    #                                shear_range=0.2,
     #                               zoom_range=0.2,
      #                              horizontal_flip=True,)

#val_datagen = ImageDataGenerator(rescale=1./255)  #We do not augment validation data. we only perform rescale


# In[17]:



#Create the image generators
#train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
#val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


# In[18]:



#The training part
#We train for 64 epochs with about 100 steps per epoch
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))


# In[ ]:


#Save the model
model.save_weights('model_cat&dog1_weights.h5')
model.save('model_cat&dog1_keras.h5')


# In[ ]:


#lets plot the train and val curve
#get the details form the history object
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
#plt.plot(epochs, acc, 'b', label='Training accurarcy')
#plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
#plt.title('Training and Validation accurarcy')
#plt.legend()

#plt.figure()

#Train and validation loss
#plt.plot(epochs, loss, 'b', label='Training loss')
#plt.plot(epochs, val_loss, 'r', label='Validation loss')
#plt.title('Training and Validation loss')
#plt.legend()

#plt.show()


# In[ ]:


probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_test)





def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(4))
    plt.yticks([])
    thisplot = plt.bar(range(4), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y_test, X_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()


# In[ ]:


probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(X_train)

predictions[0]

np.argmax(predictions[0])

y_train[0]

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y_train, X_train)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y_train)
plt.tight_layout()
plt.show()



