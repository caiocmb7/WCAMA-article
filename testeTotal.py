import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
import imageio
import skimage
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import scipy
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow
from tensorflow.python.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import layers

IMAGE_PATH = r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\todasImagens"
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 150
IMAGE_CHANNELS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.3
VAL_SIZE = 0.3
CONV_2D_DIM_1 = 16
CONV_2D_DIM_2 = 16
CONV_2D_DIM_3 = 32
CONV_2D_DIM_4 = 64
MAX_POOL_DIM = 2
KERNEL_SIZE = 3
BATCH_SIZE = 32
NO_EPOCHS_1 = 5
NO_EPOCHS_2 = 10
NO_EPOCHS_3 = 15
PATIENCE = 5
VERBOSE = 1

honey_bee_df = pd.read_csv(r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\csvDataset.csv")

train_df, test_df = train_test_split(honey_bee_df, test_size=TEST_SIZE, random_state=RANDOM_STATE, 
                                     stratify=honey_bee_df['Classe'])

train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=train_df['Classe'])

def read_image(file_name):
    image = skimage.io.imread(IMAGE_PATH + "\\" + file_name)
    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), mode='reflect')
    return image[:,:,:IMAGE_CHANNELS]

def categories_encoder(dataset, var='Classe'):
    X = np.stack(dataset['ID'].apply(read_image))
    y = pd.get_dummies(dataset[var], drop_first=False)
    return X, y

def categories_encoder_test(dataset):
    X = np.stack(dataset['ID'].apply(read_image))
    return X

X_train, y_train = categories_encoder(train_df)
X_val, y_val = categories_encoder(val_df)
X_test, y_test = categories_encoder(test_df)

image_generator = ImageDataGenerator(
        rescale = 1./255,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=180,
        zoom_range = 0.1, 
        width_shift_range=0.1,
        height_shift_range=0.1, 
        horizontal_flip=True,
        vertical_flip=True)
image_generator.fit(X_train)

model1=Sequential()
model1.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))
model1.add(MaxPool2D(MAX_POOL_DIM))
model1.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model1.add(Flatten())
model1.add(Dense(y_train.columns.size, activation='softmax'))
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_model1  = model1.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=NO_EPOCHS_1,
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train)/BATCH_SIZE)

print(model1.summary())

model2=Sequential()
model2.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', padding='same'))
model2.add(MaxPool2D(MAX_POOL_DIM))
model2.add(Dropout(0.4))
model2.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', padding='same'))
model2.add(Dropout(0.4))
model2.add(Flatten())
model2.add(Dense(y_train.columns.size, activation='softmax'))
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model2.summary())

train_model2  = model2.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=NO_EPOCHS_2,
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train)/BATCH_SIZE)

annealer3 = LearningRateScheduler(lambda x: 1e-3 * 0.995 ** (x+NO_EPOCHS_3))
earlystopper3 = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)
checkpointer3 = ModelCheckpoint('best_model_3.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

model3=Sequential()
model3.add(Conv2D(CONV_2D_DIM_1, kernel_size=KERNEL_SIZE, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS), activation='relu', kernel_regularizer=regularizers.l2(l=0.001), padding='same'))
model3.add(MaxPool2D(MAX_POOL_DIM))
model3.add(Dropout(0.4))
model3.add(Conv2D(CONV_2D_DIM_2, kernel_size=KERNEL_SIZE, activation='relu', kernel_regularizer=regularizers.l2(l=0.001), padding='same'))
model3.add(Dropout(0.4))
model3.add(Flatten())
model3.add(Dense(y_train.columns.size, activation='softmax'))
model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model3.summary())     

train_model3  = model3.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        epochs=NO_EPOCHS_3,
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train)/BATCH_SIZE,
                        callbacks=[earlystopper3, checkpointer3, annealer3])
                 
def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

print(test_accuracy_report(model1))  
print(test_accuracy_report(model2)) 
print(test_accuracy_report(model3))
    
#UFC_test = categories_encoder_test(honey_bee_df)

#def bee_prediction(model):
    #prediction = model.predict(UFC_test)
    #id_predicted = np.argmax(prediction, axis=1)
    #print(id_predicted)
    #print("############\n")
    #print (honey_bee_df['Classe'].unique()[id_predicted])
    #print("############\n")

#print(bee_prediction(model3))  


#prediction = model3.predict(UFC_test)
#id_predicted = np.argmax(prediction, axis=1)
#honey_bee_df_table = pd.read_csv(r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\csvDataset.csv", usecols=[0,1,2])
#matriz_test = honey_bee_df['Classe'].unique()[id_predicted]
#honey_bee_df_table.insert(loc=2, column= 'Previsao', value=matriz_test)
#honey_bee_df_matriz_test = pd.DataFrame(data=honey_bee_df_table)
#honey_bee_df_matriz_test.to_csv('previsaoTeste.csv')


