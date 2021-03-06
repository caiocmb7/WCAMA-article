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
import pylab as plot
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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization, LeakyReLU, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow
from tensorflow.python.keras import backend
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers

IMAGE_PATH = r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\novaBaseAtualizada\dataset"
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.3
VAL_SIZE = 0.2
CONV_2D_DIM_1 = 16
CONV_2D_DIM_2 = 16
CONV_2D_DIM_3 = 32
CONV_2D_DIM_4 = 64
MAX_POOL_DIM = 2
KERNEL_SIZE = 3
BATCH_SIZE = 64
NO_EPOCHS_1 = 5
NO_EPOCHS_2 = 25
NO_EPOCHS_3 = 175
PATIENCE = 5
VERBOSE = 1

honey_bee_df = pd.read_csv(r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\novaBaseAtualizada\dataset\datasetWCAMA.csv")

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

earlystopper3 = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=VERBOSE)
checkpointer3 = ModelCheckpoint('best_model_3.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

# build a sequential model
model3 = Sequential()
model3.add(InputLayer(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS)))
# 1st conv block
model3.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model3.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model3.add(Dropout(0.4))
# 2nd conv block
model3.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
model3.add(MaxPool2D(pool_size=(2, 2), padding='same'))
model3.add(Dropout(0.35))
# 3rd conv block
model3.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model3.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
model3.add(Dropout(0.35))
# ANN block
model3.add(Flatten())
model3.add(Dense(units=500, activation='relu'))
model3.add(Dense(units=250, activation='relu'))
model3.add(Dropout(0.2))
# output layer
model3.add(Dense(units=y_train.columns.size, activation='softmax'))
model3.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

loss_history = []

for i in range(100):
    loss_history  += [model3.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train)/BATCH_SIZE)]
                       
def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

print(test_accuracy_report(model3))   

plt.rc('legend',fontsize=25) # using a size in points
plt.rc('legend',fontsize='large') # using a named size

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams.update({'font.size': 18})

epich = np.cumsum(np.concatenate(
    [np.linspace(1.5, 2, len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plottest = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]),
             'b-',
             epich, np.concatenate(
        [mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Treinamento', 'Valida????o'])
ax1.set_title('Perda')

plottest = ax2.plot(epich, np.concatenate(
    [mh.history['accuracy'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
        [mh.history['val_accuracy'] for mh in loss_history]),
                 'r-')
ax2.legend(['Treinamento', 'Valida????o'])
ax2.set_title('Acur??cia')
plt.rc('legend',fontsize=25) # using a size in points
plt.rc('legend',fontsize='large') # using a named size
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.show()