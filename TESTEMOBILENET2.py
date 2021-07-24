from glob import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import imageio
import skimage
import skimage.io
import skimage.transform
from skimage.io import imread
from scipy.ndimage import zoom
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras import backend as k
import cv2

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3

np.random.seed(2017)

honey_bee_df = pd.read_csv(r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\novaBaseAtualizada\dataset\datasetWCAMA.csv")
IMAGE_PATH = r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\novaBaseAtualizada\dataset"

def read_image(file_name):
    image = skimage.io.imread(IMAGE_PATH + "\\" + file_name)
    image = skimage.transform.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), mode='reflect')
    return image[:,:,:IMAGE_CHANNELS]

cat_enc = LabelEncoder()
honey_bee_df['category_id'] = cat_enc.fit_transform(honey_bee_df['Classe'])
y_labels = to_categorical(np.stack(honey_bee_df['category_id'].values,0))

train_df, test_df = train_test_split(honey_bee_df, test_size=0.75, random_state=12345, 
                                     stratify=honey_bee_df['Classe'])

def categories_encoder(dataset, var='Classe'):
    X = np.stack(dataset['ID'].apply(read_image))
    y = pd.get_dummies(dataset[var], drop_first=False)
    return X, y

X_train, y_train = categories_encoder(train_df)
X_test, y_test = categories_encoder(test_df)

print('Training Size', X_train.shape)

train_datagen = ImageDataGenerator(
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
train_datagen.fit(X_train)  

test_datagen = ImageDataGenerator(
        samplewise_std_normalization = True)

train_gen = train_datagen.flow(X_train, y_train, batch_size=32)
test_gen = train_datagen.flow(X_test, y_test, batch_size=32)
fig, (ax1, ax2) = plt.subplots(2, 4, figsize = (12, 6))
for c_ax1, c_ax2, (train_img, _), (test_img, _) in zip(ax1, ax2, train_gen, test_gen):
    c_ax1.imshow(train_img[0,:,:,0])
    c_ax1.set_title('Train Image')
    
    c_ax2.imshow(test_img[0,:,:,0])
    c_ax2.set_title('Test Image')

mn_cnn = MobileNet(input_shape = train_img.shape[1:], dropout = 0.25, weights = None, 
                  classes = y_labels.shape[1])
mn_cnn.compile(loss = 'categorical_crossentropy', 
               optimizer = 'adam',
               metrics = ['acc'])

print(mn_cnn.summary())

loss_history = []

for i in range(149):
    loss_history += [mn_cnn.fit_generator(train_gen, steps_per_epoch=10,
                         validation_data=test_gen, validation_steps=10)]

def test_accuracy_report(model):
    predicted = model.predict(X_test)
    test_predicted = np.argmax(predicted, axis=1)
    test_truth = np.argmax(y_test.values, axis=1)
    print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 
    test_res = model.evaluate(X_test, y_test.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

print(test_accuracy_report(mn_cnn)) 

plt.rc('legend',fontsize=25) # using a size in points
plt.rc('legend',fontsize='large') # using a named size

plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams.update({'font.size': 18})

epich = np.cumsum(np.concatenate(
    [np.linspace(1.5, 1.5, len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
plottest = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]),
             'b-',
             epich, np.concatenate(
        [mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Treinamento', 'Validação'])
ax1.set_title('Perda')

plottest = ax2.plot(epich, np.concatenate(
    [mh.history['acc'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
        [mh.history['val_acc'] for mh in loss_history]),
                 'r-')
ax2.legend(['Treinamento', 'Validação'])
ax2.set_title('Acurácia')
plt.rc('legend',fontsize=25) # using a size in points
plt.rc('legend',fontsize='large') # using a named size
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.show()
