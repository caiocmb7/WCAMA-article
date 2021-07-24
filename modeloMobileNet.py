from glob import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
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


np.random.seed(2017)
def imread_size(in_path):
    t_img = imread(in_path)
    return zoom(t_img, [64/t_img.shape[0], 64/t_img.shape[1]]+([1] if len(t_img.shape)==3 else []),
               order = 2)

base_img_dir = os.path.join(r'C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\datasetSeparado', 'input')
all_training_images = glob(os.path.join(base_img_dir, '*', '*.jpg'))
full_df = pd.DataFrame(dict(path = all_training_images))
full_df['category'] = full_df['path'].map(lambda x: os.path.basename(os.path.dirname(x)))
full_df = full_df.query('category != "valid"')
cat_enc = LabelEncoder()
full_df['category_id'] = cat_enc.fit_transform(full_df['category'])
y_labels = to_categorical(np.stack(full_df['category_id'].values,0))
print(y_labels.shape)
full_df['image'] = full_df['path'].map(imread_size)
print(full_df.sample(10))

print(full_df['category'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(np.expand_dims(np.stack(full_df['image'].values,0),-1), 
                                                    y_labels,
                                   random_state = 12345,
                                   train_size = 0.75,
                                   stratify = full_df['category'])
                                
print('X_train Size', X_train.shape)
print('X_test Size', X_test.shape)
print('y_train Size', y_train.shape)
print('y_test Size', y_test.shape)
X_train_monochrome = X_train.mean(axis=3)
X_test_monochrome = X_test.mean(axis=3)
print('X_train_monochrome Size', X_train_monochrome.shape)
print('X_test_monochrome Size', X_test_monochrome.shape)


train_datagen = ImageDataGenerator(
        samplewise_std_normalization = True,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range = 360,
        )

test_datagen = ImageDataGenerator(
        samplewise_std_normalization = True,)


train_gen = train_datagen.flow(X_train_monochrome, y_train, batch_size=32)
test_gen = train_datagen.flow(X_test_monochrome, y_test, batch_size=32)
fig, (ax1, ax2) = plt.subplots(2, 4, figsize = (12, 6))
for c_ax1, c_ax2, (train_img, _), (test_img, _) in zip(ax1, ax2, train_gen, test_gen):
    c_ax1.imshow(train_img[0,:,:,0])
    c_ax1.set_title('Train Image')
    
    c_ax2.imshow(test_img[0,:,:,0])
    c_ax2.set_title('Test Image')


mn_cnn = MobileNet(input_shape = train_img.shape[1:], dropout = 0.25, weights = None, 
                  classes = y_labels.shape[1])
mn_cnn.compile(loss = 'categorical_crossentropy', 
               optimizer = Adam(lr = 1e-4, decay = 1e-6),
               metrics = ['acc'])
loss_history = []
print(mn_cnn.summary())

for i in range(20):
    loss_history += [mn_cnn.fit_generator(train_gen, steps_per_epoch=10,
                         validation_data=test_gen, validation_steps=10)]

test_res = mn_cnn.evaluate(X_test_monochrome, y_test, verbose=0)
print('Loss function: %s, accuracy:' % test_res[0], test_res[1])

epich = np.cumsum(np.concatenate(
    [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
plottest = ax1.plot(epich,
             np.concatenate([mh.history['loss'] for mh in loss_history]),
             'b-',
             epich, np.concatenate(
        [mh.history['val_loss'] for mh in loss_history]), 'r-')
ax1.legend(['Training', 'Validation'])
ax1.set_title('Loss')

plottest = ax2.plot(epich, np.concatenate(
    [mh.history['acc'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
        [mh.history['val_acc'] for mh in loss_history]),
                 'r-')
ax2.legend(['Training', 'Validation'])
ax2.set_title('Accuracy')
plt.show()