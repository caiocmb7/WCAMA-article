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

IMAGE_PATH = r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\todasImagens"
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNELS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1
CONV_2D_DIM_1 = 16
CONV_2D_DIM_2 = 16
CONV_2D_DIM_3 = 32
CONV_2D_DIM_4 = 64
MAX_POOL_DIM = 2
KERNEL_SIZE = 3
BATCH_SIZE = 64
NO_EPOCHS_1 = 5
NO_EPOCHS_2 = 25
NO_EPOCHS_3 = 400
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

#critério de parada antecipada:
earlystopper3 = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=VERBOSE)

#Critério para salvar dados do modelo treinado
checkpointer3 = ModelCheckpoint('best_model_3.h5',
                                monitor='val_accuracy',
                                verbose=VERBOSE,
                                save_best_only=True,
                                save_weights_only=True)

quantidades1 = [8, 16, 32, 64]
quantidades2 = [8, 16, 32, 64]
quantidades3 = [8, 16, 32, 64]
larguras1 = [3, 5, 7, 9]
larguras2 = [3, 5, 7, 9]
larguras3 = [3, 5, 7, 9]

for n_filtros1 in quantidades3:
    for n_filtros2 in quantidades3:
        for n_filtros3 in quantidades3:
            for mascara1 in larguras1:
                for mascara2 in larguras2:
                    for mascara3 in larguras3:
                    #inicio laço for
                        cont = 1
                        # inicio do modelo
                        model3 = Sequential()
                        model3.add(InputLayer(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT,IMAGE_CHANNELS)))
                        # camada 1
                        model3.add(Conv2D(n_filtros1, (mascara1, mascara1), activation='relu', padding='same'))
                        model3.add(MaxPool2D(pool_size=(2, 2), padding='same'))
                        model3.add(Dropout(0.4))
                        # camada 2
                        model3.add(Conv2D(n_filtros2, (mascara2, mascara2), activation='relu', padding='same'))
                        model3.add(MaxPool2D(pool_size=(2, 2), padding='same'))
                        model3.add(Dropout(0.35))
                        # camada 3
                        model3.add(Conv2D(n_filtros3, (mascara3, mascara3), activation='relu', padding='same'))
                        model3.add(MaxPool2D(pool_size=(2, 2), padding='valid'))
                        model3.add(Dropout(0.35))
                        # camada hidden
                        model3.add(Flatten())
                        model3.add(Dense(units=500, activation='relu'))
                        model3.add(Dense(units=250, activation='relu'))
                        model3.add(Dropout(0.2))
                        # camada output
                        model3.add(Dense(units=y_train.columns.size, activation='softmax'))
                        model3.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

                        train_model3  = model3.fit_generator(image_generator.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                                epochs=NO_EPOCHS_3,
                                                validation_data=(X_val, y_val),
                                                steps_per_epoch=len(X_train)/BATCH_SIZE)

                        def test_accuracy_report(model):
                            predicted = model.predict(X_test)
                            test_predicted = np.argmax(predicted, axis=1)
                            test_truth = np.argmax(y_test.values, axis=1)
                            print(metrics.classification_report(test_truth, test_predicted, target_names=y_test.columns)) 
                            test_res = model.evaluate(X_test, y_test.values, verbose=0)
                            print('Loss function: %s, accuracy:' % test_res[0], test_res[1])
                            

                        print(model3.summary())
                        print(test_accuracy_report(model3))
                        
                        acc = train_model3.history['accuracy']      
                        loss = train_model3.history['loss']
                        val_loss = train_model3.history['val_loss']
                        epochs = range(len(acc))
                        plt.plot(epochs, acc, 'r', label='Training accuracy')
                        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
                        plt.plot(epochs, loss, 'g', label='Training loss')
                        plt.plot(epochs, val_loss, 'y', label='Validation loss')
                        plt.title('Data test')
                        plt.legend(loc=0)
                        path =  "C:\\Users\\Caio\\Desktop\\smartbee\\imagens\\baseParaWCAMA\\dataset\\graficos\\" + "grafico" + str(cont) + ".png"
                        plt.savefig(path)

                        test_res = model3.evaluate(X_test, y_test.values, verbose=0)

                        with open(r'C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\graficos\resultados.csv', 'a') as arquivo_csv:
                            colunas = ['Loss', 'Accuracy']
                            escrever = csv.DictWriter(arquivo_csv, fieldnames=colunas, delimiter=',', lineterminator='\n')
                            escrever.writeheader()
                            escrever.writerow({'Loss': test_res[0], 'Accuracy': test_res[1]})
                        
                        cont = cont + 1


                    #fim laço for


# gerar os dados em csv

#prediction = model3.predict(UFC_test)
#id_predicted = np.argmax(prediction, axis=1)
#honey_bee_df_table = pd.read_csv(r"C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\csvDataset.csv", usecols=[0,1,2])
#matriz_test = honey_bee_df['Classe'].unique()[id_predicted]
#honey_bee_df_table.insert(loc=2, column= 'Previsao', value=matriz_test)
#honey_bee_df_matriz_test = pd.DataFrame(data=honey_bee_df_table)
#honey_bee_df_matriz_test.to_csv('previsaoTeste8.csv')

# plotagem do gráfico

#val_acc = train_model3.history['val_accuracy']
#loss = train_model3.history['loss']
#val_loss = train_model3.history['val_loss']
#epochs = range(len(acc))

#plt.plot(epochs, acc, 'r', label='Training accuracy')
#plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.plot(epochs, loss, 'g', label='Training loss')
#plt.plot(epochs, val_loss, 'y', label='Validation loss')
#plt.title('Data test')
#plt.legend(loc=0)
#plt.savefig(r'C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\graficos\grafico.png')

#test_res = model3.evaluate(X_test, y_test.values, verbose=0)

#with open(r'C:\Users\Caio\Desktop\smartbee\imagens\baseParaWCAMA\dataset\graficos\resultados.csv', 'a') as arquivo_csv:
    #colunas = ['Loss', 'Accuracy']
    #escrever = csv.DictWriter(arquivo_csv, fieldnames=colunas, delimiter=',', lineterminator='\n')
    #escrever.writeheader()
    #escrever.writerow({'Loss': test_res[0], 'Accuracy': test_res[1]})


#path =  "C:\\Users\\Caio\\Desktop\\smartbee\\imagens\\baseParaWCAMA\\dataset\\graficos\\" + "grafico" + str(cont) + ".png"
#plt.savefig(path)