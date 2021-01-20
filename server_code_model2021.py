from __future__ import print_function

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import tensorflow as tf


from skimage.transform import resize
from skimage.io import imsave
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Flatten, Conv3D, MaxPooling3D, Conv3DTranspose, GlobalAveragePooling3D, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import History

imgs_train = np.load('imgs_train2021.npy')
masks_train = np.load('masks_train2021.npy')
imgs_test = np.load('imgs_test2021.npy')

smooth = 1

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model(img_rows=192, img_cols=192, img_depth=192):
    
    inputs = Input((img_rows, img_cols, img_depth, 1))
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(
        2, 2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv3D(1, (1, 1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=1e-5),
                  loss=dice_coef_loss, metrics=['dice_coef', 'accuracy'])

    return model


def get_model_simple(img_rows=192, img_cols=192, img_depth=192):
    """Build a 3D convolutional neural network model."""

    inputs = Input((img_rows, img_cols, img_depth, 1))

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = MaxPooling3D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Dense(units=512, activation="relu")(x)
    x = Dropout(0.3)(x)

    outputs = Conv3D(1, (1, 1, 1), activation='sigmoid')(x)
    # outputs = Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = Model(inputs, outputs, name="3dcnn")
    return model


def train():
    print('I am training...')

    model = get_model()
    
    model_checkpoint = ModelCheckpoint('weights2021.h5', monitor='val_loss', save_best_only=True)
    
    history = model.fit(imgs_train, masks_train, batch_size=32, epochs=5, verbose=1, shuffle=True, validation_split=0.2, callbacks=[model_checkpoint])
    print('I finished training...')


def predict():
    print('I am predicting...')
    
    model.load_weights('weights2021.h5')
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    
    np.save('imgs_mask_test2021.npy', imgs_mask_test)
    print('I finished predicting...')

def save_graph_results():
    print('I am saving graph results...')
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coeff')
    plt.ylabel('Dice coeff')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig('progr.png')
    print('I saved graph results...')


if __name__ == '__main__':
    train()
    predict()
    save_graph_results()
    
    