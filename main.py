"""
In this file, I created an Inception CNN architecture to classify
leaf disease from Leaf Dataset. The Dataset can be downloaded from:

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, concatenate, \
    GlobalAvgPool2D, Flatten, Dense, Dropout, AvgPool2D, MaxPool2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.initializers import glorot_uniform, Constant


def inception_Block_naive(input_layer, filters):
    """
    This is the inception block for CNN architecture INCEPTION
    Augments:
        input_layer: The tensorflow layer to implement
        filters: A list of filters for 3 Convolution Layer
    return:
        output layer
    """
    conv1 = Conv2D(filters[0], kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(input_layer)
    conv3 = Conv2D(filters[1], kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(input_layer)
    conv5 = Conv2D(filters[2], kernel_size=(5, 5), padding='same', activation='relu', kernel_initializer=glorot_uniform(seed=0))(input_layer)
    pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    layer_out = concatenate([conv1, conv3, pool], axis=-1)
    return layer_out


def inception_Block(input_layer, filters):
    """
    This is the inception block for CNN architecture INCEPTION
    Augments:
        input_layer: The tensorflow layer to implement
        filters: A list of filters for 3 Convolution Layer
    return:
        output layer
    """
    x = input_layer
    # 1x1 Conv
    y = Conv2D(filters[0], kernel_size=(1, 1), padding='valid',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(x)
    y = BatchNormalization(axis=3)(y)
    conv1 = Activation('relu')(y)

    # 1x1 Conv+BN -> 3x3 conv+BN -> 3x3 conv+BN
    y = Conv2D(filters[1], kernel_size=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(x)
    y = BatchNormalization(axis=3)(y)
    y = Conv2D(filters[1], kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(y)
    y = BatchNormalization(axis=3)(y)
    y = Conv2D(filters[1], kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(y)
    y = BatchNormalization(axis=3)(y)
    conv3 = Activation('relu')(y)

    # 1x1 Conv+BN -> 5x5 Conv+BN
    y = Conv2D(filters[2], kernel_size=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(x)
    y = BatchNormalization(axis=3)(y)
    y = Conv2D(filters[2], kernel_size=(5, 5), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(y)
    y = BatchNormalization(axis=3)(y)
    conv5 = Activation('relu')(y)

    # AvgPool2D (3,3)  + stride (1,1) + conv 1x1
    y = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x)
    y = Conv2D(filters[2], kernel_size=(1, 1), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(y)
    y = BatchNormalization(axis=3)(y)
    pool = Activation('relu')(y)
    # Concatenate layers
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


if __name__ == '__main__':
    imageSize = (150, 150, 3)
    batchSize = 2

    Train_path = '../DATA/LeafDisease/Datasets'
    Images = glob(Train_path + '/*/*.jp*g')
    folders = glob(Train_path + '/*')
    print(len(folders))

    singleImg = np.random.choice(Images)
    name = singleImg.split('\\')[-2]
    plt.imshow(load_img(singleImg))
    plt.title(name)
    plt.show()

    gen = ImageDataGenerator(rotation_range=45,
                             shear_range=0.1,
                             zoom_range=0.1,
                             height_shift_range=0.1,
                             width_shift_range=0.1,
                             validation_split=0.25,
                             horizontal_flip=True,
                             vertical_flip=True)

    Train_gen = gen.flow_from_directory(Train_path,
                                        shuffle=True,
                                        target_size=imageSize[:2],
                                        batch_size=batchSize,
                                        subset='training',
                                        class_mode='categorical')
    Valid_gen = gen.flow_from_directory(Train_path,
                                        shuffle=True,
                                        target_size=imageSize[:2],
                                        batch_size=batchSize,
                                        subset='validation',
                                        class_mode='categorical')
    Test_gen = gen.flow_from_directory(Train_path,
                                       shuffle=False,
                                       target_size=imageSize[:2],
                                       batch_size=batchSize,
                                       subset='validation',
                                       class_mode='categorical')

    # Model
    X_input = Input(shape=imageSize)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(16, kernel_size=(3, 3), padding='same',
               kernel_initializer=glorot_uniform(seed=0), bias_initializer=Constant(0.2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)

    X = Conv2D(16, kernel_size=(1, 1), padding='same', strides=(1, 1))(X)
    X = Conv2D(16, kernel_size=(3, 3), padding='same', strides=(1, 1))(X)
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [32, 64, 32])
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [64, 64, 64])
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [128, 128, 128])
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [256, 256, 256])
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [128, 128, 128])
    X = MaxPool2D((3, 3), padding='same', strides=(2, 2))(X)
    X = inception_Block(X, [64, 64, 64])
    X = GlobalAvgPool2D()(X)
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation='relu')(X)
    X = Dropout(0.5)(X)
    pred = Dense(len(folders), activation='softmax')(X)

    model = Model(inputs=X_input, outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    model_checkpoints = ModelCheckpoint(monitor='val_loss',
                                        mode='min',
                                        filepath='LeafDisease.h5',
                                        save_freq='epoch',
                                        save_best_only=True)

    steps_per_epochs = Train_gen.n // batchSize
    steps_validation = Valid_gen.n // batchSize
    epochs = 75

    print("Fitting the Model")
    res = model.fit(Train_gen,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epochs,
                    validation_data=Valid_gen,
                    validation_steps=steps_validation,
                    callbacks=[model_checkpoints],
                    verbose=2)
    plt.plot(res.history['accuracy'])
    plt.show()

    results = pd.DataFrame(model.history.history)
    results[['val_loss', 'loss']].plot()
    plt.show()
    from tensorflow.keras.models import load_model
    model = load_model('LeafDisease.h5')
    print(model.evaluate(Test_gen))
    from sklearn.metrics import confusion_matrix, classification_report

    pred = model.predict(Test_gen)
    pred = np.argmax(pred, axis=-1)
    cm = confusion_matrix(Test_gen.classes, pred)
    print(cm)
    print(classification_report(Test_gen.classes, pred))
"# LeafDisease_Inception" 
