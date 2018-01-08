import numpy
import pygpu
import theano
import keras
import os
import tensorflow as tf
import skimage.io as io

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D, Reshape, Conv2DTranspose
from keras.layers import BatchNormalization, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import Adam

from keras.models import model_from_yaml
from keras.utils import plot_model
from keras.utils import np_utils
from keras import backend as K
from keras.metrics import binary_accuracy

import tensorflow as tf

from BilinearUpSampling import *

def superGenerator(image_gen, label_gen,batch):
    while True:
        x = image_gen.next()[0]
        y = label_gen.next()[0]
        #process label
        class_labels_tensor = K.equal(y, 1.0)
        background_labels_tensor = K.not_equal(y, 1.0)
        bit_mask_class = K.eval(tf.to_float(class_labels_tensor))
        bit_mask_bckg = K.eval(tf.to_float(background_labels_tensor))

        forg = numpy.reshape(bit_mask_class, (-1, bit_mask_class.shape[1]*bit_mask_class.shape[2]))
        bckg = numpy.reshape(bit_mask_bckg, (-1, bit_mask_bckg.shape[1]*bit_mask_bckg.shape[2]))
       
        label = numpy.stack((forg,bckg),axis=2)
        #label shape = (-1, 45056, 2), network output as well. 
        yield x, label


def binary_crossentropy_with_logits(ground_truth, predictions):
    return K.mean(K.binary_crossentropy(ground_truth, predictions, from_logits=True),axis=-1)


#def softmax_crossentropy_with_logits(y_true, y_pred):
#    tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true)




img_height = 352
img_width = 128

train_data_dir = '/home/piadph/MAG/SU/SU2017/Dataset/small/Train/Images'
train_label_dir = '/home/piadph/MAG/SU/SU2017/Dataset/small/Train/Labels'
validation_data_dir = '/home/piadph/MAG/SU/SU2017/Dataset/small/Validation/Images'
validation_label_dir = '/home/piadph/MAG/SU/SU2017/Dataset/small/Validation/Labels'
n_train_samples = 1000
n_validation_samples = 400
epochs = 25
batch_size = 5

input_shape = (img_height, img_width, 1)
target_shape = (img_height, img_width)

global graph
graph = tf.get_default_graph()

with graph.as_default():

    model = Sequential()

    #Layer 1

    model.add(Conv2D(80,(11,11), input_shape=input_shape, padding='same',  activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Layer 2
    model.add(Conv2D(96,(7,7), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Layer 3
    model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    #Layer 4
    model.add(Conv2D(160,(3,3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(2,2)))

    #FC Layer 1
    model.add(Conv2D(1024, (1,1), activation='relu'))

    #FC Layer 2
    model.add(Conv2D(512, (1,1), activation='relu'))

    #Classification layer
    model.add(Conv2D(2, (1,1)))

    #Bilinear upsampling
    model.add(BilinearUpSampling2D(target_size=(img_height, img_width)))

    #Softmax
    model.add(Conv2D(2, (1,1), activation='softmax'))

    #Reshape, to match labels (batch, w*h, 2)
    model.add(Reshape((img_height*img_width,2)))


    model.summary()



    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data_gen_args = dict(
            rescale= 1./ 255,
            rotation_range=10,
            height_shift_range=0.2,
            fill_mode='reflect',
            horizontal_flip=True,
            vertical_flip=True
            )

    train_datagen = ImageDataGenerator(**data_gen_args)
    train_label_datagen = ImageDataGenerator(**data_gen_args)
    test_datagen = ImageDataGenerator(**data_gen_args)
    test_label_datagen = ImageDataGenerator(**data_gen_args)

    seed = 5

    train_image_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=target_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)
    train_label_generator = train_label_datagen.flow_from_directory(
        train_label_dir,
        target_size=target_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    validation_image_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=target_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    validation_label_generator = test_label_datagen.flow_from_directory(
        validation_label_dir,
        target_size=target_shape,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed)

    train_generator = superGenerator(train_image_generator, train_label_generator,batch_size)
    test_generator = superGenerator(validation_image_generator, validation_label_generator,batch_size)

    model.fit_generator(
        train_generator,
        steps_per_epoch= n_train_samples // batch_size,
        epochs=25,
        verbose=1,
        validation_data=test_generator,
        validation_steps=n_validation_samples // batch_size)

    model.save_weights('first_try.h5')
    model_yaml = model.to_yaml()
    with open("model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
