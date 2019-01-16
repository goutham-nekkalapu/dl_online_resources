
from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
import tensorflow as tf
from keras import layers 
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input, Activation, merge
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D  # Conv2D: Keras2
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D 
import keras.preprocessing.image as image
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model


class InceptionV3():
    """ The Inception V3 Imagenet model """

    # to be modified 
    def __init__(self, size=(224,224), include_top=True):
        self.class_FILE_PATH = 'http://files.fast.ai/models/'
        self.FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/'
        self.vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
        self.create(size, include_top)
        self.get_classes()
    
    # to be modified  
    def get_classes(self):
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.class_FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes

    def vgg_preprocess(self, x):
        x = x - self.vgg_mean
        axis = 3
        r,g,b = tf.split(x, 3, axis)
        return tf.concat([b,g,r], axis)

    def conv2d_bn(self, x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
        '''
        # Utility function to apply conv + BN.
    	# Arguments
       		 x: input tensor.
	         filters: filters in `Conv2D`.
        	 num_row: height of the convolution kernel.
	         num_col: width of the convolution kernel.
        	 padding: padding mode in `Conv2D`.
        	 strides: strides in `Conv2D`.
	         name: name of the ops; will become `name + '_conv'`
                 for the convolution and `name + '_bn'` for the batch norm layer.
        # Returns
          Output tensor after applying `Conv2D` and `BatchNormalization'
        '''
        
        if name is not None:
            bn_name = name + '_bn'
            conv_name = name + '_conv'
        else:
            bn_name = None
            conv_name = None
        
        bn_axis = 3 # for tensorflow 

        x = Conv2D(
            filters, (num_row, num_col),
	    strides=strides,
            padding=padding,
            use_bias=False,
            name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x

    def create(self, size, include_top):
        #def InceptionV3(include_top=True,
        #        weights='imagenet',
        #        input_tensor=None,
        #        input_shape=None,
        #        pooling=None,
        #        classes=1000)

        input_shape = size + (3,)  #to be modified to replace the below line  
        img_input = Input(shape=input_shape)
        channel_axis = 3
        classes = 1000 # for model with top layer (number of original imagent classes) 

        x = Lambda(self.vgg_preprocess)(img_input)
        x = self.conv2d_bn(x, 32, 3, 3, strides=(2, 2), padding='valid')
        x = self.conv2d_bn(x, 32, 3, 3, padding='valid')
        x = self.conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv2d_bn(x, 80, 1, 1, padding='valid')
        x = self.conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
                 [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                 axis=channel_axis,
                 name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
                     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                     axis=channel_axis,
                     name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = self.conv2d_bn(x, 64, 1, 1)

        branch5x5 = self.conv2d_bn(x, 48, 1, 1)
        branch5x5 = self.conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
                     [branch1x1, branch5x5, branch3x3dbl, branch_pool],
                     axis=channel_axis,
                     name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = self.conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = self.conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = self.conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = self.conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
                 [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 128, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
                     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 192, 1, 1)

            branch7x7 = self.conv2d_bn(x, 160, 1, 1)
            branch7x7 = self.conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = self.conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                         (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                     [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                     axis=channel_axis,
                     name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = self.conv2d_bn(x, 192, 1, 1)

        branch7x7 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = self.conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = self.conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = self.conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
                 [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = self.conv2d_bn(x, 192, 1, 1)
        branch3x3 = self.conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

        branch7x7x3 = self.conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = self.conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = self.conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
                 [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = self.conv2d_bn(x, 320, 1, 1)

            branch3x3 = self.conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = self.conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = self.conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                         [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

            branch3x3dbl = self.conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = self.conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = self.conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = self.conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                             [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

            branch_pool = AveragePooling2D(
                             (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = self.conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                             [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

        if include_top:
             # Classification block
             x = GlobalAveragePooling2D(name='avg_pool')(x)
             x = Dense(classes, activation='softmax', name='predictions')(x)
             fname = "inception_v3_weights_tf_dim_ordering_tf_kernels.h5"
        else:
             # this can be made programmable 
             '''
             if pooling == 'avg':
                 x = GlobalAveragePooling2D()(x)
             elif pooling == 'max':
                 x = GlobalMaxPooling2D()(x)
             '''
             x = GlobalMaxPooling2D()(x)
             fname = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5" 

        self.img_input = img_input
        self.model = Model(self.img_input, x, name = 'inception_v3')
        self.model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))

    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
        """
            Takes the path to a directory, and generates batches of augmented/normalized data. 
            Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def finetune(self, batches):
        model = self.model
        model.layers.pop()
        for layer in model.layers: layer.trainable=False
        m = Dense(batches.num_class, activation='softmax')(model.layers[-1].output)
        self.model = Model(model.input, m)
        self.model.compile(optimizer=RMSprop(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

    # Keras2
    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        """
            Trains the model for a fixed number of epochs (iterations on a dataset).
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit(trn, labels, epochs=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)

    # Keras2
    def fit(self, batches, val_batches, batch_size, nb_epoch=1):
        """
            Fits the model on data yielded batch-by-batch by a Python generator.
            See Keras documentation: https://keras.io/models/model/
        """
        self.model.fit_generator(batches, steps_per_epoch=int(np.ceil(batches.samples/batch_size)), epochs=nb_epoch,
                validation_data=val_batches, validation_steps=int(np.ceil(val_batches.samples/batch_size)))

  
    # Keras2
    def test(self, path, batch_size=8):
        """
            Predicts the classes using the trained model on data yielded batch-by-batch.

            Args:
                path (string):  Path to the target directory. It should contain one subdirectory 
                                per class.
                batch_size (int): The number of images to be considered in each batch.
            
            Returns:
                test_batches, numpy array(s) of predictions for the test_batches.
    
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, int(np.ceil(test_batches.samples/batch_size)))


    def summary(self):
        """
        gives the summary of the architecture of the odel instantiated
        """
        model = self.model
        return model.summary()


