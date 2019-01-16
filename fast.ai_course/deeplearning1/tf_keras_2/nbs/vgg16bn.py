from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D  # Conv2D: Keras2
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image


# vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1)) # Theano 
vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))

def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        # the below comments might be different as they are changed from 'theano' to 'tensorflow'. need to be updated 
        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    '''
    # "for theano"
    x = x - vgg_mean
    return x[:, ::-1] # reverse axis rgb->bgr
    '''
    x = x - vgg_mean
    axis = 3
    r,g,b = tf.split(x, 3, axis)
    return tf.concat([b,g,r], axis)



class Vgg16BN():
    """The VGG 16 Imagenet model with Batch Normalization for the Dense Layers"""


    def __init__(self, size=(224,224), include_top=True):
        # self.FILE_PATH = 'http://files.fast.ai/models/' # to download 'Theano' model
        self.FILE_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/'
        self.create(size, include_top)
        self.get_classes()

    def get_classes(self):
        """
            Downloads the Imagenet classes index file and loads it to self.classes.
            The file is downloaded only if it not already in the cache.
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        """
            Predict the labels of a set of images using the VGG16 model.

            Args:
                imgs (ndarray)    : An array of N images (size: N x width x height x channels).
                details : ??
            
            Returns:
                preds (np.array) : Highest confidence value of the predictions for each image.
                idxs (np.ndarray): Class index of the predictions with the max confidence.
                classes (list)   : Class labels of the predictions with the max confidence.
        """
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        """
            Adds a specified number of ZeroPadding and Covolution layers
            to the model, and a MaxPooling layer at the very end.

            Args:
                layers (int):   The number of zero padded convolution layers
                                to be added to the model.
                filters (int):  The number of convolution filters to be 
                                created for each layer.
        """
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))  # Keras2
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    def FCBlock(self):
        """
            Adds a fully connected layer of 4096 neurons to the model with a
            Dropout of 0.5

            Args:   None
            Returns:   None
        """
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))


    def create(self, size, include_top):
        """
            Creates the VGG16 network achitecture and loads the pretrained weights.

            Args:   None
            Returns:   None
        """
        if size != (224,224):
            include_top=False

        model = self.model = Sequential()
        no_channels = (3,) 
        # model.add(Lambda(vgg_preprocess, input_shape=(3,)+size, output_shape=(3,)+size)) #for theano 
        model.add(Lambda(vgg_preprocess, input_shape= size + no_channels, output_shape= size + no_channels))

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        if not include_top:
            fname = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))
            return

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        # fname = 'vgg16_bn.h5' # for a 'theano' trained model 
        fname = 'vgg16_weights_tf_dim_ordering_tf_kernels.h5' 
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):
       """
            Takes the path to a directory, and generates batches of augmented/normalized data. 
            Yields batches indefinitely, in an infinite loop.

            See Keras documentation: https://keras.io/preprocessing/image/
        """
       return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

    def ft(self, num):
        """
            Replace the last layer of the model with a Dense (fully connected) layer of num neurons.
            Will also lock the weights of all layers except the new layer so that we only learn
            weights for the last layer in subsequent training.

            Args:
                num (int) : Number of neurons in the Dense layer
            Returns:
                None
        """
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        """
            Modifies the original VGG16 network architecture and updates self.classes for new training data.
            
            Args:
                batches : A keras.preprocessing.image.ImageDataGenerator object.
                          See definition for get_batches().
        """
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(batches.num_class, activation='softmax'))  # Keras2
        self.compile()


    def compile(self, lr=0.001):
       """
            Configures the model for training.
            See Keras documentation: https://keras.io/models/model/
        """
       self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


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
