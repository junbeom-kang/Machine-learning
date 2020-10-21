import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def make_filter():
    weight = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    weight=weight.reshape((1,3,3,1))
    weight_init = tf.constant_initializer(weight)
    return weight_init

def cnn_valid(image, weight, option, sizeshape_h,sizeshape_w):
    conv2d = keras.layers.Conv2D(filters=1, kernel_size=(1,3), padding=option,
                                 kernel_initializer=weight)(image)
    print("conv2d.shape", conv2d.shape)
    plt.imshow(conv2d.numpy().reshape(sizeshape_h,sizeshape_w), cmap='gray')
    plt.show()

def main():
    image=plt.imread('C:/User/edge_detection_ex.jpg')
    image = image.reshape((1, 720,1280, 3))
    image = tf.constant(image, dtype=tf.float64)
    filter = make_filter()
    cnn_valid(image, filter, 'SAME',720,1280)
main()