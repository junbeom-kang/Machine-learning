from keras import layers
from keras import models
def makemodel(input_data):
    model1 = layers.Conv2D(128, (1,1), padding='same')(input_data)

    model2 = layers.Conv2D(64, (1,1), padding='same')(input_data)
    model2 = layers.Conv2D(192, (3, 3), padding='same')(model2)

    model3 = layers.Conv2D(64, (1,1), padding='same')(input_data)
    model3 = layers.Conv2D(96, (5, 5), padding='same')(model3)

    model4 = layers.MaxPooling2D((3, 3),padding='same')(input_data)
    model4 = layers.Conv2D(64, (1, 1), padding='same')(model4)
    return layers.concatenate([model1, model2, model3, model4], axis = -1)
#Filter_concatenation=makemodel(Previous_Layer)
