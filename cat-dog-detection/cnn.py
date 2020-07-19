# jdrae: sum up article and practice code from the link below
# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

from keras.models import Sequential # initialising a neural network (sequence of layers / graph)
from keras.layers import Conv2D # images are 2 dimensional (3D is for video)
from keras.layers import MaxPooling2D # choose max value pixel from region (while building cnn)
from keras.layers import Flatten # 2D arrays into a single continuous linear vector
from keras.layers import Dense # to perform full connection of the neural network


# Step1: Convolution
classifier = Sequential()
classifier.add(
    # num of filters, shape of each filters, image setting(3 stands for RGB), activation func
    Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu')
)

# Step2: Pooling - to reduce the size of images(reduce total number of node for the upcoming layers)
classifier.add(MaxPooling2D(pool_size=(2,2))) # reduce the complexity of the model without reducing it’s performance.

# Step3: Flattening
classifier.add(Flatten())

# Step4: Full connection
# to choose 'units' number: between the number of input nodes and output nodes -> need tries
classifier.add(Dense(units=128, activation='relu')) # hidden layer
# contain only one node(because it's binary classification)
classifier.add(Dense(units=1, activation='sigmoid')) # output(final) layer

# compile CNN model
# stochastic gradient descent algorithm, loss function, performance metric
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

