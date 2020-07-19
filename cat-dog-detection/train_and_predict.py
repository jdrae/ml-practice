# jdrae: sum up article and practice code from the link below
# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

from cnn import classifier
from preprocessing import training_set, test_set

PRED_PATH = 'data/prediction/hahaha.jpg'

# fit data to model
classifier.fit(
    training_set,
    steps_per_epoch= 8000, # num of training images
    epochs = 25, # one epoch means every training samples has trained
    validation_data=test_set,
    validation_steps=2000
)

# new prediction from traied model
import numpy as np
from keras.preprocessing import image

test_image = image.load_img(
    PRED_PATH,
    target_size = (64,64)
)
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(PRED_PATH, ": ", prediction)
