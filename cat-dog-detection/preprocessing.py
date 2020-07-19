# jdrae: sum up article and practice code from the link below
# https://becominghuman.ai/building-an-image-classifier-using-deep-learning-in-python-totally-from-a-beginners-perspective-be8dbaf22dd8

from keras.preprocessing.image import ImageDataGenerator

TRAIN_PATH = 'data/training_set'
TEST_PATH = 'data/test_set'

# rescale images
train_data = ImageDataGenerator(
    rescale = 1./255,
    shear_range=0.2,
    zoom_range= 0.2,
    horizontal_flip= True
)

test_data = ImageDataGenerator(
    rescale= 1./255
)

# labelling by directory
training_set = train_data.flow_from_directory(
    TRAIN_PATH,
    target_size=(64,64),
    batch_size= 32,
    class_mode='binary'
)

test_set = test_data.flow_from_directory(
    TEST_PATH,
    target_size=(64,64),
    batch_size= 32,
    class_mode='binary'
)
