import numpy as np
import os
import glob
import tensorflow as tf
import PIL.Image

ROOT_DIR = 'data'
DATA_LIST = glob.glob(ROOT_DIR + '\\*\\*.jpg')
IMG_SIZE = 256

def get_label(path):
    return path.split('\\')[1]

def get_file_name(path):
    return path.split('\\')[2]

def make_dir(label, keyword):
    dir = ROOT_DIR + "\\" +label + "\\"+ keyword
    if not os.path.exists(dir):
        os.makedirs(dir)
        print("--> dir made")
    return dir

def resize(org_img, keyword):
    image = PIL.Image.open(org_img)
    resized = image.resize((IMG_SIZE,IMG_SIZE))
    dir = make_dir(get_label(org_img), keyword)
    resized.save(dir + "\\re-" + get_file_name(org_img), "JPEG", quality=100)
    return resized

def augment(org_img, folder):
    image_string = tf.io.read_file(org_img)
    image = tf.image.decode_jpeg(image_string, channels = 3)
    
    flipped = tf.image.flip_left_right(image)
    flipped.save("flipped.jpg", "JPEG", quality=100)

    rotated = tf.image.rot90(image)
    grayscaled = tf.image.rgb_to_grayscale(image)
    saturated = tf.image.adjust_saturation(image, 3)
    bright = tf.image.adjust_brightness(image, 0.4)


org_img = DATA_LIST[0]

image_string = tf.io.read_file(org_img)
image = tf.image.decode_jpeg(image_string, channels = 3)

flipped = tf.image.flip_left_right(image)
# enc = tf.image.encode_jpeg(flipped)
enc = tf.io.encode_jpeg(flipped, quality = 100)
tf.io.write_file("hi.jpg", enc)