import sys,os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져오도록 설정
import numpy as np
from PIL import Image
import pickle
from dataset.mnist import load_mnist
from activation_function import sigmoid
from output_layer import softmax

# show image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# neural network
def get_data():
    (x_train, y_train), (x_test, y_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)

    return y

x_test, y_test = get_data()
network = init_network()

# img_show(x_test[0].reshape(28,28)) # shows black image because normalized at first

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p =  np.argmax(y_batch, axis=1) # axis=1 means first dimension
    accuracy_cnt += np.sum(p == y_test[i:i+batch_size])

print("Accuracy:", float(accuracy_cnt)/len(x_test))
