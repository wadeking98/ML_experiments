import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

DATADIR = "../datasets/PetImages"
CATEGORIES = ['Dog', 'Cat']
IMG_SIZE = 64

training_data = []

if "--help" in sys.argv or "-h" in sys.argv:
    print("python3 main.py <options>\noptions: -h|--help : display help message\n\
        -load_data : load data and labels from <project_dir>/training\n\
        -load_model : load existing model from <project_dir>/models\n\
        -show <(int) low> <(int) high> : show training samples straining from low to high\n\
        -predict <(string) path> : run model on specified image file")
    exit(0)


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                img_normalized = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([img_normalized, class_num])
            except Exception as e:
                pass

X = []
Y = []

if "-load_data" in sys.argv:
    X = np.load("../training/inputs.npy")
    Y = np.load("../training/labels.npy")
else:
    create_training_data()
    random.shuffle(training_data)


    for features, label in training_data:
        X.append(features)
        Y.append(label)
    
    X = tf.keras.utils.normalize(X, axis=1)

    np.save("../training/inputs.npy", X)
    np.save("../training/labels.npy", Y)

# plt.imshow(X[2], cmap="gray")
# plt.show()

print(len(X), len(Y))
print(X.shape)

X = np.expand_dims(X, axis=-1)
print(X.shape)
model = None

if "-load_model" in sys.argv:
    model = tf.keras.models.load_model("../models/cdmodel.h5")
else:
    model = tf.keras.models.Sequential()
    model.add(Conv2D(64, (3,3), 
        data_format="channels_last", 
        input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss = "binary_crossentropy",
        optimizer="adam",
        metrics=['accuracy'])


    model.fit(X, Y, batch_size=64, epochs=3, validation_split=0.1)
    model.save("../models/cdmodel.h5")

if "-predict" in sys.argv:
    cmd_idx = sys.argv.index("-predict")
    fname = sys.argv[cmd_idx+1]

    img_array = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    img_normalized = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))

    img_normalized = np.expand_dims(img_normalized, axis=0)
    img_normalized = np.expand_dims(img_normalized, axis=-1)

    pred = model.predict(img_normalized)

    print(pred)
    index = int(round(pred[0][0]))
    print(f"prediction:{CATEGORIES[index]}")

    img_normalized = np.squeeze(img_normalized, axis=0)
    img_normalized = np.squeeze(img_normalized, axis=-1)
    plt.imshow(img_normalized, cmap="gray")
    plt.show()


if "-show" in sys.argv:
    cmd_idx = sys.argv.index("-show")
    low = int(sys.argv[cmd_idx+1])
    high = int(sys.argv[cmd_idx+2])
    pred = model.predict(X[low:high])
    X = np.squeeze(X, axis=-1)
    
    correct = 0
    curr = low
    for p in pred:
        index = int(round(p[0]))
        print(p)
        print(f"prediction:{CATEGORIES[index]} actual: {CATEGORIES[Y[curr]]}")

        if CATEGORIES[index] == CATEGORIES[Y[curr]]:
            correct+=1

        plt.imshow(X[curr], cmap="gray")
        plt.show()
        curr += 1

    print(f"Score: {correct/(high-low)}")

    