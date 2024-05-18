import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import xml.etree.ElementTree as xet
from keras.models import load_model, Sequential
from sklearn.model_selection import train_test_split

def load_images(directory):
    images = []

    for filepath in tqdm(os.listdir(directory)):
        img_path = os.path.join(directory, filepath)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = cv2.medianBlur(img, 5)
        images.append(img)

    images = np.array(images).reshape(len(images), 224, 224, 3)
    return images

path = glob("./annotations/*.xml")
labels_dict = dict(filepath=[], xmin=[], xmax=[], ymin=[], ymax=[])

for filename in path:
    data = xet.parse(filename)
    root = data.getroot()
    obj = root.find('object')
    labels_info = obj.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)
    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv', index=False)
df.head()

df.shape

images = load_images('./images')
images.shape

filename = df['filepath'][0]
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./images', filename_image)
    return filepath_image

getFilename(filename)

image_path = list(df['filepath'].apply(getFilename))
image_path[:10]

bbox_coords = df.iloc[:, 1:].values
labels = []

for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h, w, d = img_arr.shape
    xmin, xmax, ymin, ymax = bbox_coords[ind]
    nxmin, nxmax = int(xmin / (w / 255.)), int(xmax / (w / 255.))
    nymin, nymax = int(ymin / (h / 255.)), int(ymax / (h / 255.))
    labels.append((nxmin, nxmax, nymin, nymax))

labels = np.array(labels)
labels

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.float32)

X = X / 255.
y = y / 255.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

model = Sequential()
model.add(VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(4, activation="sigmoid"))

model.layers[-6].trainable = False

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

r = model.fit(X_train,
              y_train,
              epochs=10,
              batch_size=32,
              validation_data=(X_val, y_val),
              verbose=1)

plt.figure(figsize=(12, 8))
plt.plot(r.history['loss'], 'r', label='train loss')
plt.plot(r.history['val_loss'], 'b', label='validation loss')
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.title('Loss Graph')
plt.legend()

plt.figure(figsize=(12, 8))
plt.plot(r.history['accuracy'], 'r', label='train accuracy')
plt.plot(r.history['val_accuracy'], 'b', label='validation accuracy')
plt.xlabel('No. of Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Graph')
plt.legend()

loss, acc = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", acc)

model.save('license_plate_detector.h5')
detector = load_model('license_plate_detector.h5')
detector
