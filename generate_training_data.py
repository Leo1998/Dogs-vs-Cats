import numpy as np 
import os
import cv2
import random

DATADIR = "C:/Users/Gott/Desktop/NN_Hacking/keras/dogsncats/PetImages"
CATEGORIES = ["Dog", "Cat"]

IMG_SIZE=50

training_data = []

def create_training_data():
	for category in CATEGORIES:
		path = os.path.join(DATADIR, category)
		class_idx = CATEGORIES.index(category)
		
		for img in os.listdir(path):
			try:
				img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
				training_data.append([new_array, class_idx])
			except Exception as e:
				pass

create_training_data()

random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
	X.append(features)
	Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = np.array(Y).reshape(-1)

print(X.shape)
print(Y.shape)

np.save("X", X)
np.save("Y", Y)