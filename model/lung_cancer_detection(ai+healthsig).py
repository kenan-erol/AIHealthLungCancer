from google.colab import files
import io
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
from google.colab import drive
import numpy as np 
!pip install pydicom
import pydicom 
import matplotlib.pyplot as plt
!pip install python-docx
import docx 
import pandas as pd 

drive.mount('/content/drive')

x = np.load(r'/content/drive/My Drive/images.npy')
y = np.load(r'/content/drive/My Drive/labels.npy')
patient_IDs = np.load(r'/content/drive/My Drive/patient_IDs.npy')

print(np.unique(patient_IDs).shape)

indices_to_drop = []

for i in range(len(y)):
    # print(y)
    if y[i] == 'N/A' or y[i] == '4':
        indices_to_drop.append(i)
        # print('true')

print('len x', len(x))
x = [x[i] for i in range(len(x)) if i not in indices_to_drop]
y = [y[i] for i in range(len(y)) if i not in indices_to_drop]
patient_IDs = [patient_IDs[i] for i in range(len(patient_IDs)) if i not in indices_to_drop]

y_new = []

for i in range(len(y)):
    if y[i] == '1A' or y[i] == '1B':
        y_new.append('1')
    elif y[i] == '2A' or y[i] == '2B':
        y_new.append('2')
    elif y[i] == '3A' or y[i] == '3B' or y[i] == '3':
        y_new.append('3')
    elif y[i] == '4':
        y_new.append('4')
    else:
        print(y_new)

y = y_new

print(len(x), len(y), len(patient_IDs))

groups = []

groups = [int(patient) for patient in patient_IDs]

# groups.append
# for index, row in data.iterrows():
    #  groups.append(int(row['patient']))
#     print(row)
#     groups.append(int(row['patient']))
# groupings by patient ID 
# please see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html

print(groups)

## Grouping code 


from google.colab import drive
drive.mount('/content/drive')
import sklearn
from sklearn.model_selection import train_test_split, ShuffleSplit

## Create our training, validation, testing datasets (70-15-15 split)
# [[list of all patient 1], [patient 2], [etc]] ## labels 
# [[patient 21], [patient 3]]

# 1 label, 1 patient ID, [all of the images]


# x = images 
# y = labels
patient_id = patient_IDs

# GroupShuffleSplit
gss = ShuffleSplit(n_splits=1, train_size=.7, random_state=60)
gss.get_n_splits()

# train_idx, test_idx = gss.split(x, y, groups)
for train_idx, test_idx in gss.split(x, y):
    train_idx = train_idx
    test_idx = test_idx
    # print("TRAIN:", train_idx, "TEST:", test_idx)

## Temporarily split into train and test before doing a test/validation split 
print(len(x), len(train_idx))


# x, y, patient_id = np.array(x), np.array(y), np.array(patient_id)
print('line 0')
x, y, patient_id = np.array(x), np.array(y), np.array(patient_id)
print("line 1")
x_train = [x[i] for i in train_idx]
y_train = [y[i] for i in train_idx]
patient_id_train = [patient_id[i] for i in train_idx]
print('done')
# x_train, y_train, patient_id_train = [x[i] for i in train_idx], [y[i] for i in train_idx], [patient_id[i] for i in train_idx]
# x_train = x[train_idx]
print("line2")

# x_train, y_train, patient_id_train = np.array(x)[train_idx], np.array(y)[train_idx], np.array(patient_id)[train_idx]
print('line 3')
### gets to this point ^^

x_test_temp, y_test_temp, patient_id_test_temp = [x[i] for i in test_idx], [y[i] for i in test_idx], [patient_id[i] for i in test_idx]
print(len(x_test_temp))
groups_test = [groups[i] for i in test_idx]


gss = ShuffleSplit(n_splits=1, train_size=.5, random_state=27) # was 43
gss.get_n_splits()

for test_idx, valid_idx in gss.split(x_test_temp, y_test_temp):
    test_idx = test_idx
    valid_idx = valid_idx 

x_valid, x_test = [list(x_test_temp)[i] for i in valid_idx], [list(x_test_temp)[i] for i in test_idx]
y_valid, y_test = [list(y_test_temp)[i] for i in valid_idx], [list(y_test_temp)[i] for i in test_idx]
patient_id_valid, patient_id_test = [list(patient_id_test_temp)[i] for i in valid_idx], [list(patient_id_test_temp)[i] for i in test_idx]

print(len(x_train), len(x_valid), len(x_test))

"""# Model code"""

import tensorflow
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

EPOCHS = 30 
BS = 16
INIT_LR = 1e-4

## tf model 
height, width, depth = 512, 512, 1
n_classes = 3
model = Sequential()
inputShape = (height, width, depth)
chanDim = -1

model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape, activation='relu'))

model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same",activation='relu'))

model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(n_classes))
model.add(Activation("softmax"))

import sklearn 
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

y_train_bin = LabelBinarizer().fit_transform(y_train)
y_test_bin = LabelBinarizer().fit_transform(y_test)
y_valid_bin = LabelBinarizer().fit_transform(y_valid)
y_valid = y_valid_bin
y_train = y_train_bin
y_test = y_test_bin

print(np.array(x_train, dtype=np.uint8).shape)

print(np.unique(y_test))
print(np.unique(y_train))

x_train = np.array(x_train, dtype=np.uint8)
y_train = np.array(y_train)
x_test = np.array(x_test, dtype=np.uint8)
y_test = np.array(y_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

x_test = np.concatenate([np.array(x_test), np.array(x_valid)])
print(x_test.shape)
y_test = np.concatenate([np.array(y_test), np.array(y_valid)])

# train the network
print("[INFO] training network...")
model.compile(loss="categorical_crossentropy", 
              optimizer=Adam(learning_rate=INIT_LR), 
              metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=BS,
                    validation_data=(np.array(x_test), np.array(y_test)), #was np.array(x_valid)
                    epochs=EPOCHS, verbose=1, shuffle=True)

test_preds = model.predict(np.array(x_test))
test_preds = np.argmax(test_preds, axis=1)
y_test_calc = np.argmax(y_test, axis=1)
print(test_preds)
from sklearn.metrics import accuracy_score

print("Test accuracy: ", accuracy_score(y_test_calc, test_preds))

