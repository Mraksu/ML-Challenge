
# Murat Aksu
# Felipe Moscoso Cruz
# Machine Learning Challenge
# 11 December 2019

# NOTE:
# This code orginially existed in three separate notebooks to increase workflow.




# Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Random seeds
np.random.seed(62)
tf.random.set_seed(63)
# Use this in case of an error due to different TensorFlow versions.
# tf.set_random_seed(63)




#####################
## DATA PREPARATION #
#####################

# This section pads the features with 0s to make them equal length and
# exports training and testing features and paths as separate file for later use.
# NOTE: Assumes feat.npy, path.npy, train.csv and test.csv are in the same directory as this file.

print("Preparing data (this will take a couple of minutes) ...")

# Import data
features = np.load('feat.npy', allow_pickle = True)
path = np.load('path.npy')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Returns the highest numer of frames in the dataset.
def maxFrames(features):
    maxFrames = 0

    for i, row in enumerate(features):
        rowLen = len(row)

        if rowLen > maxFrames:
            maxFrames = rowLen

    return maxFrames


# Returns the row indices of observations with a frame number higher than the given limit.
def invalidRows(features, limit):
    invalid = []

    for i, row in enumerate(features):
        rowLen = len(row)

        if rowLen >= limit:
            invalid.append(i)

    return invalid


# Find the invalid rows based on a calculated limit.
limit = maxFrames(features)
invalid = invalidRows(features, limit)


# Delete the rows with invalid files.
features_clean = np.delete(features, invalid)
path_clean = np.delete(path, invalid)


# Pad the dataset with 0s to deal with the different number of frames.

x = len(features_clean)
y = maxFrames(features_clean)
z = len(features_clean[0][0])

feat = np.zeros((x, y, z))

for i, row in enumerate(features_clean):
    for j, col in enumerate(row):
        for k, coef in enumerate(col):
            feat[i][j][k] = features_clean[i][j][k]


# Create an array of boolean values that indicate if a file belongs to training or testing.

testPath = np.array(test['path'])

isTest = np.isin(path_clean, testPath)
isTrain = np.logical_not(isTest)


# Split the array of features in testing in training.
featTest = feat[isTest]
featTrain = feat[isTrain]

np.save("feat_test.npy", featTest)
np.save("feat_train.npy", featTrain)


# Split the path in testing and training
pathTest = path_clean[isTest]
pathTrain = path_clean[isTrain]

np.save("path_test.npy", pathTest)
np.save("path_train.npy", pathTrain)

print("Data preparation complete")

####################
## MODEL TRAINING ##
####################

# This section trains the model, using the data files created during data preparation.
# It exports the fitted model and the training history for later use.

print("Training model ...")

# Import train data
feat = np.load("feat_train.npy")
path = np.load("path_train.npy")
label = pd.read_csv('train.csv')


# Check if the label and path arrays have the same indices.
for i, file in enumerate(path):
    if label["path"][i] != file:
        print("Paths and labels do not correspond.")
        break

# Encode the labels.
encoder = LabelEncoder()
y = encoder.fit_transform(label['word'])


# Flatten the features.
X_shape = feat.shape
X = feat.reshape((feat.shape[0], feat.shape[1] * feat.shape[2]))


# Compute the number of output neurons (i.e. classes).
n_output = len(np.unique(y))


# Scale the features.
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)


# Reshape the features back to a matrix to be able to feed it to the RNN.
X = X.reshape(X_shape)


# Copy X and y before proceeding.
X_train, y_train = X, y


# One hot encoding of the target labels.
y_train_hot = to_categorical(y_train)


# Build model, set hyper-parameters and fit it.
model_rnn = keras.Sequential([
    keras.layers.LSTM(512,dropout = 0.20, recurrent_dropout = 0.20, input_shape = X_train.shape[1:]),
    keras.layers.Dense(128,activation = "tanh"),
    keras.layers.Dense(n_output, activation = "softmax")
    ])

model_rnn.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

history = model_rnn.fit(X_train, y_train_hot, epochs = 40, batch_size = 300)


# Save model and model history
model_rnn.save("deep_focus.h5")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv('deep_history.csv', index = False)


# Plot training accuracy and loss.
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc = 'upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc = 'upper left')
plt.show()

print("Model training complete")


###################
## MODEL TESTING ##
###################

# This section tests the model created during model training.
# It exports the predictions to a csv file.

print("Testing model ...")

# Import test data
feat_test = np.load("feat_test.npy")
path_test = np.load("path_test.npy")
label_test = pd.read_csv("test.csv")


# Flatten the features in order to scale them.
X_test_shape = feat_test.shape
X_test = feat_test.reshape((feat_test.shape[0], feat_test.shape[1] * feat_test.shape[2]))


# Scale the features with the same scaler used before.
X_test = sc.transform(X_test)


# Reshape the features back to a matrix to be able to feed it to the RNN.
X_test = X_test.reshape(X_test_shape)


# Predict
y_test = model_rnn.predict_classes(X_test)


# Decode with the same encoder used before.
y_test_decoded = encoder.inverse_transform(y_test)


# Test if paths and labels correspond. (They don't, so we need to fix that below.)
for i, file in enumerate(path_test):
    if label_test['path'][i] != file:
        print("Paths and labels do NOT correspond!")
        break

    if i == len(path_test) - 1 and label_test['path'][i] == file:
        print("Paths and labels correspond.")


# IMPORTANT
# Change the order of the labels to match with the paths.
for i, file in enumerate(path_test):
    label_test['path'][i] = file


# Test if paths and labels correspond. (This should be fixed now.)
for i, file in enumerate(path_test):
    if label_test['path'][i] != file:
        print("Paths and labels do NOT correspond!")
        break

    if i == len(path_test) - 1 and label_test['path'][i] == file:
        print("Paths and labels correspond.")


# Export as csv file.
pred = pd.DataFrame({'path':label_test['path'], 'word':y_test_decoded})
pred.to_csv('result.csv', index = False)

print("Model testing complete")
