# artifitial deep neural network

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_country = LabelEncoder()
X[:, 1] = labelencoder_X_country.fit_transform(X[:, 1])
labelencoder_X_gen = LabelEncoder()
X[:, 2] = labelencoder_X_gen.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

import keras 
from keras.models import Sequential
from keras.layers import Dense

# making the layers
classifier = Sequential()
# input layer and 1st hidden layer
classifier.add(Dense(output_dim = 10, init="uniform", activation="relu", input_dim=11))
# 2nd hidden layer
classifier.add(Dense(output_dim = 10, init="uniform", activation="relu"))
# output layer
classifier.add(Dense(output_dim = 1, init="uniform", activation="sigmoid"))
# compiling the neural network
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# training the neural network
classifier.fit(X_train, y_train, batch_size=5, epochs=100)

# predicting test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred>.5)

# creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

 







