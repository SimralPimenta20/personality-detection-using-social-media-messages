from keras.preprocessing.text import Tokenizer #this is used to assign some numeric value to every word that appear in the training set
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import nltk
import numpy as np
import re
from sklearn.model_selection import train_test_split
import string

#Read in and clean the text

#Load the stopwords
stopwords = nltk.corpus.stopwords.words("english")

data = pd.read_csv("agr_balanced.csv", encoding = "utf-8")
data.columns = ["label", "text"]
#we replace the classes by 1 and 0 for y and n respectively
data["label"] = np.where(data["label"]=="y",1,0)

#define function to clean text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [word for word in tokens if word not in stopwords]
    return text

#actually clean the text and create a new column for it
data["clean_text"] = data["text"].apply(lambda x: clean_text(x))

#Now split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data["clean_text"], data["label"], test_size = 0.2)

#Train the tokenizer and use that tokenizer to convert the sentences to sequences of numbers
tokenizer = Tokenizer() #Load the tokenizer
tokenizer.fit_on_texts(X_train)#Train the tokenizer
X_train_seq = tokenizer.texts_to_sequences(X_train)#create token sequences for the sentences in the X_train file
X_test_seq = tokenizer.texts_to_sequences(X_test)#create token sequences for the sentences in the X_test file

#Now since the number of tokens in each sentence is not the same we pad it- meaning we either truncate or pad each sentence so that the number of tokens is as we want it to be

#Pad the sequences so that each sequence is the same length in our case 50
X_train_seq_padded = pad_sequences(X_train_seq, 64)#####WE CAN CHANGE 50
X_test_seq_padded = pad_sequences(X_test_seq, 64)

import keras.backend as K
from keras.layers import Dense, Embedding, LSTM, SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dropout, GRU
from keras.models import Sequential

#Since its not an sklearn model but rather a neural network we define our own recall and precision functions
#recall
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

#precision
def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives/ (predicted_positives + K.epsilon())
    return precision

###CONSTRUCT the basic RNN model framework
#we are creating a sequential model
model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, 64))
model.add(SeparableConv1D(filters = 64, kernel_size = 3, activation = "relu", bias_initializer = "random_uniform", padding = "same"))
model.add(MaxPooling1D(pool_size = 3))
model.add(SeparableConv1D(filters = 64, kernel_size = 3, activation = "relu", bias_initializer = "random_uniform", padding = "same"))
model.add(SeparableConv1D(filters = 64, kernel_size = 3, activation = "relu", bias_initializer = "random_uniform", padding = "same"))
model.add(GlobalAveragePooling1D())
model.add(Dense(units = 2, activation = "softmax"))
model.summary()

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es = EarlyStopping(monitor = "val_loss", min_delta = 0.005, patience = 3, verbose = 1, mode = "min", baseline = 1.5)
mc = ModelCheckpoint("best_weights.h5", monitor = "val_loss", verbose = 1, mode = "auto")
rd = ReduceLROnPlateau(monitor = "val_loss", factor = 0.1, patience = 3, verbose = 1, mode = "auto", min_delta = 0.0001, cooldown = 0, min_lr = 0)

#Fit the RNN - train the neural network
history = model.fit(X_train_seq_padded, y_train, epochs = 10, validation_data = (X_test_seq_padded, y_test), verbose = 1,callbacks=[es, mc, rd])

