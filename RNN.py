from keras.preprocessing.text import Tokenizer #this is used to assign some numeric value to every word that appear in the training set
from keras.preprocessing.sequence import pad_sequences
import pandas as pd

#Load the cleaned training and test sets
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

#Train the tokenizer and use that tokenizer to convert the sentences to sequences of numbers
tokenizer = Tokenizer() #Load the tokenizer
tokenizer.fit_on_texts(X_train["clean_text"])#Train the tokenizer
X_train_seq = tokenizer.texts_to_sequences(X_train["clean_text"])#create token sequences for the sentences in the X_train file
X_test_seq = tokenizer.texts_to_sequences(X_test["clean_text"])#create token sequences for the sentences in the X_test file

#Now since the number of tokens in each sentence is not the same we pad it- meaning we either truncate or pad each sentence so that the number of tokens is as we want it to be

#Pad the sequences so that each sequence is the same length in our case 50
X_train_seq_padded = pad_sequences(X_train_seq, 64)#####WE CAN CHANGE 50
X_test_seq_padded = pad_sequences(X_test_seq, 64)

import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Dropout
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
#check with tanh and two lstm layers
model.add(Embedding(len(tokenizer.index_word)+1, 128)) #its like embedding text in the form of vectors
model.add(LSTM(128, dropout = 0.5, recurrent_dropout = 0.5, activation = "softmax", recurrent_activation = "softmax"))#dropouts can usually take values between 0.0 to 1.0 [kind of like percent]# return_sequences = True
#LSTM is an RNN layer
#dropout params are added in case the RNN overfits the data
#that happens when the training accuracy is significantly greater than the testing accuracy

model.add(Dense(128,activation = "sigmoid"))#relu, sigmoid, softmax, softplus, softsign, tanh, selu, elu, exponential
model.add(Dropout(0.2))
model.add(Dense(1, activation = "sigmoid"))#since we wish to predict only one class
model.summary()

#Compile the model
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy", precision_m, recall_m])

from keras.callbacks import EarlyStopping
es_callback = EarlyStopping(monitor='val_loss', patience=3)
#Fit the RNN - train the neural network
history = model.fit(X_train_seq_padded, y_train["label"], batch_size = 50, epochs = 10, validation_data = (X_test_seq_padded, y_test), verbose = 1,callbacks=[es_callback])

