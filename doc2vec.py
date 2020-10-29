import gensim
import nltk
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import string

#Read in and clena the text

#Load the stopwords
stopwords = nltk.corpus.stopwords.words("english")

data = pd.read_csv("neutral_balanced.csv", encoding = "utf-8")
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
X_train, X_test, y_train, y_test = train_test_split(data["clean_text"], data["label"], test_size = 0.35)

#print(X_train[:10])
#Create tagged document vectors [for now with the message index] for each social media message in the training and test sets
tagged_docs_train = [gensim.models.doc2vec.TaggedDocument(v,[i]) for i, v in enumerate(X_train)]
tagged_docs_test = [gensim.models.doc2vec.TaggedDocument(v,[i]) for i, v in enumerate(X_test)]

#print("tagged docs")
#print(tagged_docs_train[:10])

#Train the basic doc2vec model on the training tagged vectors
d2v_model = gensim.models.Doc2Vec(tagged_docs_train, vector_size = 75, window = 5, min_count = 2)
#window looks for the context of the word within 5 words around it min_count mentions how many times the word should appear in the dataset for it to influence the vectors

#the words section in the tagged docs have strings of the list, hence we use the eval function below to extract the list - dont have to use anymore because the string prob was solved
#create doc vectors from training and test tagged docs using the trained model
train_vectors = [d2v_model.infer_vector(v.words) for v in tagged_docs_train]
test_vectors = [d2v_model.infer_vector(v.words) for v in tagged_docs_test]

#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

rf = RandomForestClassifier(n_estimators = 30, criterion = "entropy", max_depth = 10, bootstrap = True, oob_score = True, random_state = 42, warm_start = True,max_samples = int(X_train.shape[0]*(3/4)))#Load
rf_model = rf.fit(train_vectors, y_train.values.ravel())#Fit the model

#predict the values
y_pred = rf_model.predict(test_vectors)
y_train_pred = rf_model.predict(train_vectors)

#Calculate the scores
precision = precision_score(y_test, y_pred)
precision_train = precision_score(y_train, y_train_pred)
recall = recall_score(y_test, y_pred)
recall_train = recall_score(y_train, y_train_pred)
accuracy = accuracy_score(y_test, y_pred)
accuracy_train = accuracy_score(y_train, y_train_pred)
print("Test: Precision: {} / Recall: {} /Accuracy: {}".format(round(precision,3),round(recall,3),round(accuracy,3)))
print("Training: Precision: {} / Recall: {} /Accuracy: {}".format(round(precision_train,3),round(recall_train,3),round(accuracy_train,3)))
