import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib
import pickle
#Read in and clean text
stopwords = nltk.corpus.stopwords.words('english')
#can use wordnet lemmatizer here also
ps = nltk.PorterStemmer()

data = pd.read_csv("open_balanced.csv", sep=',')
data.columns = ['agr_index', 'social_media_post']
#data.columns = ['agr_index', 'social_media_post','sentiment_polarity','sentiment_subjectivity']
print(data.shape)

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])#Reomve punctuations
    tokens = re.split('\W+', text)#Tokenize
    text = [ps.stem(word) for word in tokens if word not in stopwords] #Apply Porter Stemmer
    return text

###Vectorize text
###{Tfidf vectorization
##tfidf_vect = TfidfVectorizer(analyzer=clean_text)
##tfidf_vect_fit = tfidf_vect.fit(data['social_media_post'])#this will only generate the cols from the training set
##tfidf_x = tfidf_vect_fit.transform(data['social_media_post'])#this will calculate the frequency of unique words only from test set hence instead of having 8000 words we have 7000 words
###Also remeber that transform function only returns a sparse matrix, hence while giving it to the ML model we have to convert it to an array   
##
##tfidf_x_df = pd.DataFrame(tfidf_x.toarray())
##tfidf_x_df.columns = tfidf_vect.get_feature_names()
###}

#{Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer = clean_text)
X_counts = count_vect.fit_transform(data["social_media_post"])#creates a sparse matrix of the counts
tfidf_x_df = pd.DataFrame(X_counts.toarray())

#tfidf_x_df = pd.concat([data[['sentiment_polarity', 'sentiment_subjectivity']].reset_index(drop=True), pd.DataFrame(X_counts.toarray())], axis=1)
#}


###{N-Gram Vectorization
##from sklearn.feature_extraction.text import CountVectorizer
##
##def clean_text(text):
##    text = "".join([word.lower() for word in text if word not in string.punctuation])#Remove punctuations[like ,.[]{},etc] using list comprehension and the list of punctuations in the string library
##    tokens = re.split('\W+', text)#Tokenize [basically split the sentence into a list of words]
##    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords]) #Apply Porter Stemmer [reduce to the root word by choppping - less accurate more fast than lemmatization]
##    #text = " ".join([wn.lemmatize(word) for word in tokens if word not in stopwords]) #For testing Lemmatization
##    return text
##
##data["cleaned_text"] = data["social_media_post"].apply(lambda x: clean_text(x))
##n_gram_vect = CountVectorizer(ngram_range = (1,2))
##X_counts = n_gram_vect.fit_transform(data["cleaned_text"])
##tfidf_x_df = pd.DataFrame(X_counts.toarray())
###}


#random printing of info to check

#print(tfidf_x_df[0:50])
##print("\nSample feature names identified:" , tfidf_vect.get_feature_names())
##print("\nSize of TFIDF matrix", tfidf_x.shape)

#create a list for the labels assigned to the messages
classifications = list(data["agr_index"])

#identify the classes present in the dataset
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(classifications)
print("Classes found : ",le.classes_)

#convert the list of classes to list of class specific integers to input the model
int_classes = le.transform(classifications)
#print("\nClasses converted to integers for first section:",int_classes[0:5])

#prepare and apply naive bayesian algo
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

#Split as training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(tfidf_x_df, int_classes, random_state = 0)

#Build Models
classifier = MultinomialNB().fit(xtrain, ytrain)

#making predictions
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
print("Testing with Test Data :\n----------------------")
#Predcit on test data
predictions = classifier.predict(xtest)
print("Some predictions: ", predictions[0:6])
print("Confusion Matrix: ")
print(metrics.confusion_matrix(ytest, predictions))
print("\nPrediction Accuracy: ", metrics.accuracy_score(ytest, predictions))

#Evaluate model predictions using precision and recall
precision = precision_score(ytest, predictions, pos_label = 1)
recall = recall_score(ytest, predictions, pos_label = 1)
#f1score
f1_score = (2*precision*recall)/(precision + recall)
print("Precision: {} / Recall: {} /F1 Score: {}".format(round(precision, 3), round(recall, 3), round(f1_score,3)))

##
##print("\nTesting with Full Corpus: \n----------------------")
###Predict on entire corpus data
##predictions = classifier.predict(tfidf_x_df)
##print("Confusion Matrix: ")
##print(metrics.confusion_matrix(int_classes, predictions))
##print("\nPrediction Accuracy: ", metrics.accuracy_score(int_classes,predictions))
##print()
##







#Build Models
classifier  = ComplementNB(norm = True).fit(xtrain, ytrain)

#making predictions
print("Testing with Test Data :\n----------------------")
#Predcit on test data
predictions = classifier.predict(xtest)
print("Some predictions: ", predictions[0:6])
print("Confusion Matrix: ")
print(metrics.confusion_matrix(ytest, predictions))
print("\nPrediction Accuracy: ", metrics.accuracy_score(ytest, predictions))

#Evaluate model predictions using precision and recall
precision = precision_score(ytest, predictions, pos_label = 1)
recall = recall_score(ytest, predictions, pos_label = 1)
#f1score
f1_score = (2*precision*recall)/(precision + recall)
print("Precision: {} / Recall: {} /F1 Score: {}".format(round(precision, 3), round(recall, 3), round(f1_score,3)))

##print("\nTesting with Full Corpus: \n----------------------")
###Predict on entire corpus data
##predictions = classifier.predict(tfidf_x_df)
##print("Confusion Matrix: ")
##print(metrics.confusion_matrix(int_classes, predictions))
##print("\nPrediction Accuracy: ", metrics.accuracy_score(int_classes,predictions))


##Classes found :  ['n' 'y']
##Testing with Test Data :
##----------------------
##Some predictions:  [1 1 1 1 1 1]
##Confusion Matrix: 
##[[ 432  697]
## [ 280 1070]]
##
##Prediction Accuracy:  0.6058894715611134
##
##Testing with Full Corpus: 
##----------------------
##Confusion Matrix: 
##[[2970 1678]
## [ 448 4820]]
##
##Prediction Accuracy:  0.7855990318676885
######Conclusion ......its pretty good!! but only for agreeableness
######Its pathetic with mbti - accuracy of 22% approx
