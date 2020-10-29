import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib
import pickle
#Read in and clean text

#Load the stopwords
stopwords = nltk.corpus.stopwords.words('english')

#Load the Porter stemmer
ps = nltk.PorterStemmer()

###Load the Wordnet lemmatizer
##wn = nltk.WordNetLemmatizer()

#Read data using Pandas lib
data = pd.read_csv("open.csv", sep=',')
#Name the columns of the csv file
data.columns = ['label', 'body_text']

#Define the function to clean the text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])#Remove punctuations[like ,.[]{},etc] using list comprehension and the list of punctuations in the string library
    tokens = re.split('\W+', text)#Tokenize [basically split the sentence into a list of words]
    text = [ps.stem(word) for word in tokens if word not in stopwords] #Apply Porter Stemmer [reduce to the root word by choppping - less accurate more fast than lemmatization]
    #text = [wn.lemmatize(word) for word in tokens if word not in stopwords] #For testing Lemmatization
    return text

###Vectorize text
###{TfidfVectorization
##tfidf_vect = TfidfVectorizer(analyzer=clean_text) #It takes the clean textfunction and applies it internally
##tfidf_vect_fit = tfidf_vect.fit(data['body_text'])#this will only generate the cols from the data set [this is an indirect way, fit and transform can be done together but we want to load the features while predicting]
###[so its better to save them][also it can be used in the test data since the test data may result in a different set of columns.]
##
##tfidf_x = tfidf_vect_fit.transform(data['body_text'])#this will calculate the frequency of unique words only from test set hence instead of having 8000 words we have 7000 words
###Also remeber that transform function only returns a sparse matrix, hence while giving it to the ML model we have to convert it to an array   
###so we convert it into the array with column name [feature names] as given below
##tfidf_x_df = pd.DataFrame(tfidf_x.toarray())
###}

#{Count Vectorization
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer = clean_text)
X_counts = count_vect.fit_transform(data["body_text"])#creates a sparse matrix of the counts
tfidf_x_df = pd.DataFrame(X_counts.toarray())
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
##data["cleaned_text"] = data["body_text"].apply(lambda x: clean_text(x))
##n_gram_vect = CountVectorizer(ngram_range = (1,2))
##X_counts = n_gram_vect.fit_transform(data["cleaned_text"])
##tfidf_x_df = pd.DataFrame(X_counts.toarray())
###}




#GRADIENT BOOSTING METHOD
from sklearn.ensemble import GradientBoostingClassifier

#Import methods that will be needed to evaluate the basic model
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

#Split data into training and test sets
X_train,X_test, y_train, y_test = train_test_split(tfidf_x_df, data["label"], test_size = 0.2)

#fit the Gradient Boosting model
gb = GradientBoostingClassifier(n_estimators=50, max_depth=11, random_state = 42, warm_start = True,n_iter_no_change = 7, tol = 0.02, validation_fraction = 0.3)#n_iter_no_change = 5, tol = 0.01, validation_fraction = 0.2
gb_model = gb.fit(X_train, y_train)

###If you wish to save the model
##joblib.dump(gb_model,'agr.pkl')
##gb_model = joblib.load('agr.pkl')

#Perform predictions
y_pred = gb_model.predict(X_test)

#Evaluate model predictions using precision and recall
precision = precision_score(y_test, y_pred, pos_label = "y")
recall = recall_score(y_test, y_pred, pos_label = "y")
#f1score
f1_score = (2*precision*recall)/(precision + recall)
print("Precision: {} / Recall: {} /F1 Score: {}".format(round(precision, 3), round(recall, 3), round(f1_score,3)))
print("\nPrediction Accuracy: ", metrics.accuracy_score(y_test, y_pred))

if f1_score>0.80:
    #store the model
    joblib.dump(gb_model, "GB_model.pkl")

##Precision: 0.569 / Recall: 0.842 /F1 Score: 0.679
##
##Prediction Accuracy:  0.5806451612903226
