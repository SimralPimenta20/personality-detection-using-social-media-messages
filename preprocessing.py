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
data = pd.read_csv("agr.csv", sep=',')
#Name the columns of the csv file
data.columns = ['label', 'body_text']

#Define the function to clean the text
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])#Remove punctuations[like ,.[]{},etc] using list comprehension and the list of punctuations in the string library
    tokens = re.split('\W+', text)#Tokenize [basically split the sentence into a list of words]
    text = [ps.stem(word) for word in tokens if word not in stopwords] #Apply Porter Stemmer [reduce to the root word by choppping - less accurate more fast than lemmatization]
    #text = [wn.lemmatize(word) for word in tokens if word not in stopwords] #For testing Lemmatization
    return text

#Vectorize text
tfidf_vect = TfidfVectorizer(analyzer=clean_text) #It takes the clean textfunction and applies it internally
tfidf_vect_fit = tfidf_vect.fit(data['body_text'])#this will only generate the cols from the data set [this is an indirect way, fit and transform can be done together but we want to load the features while predicting]
#[so its better to save them][also it can be used in the test data since the test data may result in a different set of columns.]
pickle.dump(tfidf_vect_fit, open("agr.pickle", "wb"))
pickle_in = open("agr.pickle","rb")
tfidf_vect_fit = pickle.load(pickle_in)
tfidf_x = tfidf_vect_fit.transform(data['body_text'])#this will calculate the frequency of unique words only from test set hence instead of having 8000 words we have 7000 words
#Also remeber that transform function only returns a sparse matrix, hence while giving it to the ML model we have to convert it to an array   
#so we convert it into the array with column name [feature names] as given below
tfidf_x_df = pd.DataFrame(tfidf_x.toarray())
tfidf_x_df.columns = tfidf_vect.get_feature_names()
print(tfidf_x_df[0:5])

###{Count Vectorization
##from sklearn.feature_extraction.text import CountVectorizer
##
##count_vect = CountVectorizer(analyzer = clean_text)
##X_counts = count_vect.fit_transform(data["social_media_post"])#creates a sparse matrix of the counts
##tfidf_x_df = pd.DataFrame(X_counts.toarray())
##
###tfidf_x_df = pd.concat([data[['sentiment_polarity', 'sentiment_subjectivity']].reset_index(drop=True), pd.DataFrame(X_counts.toarray())], axis=1)
###}


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
