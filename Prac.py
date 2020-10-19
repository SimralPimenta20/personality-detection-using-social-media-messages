import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import joblib
import pickle
#Read in and clean text
stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("mbti_1.csv", sep=',')
data.columns = ['personality', 'social_media_post']

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])#Reomve punctuations
    tokens = re.split('\W+', text)#Tokenize
    text = [ps.stem(word) for word in tokens if word not in stopwords] #Apply Porter Stemmer
    return text

#Vectorize text
tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(data['social_media_post'])#this will only generate the cols from the training set
pickle.dump(tfidf_vect_fit, open("test1.pickle", "wb"))
pickle_in = open("test1.pickle","rb")
tfidf_vect_fit = pickle.load(pickle_in)
tfidf_x = tfidf_vect_fit.transform(data['social_media_post'])#this will calculate the frequency of unique words only from test set hence instead of having 8000 words we have 7000 words
#Also remeber that transform function only returns a sparse matrix, hence while giving it to the ML model we have to convert it to an array   

tfidf_x_df = pd.DataFrame(tfidf_x.toarray())
tfidf_x_df.columns = tfidf_vect.get_feature_names()
print(tfidf_x_df[0:50])
