"""
created by : aek426rahmani
date : 28-12-2022
"""
# this file is for preprocessing text data using nltk library
# importing libraries

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# creating object of WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# creating object of stopwords
stop_words = set(stopwords.words('english'))

# creating function for preprocessing text data
def preprocess(text):
    # removing html tags
    text = re.sub(r'<.*?>', '', text)
    # removing urls
    text = re.sub(r'http\S+', '', text)
    # removing punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    # removing numbers
    text = re.sub(r'\d+', '', text)
    # removing special characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    # removing whitespaces
    text = text.strip()
    # removing stopwords
    text = [word for word in word_tokenize(text) if word not in stop_words]
    # lemmatizing words
    text = [lemmatizer.lemmatize(word) for word in text]
    # converting list to string
    text = ' '.join(text)
    return text

# test preprocess function
text = 'I am a student number 25992.#@14 I am learning data mining. I am lea5899rning natural language processing.'
text = preprocess(text)
print(text)