## What's DATA
data is a collection of facts, such as numbers, words, measurements, observations, or even just descriptions of things. Data can be collected from a variety of sources, including paper forms, databases, and sensors. Data can be processed by a computer to produce useful information.

## What's DATA MINING
Data mining is the process of discovering patterns in large data sets involving methods at the intersection of machine learning, statistics, and database systems. Data mining is an interdisciplinary subfield of computer science and statistics with an overall goal to extract information (with intelligent methods) from a data set and transform the information into a comprehensible structure for further use.

## DATA preprocessing
Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues.

## how to preprocess data
1. Data Cleaning : remove or replace missing values
2. Data Integration : combine data from multiple sources
3. Data Transformation : normalize, aggregate, bin, discretize, etc.
4. Data Reduction : reduce the size of the data set
5. Data Discretization : discretize continuous attributes

## preprocessing tools
1. Weka
2. RapidMiner
3. KNIME
4. Orange
5. python: pandas, numpy, nltk.

## preprocess text data
1. remove punctuation
2. remove stop words
3. stemming
4. lemmatization

### example
```python
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# remove punctuation
def remove_punctuation(text):
    return ''.join([c for c in text if c not in punctuation])

# remove stop words
def remove_stop_words(text):
    return [word for word in text if word not in stopwords.words('english')]

# stemming
def stemming(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

# lemmatization
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

# preprocess text
def preprocess_text(text):
    text = remove_punctuation(text)
    text = word_tokenize(text)
    text = remove_stop_words(text)
    text = stemming(text)
    text = lemmatization(text)
    return text

text = 'I am a student. I am learning data mining. I am learning natural language processing.'
text = preprocess_text(text)
print(text)
```

