"""
created by : aek426rahmani
date : 28-12-2022
"""
from sklearn.feature_extraction.text import TfidfVectorizer

text = ['I am a student. I am learning data mining. I am learning natural language processing.',
        'I am a student. I am learning data mining. I am learning natural language processing.']

vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(text)
print(tfidf.toarray())
print(vectorizer.get_feature_names())