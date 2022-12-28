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

# we can now plot wordcloud using this tfidf matrix

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating wordcloud object
wordcloud = WordCloud().generate_from_frequencies(vectorizer.vocabulary_)
# plotting wordcloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# we can also plot this matrix without help of wordcloud by using the words as labels

import seaborn as sns

# plotting heatmap
sns.heatmap(tfidf.toarray(), annot=True, xticklabels=vectorizer.get_feature_names(), yticklabels=['text1', 'text2'])
plt.show()

