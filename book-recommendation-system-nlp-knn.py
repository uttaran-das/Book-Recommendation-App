# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 14:36:35 2022

@author: user
"""

import pandas as pd

dataset = pd.read_csv('Book_Dataset.csv')
dataset = dataset.drop_duplicates().reset_index()

'''
Columns to keep
'title', 'author', 'description', 'language', 'genres', 'characters', 'bookFormat', 'publisher' , 'awards', 'setting', 'coverImg'
'''

df = dataset[['title', 'author', 'description', 'language', 'genres', 'characters', 'bookFormat', 'publisher' , 'awards', 'setting', 'coverImg']]
df = df.drop_duplicates().reset_index()
df = df.head(20000)

df['title'] = df['title'] + " by " + df['author']

import ast
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: [i.replace(" ","") for i in x])
df['characters'] = df['characters'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: [i.replace(" ","") for i in x])
df['awards'] = df['awards'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: [i.replace(" ","") for i in x])
df['setting'] = df['setting'].apply(lambda x: ast.literal_eval(x)).apply(lambda x: [i.replace(" ","") for i in x])

df['author'] = df['author'].apply(lambda x: x.split() if isinstance(x, str) else [])
df['description'] = df['description'].apply(lambda x: x.split() if isinstance(x, str) else [])
df['language'] = df['language'].apply(lambda x: x.split() if isinstance(x, str) else [])
df['bookFormat'] = df['bookFormat'].apply(lambda x: x.split() if isinstance(x, str) else [])
df['publisher'] = df['publisher'].apply(lambda x: x.split() if isinstance(x, str) else [])

df['tags'] = df['author'] + df['description'] + df['language'] + df['genres'] + df['characters'] + df['bookFormat'] + df['publisher'] + df['awards'] + df['setting']

book_df = df[['title', 'tags', 'coverImg']]

book_df['tags'] = book_df['tags'].apply(lambda x: " ".join(x))
book_df['tags'] = book_df['tags'].apply(lambda x: x.lower())

import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def nlp_preprocessing(s):
    s = str(s)
    s = re.sub('[^a-zA-Z]', ' ', s)
    s = s.lower()
    s = s.split()
    return " ".join([stemmer.stem(word) for word in s])

df['tags'] = df['tags'].apply(nlp_preprocessing)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1000, stop_words='english')

vectors = cv.fit_transform(book_df['tags']).toarray()

from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors()
neigh.fit(vectors)

def recommend_book_knn(book, n):
    book_index = df[df['title']==book].index[0]
    similar_books = neigh.kneighbors([vectors[book_index]],n+1)
    return [df.iloc[i].title for i in similar_books[1][0][1:]]