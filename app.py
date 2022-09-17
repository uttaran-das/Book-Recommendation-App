# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:25:21 2022

@author: user
"""

import pandas as pd
small_df = pd.read_csv('small_df_nlp.csv')
vectors = pd.read_csv('vectors_nlp.csv').iloc[:,1:]

books = small_df['title']

small_df['coverImg'] = small_df['coverImg'].fillna('https://bitsofco.de/content/images/2018/12/broken-1.png')

import streamlit as st

st.set_page_config(page_title="Book Recommendation App by Uttaran Das", page_icon='book_image.jpg')
st.title('BOOK RECOMMENDATION APP')

option = st.selectbox('Search for or choose a book from the list so that we can show you similar books',books)

st.write('Current Selection :', option)

st.image(small_df.iloc[small_df[small_df['title']==option].index[0]].coverImg, use_column_width='always')
url = "https://www.google.com/search?q="
for k in option.split():
    url += k+"+"
st.write("[Search Google](%s)" % url)

number = st.number_input('Number of recommendation', min_value=1, max_value=100, value=5)
st.write('The current number is ', number)

import pickle
neigh = pickle.load(open('book_recommendation_model_nlp.sav','rb'))

def recommend_book_knn(book, n):
    book_index = small_df[small_df['title']==book].index[0]
    similar_books = neigh.kneighbors([vectors.iloc[book_index]],n+1)
    return [[small_df.iloc[i].title, small_df.iloc[i].coverImg] for i in similar_books[1][0][1:]]

if st.button('Recommend Books'):
    recommendations = recommend_book_knn(option, number)
    c = 0
    for i in range(0,(number+1)//2+1):
        cols = st.columns(2)
        for j in range(0,2):
            if c==number:
                continue
            cols[j].text(recommendations[c][0])
            cols[j].image(recommendations[c][1],use_column_width='always')
            url = "https://www.google.com/search?q="
            for k in recommendations[c][0].split():
                url += k+"+"
            cols[j].write("[Search Google](%s)" % url)
            c += 1