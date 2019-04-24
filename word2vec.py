# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:14:50 2018

@author: Dhruvin
"""

from gensim.models import Word2Vec
from utils import load, save
import numpy as np

abstracts = load('./abstracts.p')
abstracts = [abstract.split() for abstract in abstracts]

#train model
#size: size of vectors
#window: maximum distance between the current and predicted word within a sentence
#min_count: 
model = Word2Vec(abstracts, size=50, window=5, min_count=5, workers=4)
model.save("word2vec.model")

text_vectors = []
for abstract in abstracts:
    vector = np.zeros(50,)
    vector_length = len(abstract)
    #print(vector_length)
    
    # get vector based on all the words in the abstracts
    for word in abstract:
        if word in model.wv:
            vector = vector+model.wv[word]
        else:
            vector_length -= 1
    
    vector = vector/vector_length
    #vector = np.multiply(vector,100)
    text_vectors.append(vector)

save('./text_vectors.p',text_vectors)