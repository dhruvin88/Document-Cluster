# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:57:16 2018

@author: Dhruvin Patel
"""
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from utils import load, save

abstracts = load('./abstracts.p')

# iterable list of tagged abstracts
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=
                              [str(i)]) for i, _d in enumerate(abstracts)]

#alpha is the initial learning rate
max_epochs = 100
vec_size = 50
alpha = 0.025

#dm: 1 = using distributed memory algorithm
model = Doc2Vec(vector_size= vec_size,
                alpha = alpha, 
                min_alpha = 0.00025,
                min_count = 1,
                dm = 1)
#builds vocab based on the abstracts
model.build_vocab(tagged_data)

#train model
for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples = model.corpus_count,
                epochs = model.epochs)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
print("Model Saved")

#save document vectors
vectors = []
for i in range(len(abstracts)):
    vectors.append(model.docvecs[i])

save('./text_vectors.p',vectors)