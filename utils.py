#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:30:50 2018

@author: dhruvinpatel
"""
import re
import math
import glob
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tqdm import tqdm 
import pickle

#returns all files in the path
def get_files(dir_path):
    files = glob.glob(dir_path+"/*/*/*.txt")
    #print(len(files))
    return files

# cleans abstracts
def clean(text):
    #remove newline and tabs from string
    text = re.sub('\s+',' ',text)
    #remove 'Abstract : '
    text =  text[11:]
    text = re.sub(r'\W+ ',' ',text)
    
    #remove all non letter words
    #text = re.sub(r'[^a-zA-Z\s]',' ', text)
    
    #remove puncuations
    text = re.sub(r'[^\w\s]',' ',text)
    text = text.lower()
    #Stem and Lemmatize each word
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    #remove stop words that are not significant in finding simaiarlity
    stop_words = set(stopwords.words('english'))
      
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    if len(filtered_sentence) < 5:
        return ''
    text = ''
    for word in filtered_sentence:
        word = stemmer.stem(word)
        word = lemmatizer.lemmatize(word)
        text += word + ' '
    #remove empty abstract with just n in it
    if text == "n ":
        text = ''
        return text
    else:
        return text

def get_abstracts(file_list):
    abstracts = []
    files = []
    for file in file_list:   
        content = open(file,'r',encoding="utf-8",errors='ignore')
        copy = False
        abstract = ''
        #open each file and just get the content
        for line in content.readlines():
            if "Abstract" in line:
                copy = True
            if copy:
                abstract+= line + ' '
        files.append(file)
        abstracts.append(abstract)
    
    #filter out the text per each article
    new_list = []
    new_files = []
    print('Cleaning abstracts')
    for i in tqdm(range(len(abstracts))):      
        if not "Not Available" in abstracts[i]:
            text = clean(abstracts[i])
            if text != '':
                new_files.append(file)
                new_list.append(text)
    
    return new_files,new_list 

#count the number of words in the abstract
def get_word_counts(text):
    word_count = {}
    words = text.split()
    for word in words:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    return word_count

#get tf for each word in the abstract
def get_tf(word_count,text):
    tf = {}
    total_words = len(text.split())
    for word, count in word_count.items():
        tf[word] = 1 + math.log10(count/total_words)
    return tf

#get idf of words in the abstract
def get_idf(abstracts):
    idf = {}
    N = len(abstracts)
    for abstract in abstracts:
        seen = set()
        words = abstract.split()
        for word in words:
            if word not in seen and word in idf:
                idf[word] += 1
                seen.add(word)
            elif word not in seen:
                idf[word] = 1
                seen.add(word)
    for word, count in idf.items():
        idf[word] = math.log10(N/count)
    voc = list(idf.keys())
    return idf,voc

#returns tfidf vector
def get_tfidf_vector(text,idf):
    #zero vector the size of the voc
    tf_vector = []
    #create dict for each abstract
    word_dic = get_word_counts(text) 
    #get tf dictionary for the abstact
    tf = get_tf(word_dic, text)
    i = 0 
    #for each of the words in the voc
    #get the tf and idf values and save it to the vector
    for word in idf.keys():
        if word in word_dic:
            tf_vector.append(round(tf[word]*idf[word],4))
        else:
            tf_vector.append(0)
        i += 1
    return tf_vector

def cosine_simailarity(vector1,vector2):
    #computes cosine similarity of vector1 to vector2
    vector1 = np.asarray(vector1)
    vector2 = np.asarray(vector2)
    
    sumyy = np.sum(np.multiply(vector2,vector2))
    sumxx = np.sum(np.multiply(vector1,vector1))
    sumxy = np.dot(vector1, vector2.T)
    return sumxy/np.sqrt(np.multiply(sumxx, sumyy))

def distance_matrix(abstracts):
    print('Creating dictionary for cos similarity between abstracts')
    cos_dic = {}
    for i in tqdm(range(len(abstracts))):
        for j in range(len(abstracts)):
            if (i,j) not in cos_dic and (j,i) not in cos_dic:
                sim = distance(abstracts[i], abstracts[j])
                cos_dic[(i,j)] = sim
                #print(i,j)
    return cos_dic

def cal_sse(clusters, centroids):
    sse = 0
    for i in range(len(centroids)):
        cluster = clusters[i]
        total_distance = 0
        #for all the points get the cos distance and get it up
        for vector in cluster:
            total_distance += ((1-cosine_simailarity(vector, centroids[i]))**2)
        sse += total_distance
    return sse

#euclidean distance
def distance(vector1, vector2):
    return np.sqrt(np.sum((vector1-vector2)**2))

#save data
def save(filename,item):
    with open(filename, 'wb') as f:
        pickle.dump(item, f)
    print('saved '+filename)

#load data
def load(filename):
    with open(filename, 'rb') as f:
        item = pickle.load(f)
        print('loaded '+ filename)
    return item