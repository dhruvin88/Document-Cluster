#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:30:50 2018

@author: dhruvinpatel
"""

from utils import load, save, cal_sse, cosine_simailarity
from dbscan import dbscan
import numpy as np
#import matplotlib.pyplot as plt


#####
'''
text_vectors = load('./text_vectors.p')
top = {}
for i in range(len(text_vectors)):
    score = []
    for j in range(len(text_vectors)):
        if i != j:
            score.append((1-cosine_simailarity(text_vectors[i],text_vectors[j])))
    score.sort()
    top[i] = score[:10]
'''
####
text_vectors = load('./text_vectors.p')
text_vectors = text_vectors[:1000]
labels = dbscan(text_vectors, eps=.1, minPts=10)
save('labels.p',labels)

#get unique labels
n_labels = set(labels)

#create empty clusters
clusters = {}
for n in n_labels:
    if n == -1:
        continue
    else:
        clusters[n] = []

#assign each abstract to a vector based on the labels       
for i in range(len(labels)):
    #ignore the noise points
    if labels[i] == -1:
        continue
    else:
       clist = clusters[labels[i]]
       clist.append(text_vectors[i])
       clusters[labels[i]] = clist

cluster_center = []
for cluster in clusters.keys():
    cluster_center.append(np.mean(clusters[cluster], axis=0))
    
    
sse = cal_sse(clusters,cluster_center)
print('SEE of DBSCAN: '+str(sse))
