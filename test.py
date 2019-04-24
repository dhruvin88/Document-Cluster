#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:44:00 2018

@author: dhruvinpatel
"""

from sklearn.cluster import DBSCAN
import numpy as np
from utils import load, cal_sse
import matplotlib.pyplot as plt

text_vectors = load('text_vectors.p')
clustering = DBSCAN(eps=.5, min_samples=10,metric='cosine').fit(text_vectors)
labels = clustering.labels_
n_labels = set(labels)

clusters = {}
for n in n_labels:
    if n == -1:
        continue
    else:
        clusters[n] = []
        
for i in range(len(labels)):
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
print(sse)