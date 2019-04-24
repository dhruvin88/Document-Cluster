#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 22:30:50 2018

@author: dhruvinpatel
"""

from utils import load, cal_sse
from kmean import kmeans
import matplotlib.pyplot as plt

text_vectors = load('./text_vectors.p')

#find the min and max values from all the vectors
maxValue = 0
minValue = 0
for i in range(len(text_vectors)):
    localm = max(text_vectors[i])
    localmin = min(text_vectors[i])
    if localm > maxValue:
        maxValue = localm
    if localmin < minValue:
        minValue = localmin

k_values = [5,7,10,12,15]
sse = []
for k in k_values:
    clusters, centroids = kmeans(k, text_vectors,maxValue, minValue, 100)
    sse_value = cal_sse(clusters, centroids)
    print("SSE Score for K as "+str(k)+":\t"+str(sse_value))
    sse.append(sse_value)

plt.plot(k_values, sse)
plt.xlabel('K Values')
plt.ylabel('SSE Scores')
plt.title('SEE of K-Means')
plt.grid(True)
plt.savefig("SEE of K-Means.png")
plt.show()