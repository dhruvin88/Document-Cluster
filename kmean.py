#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import cosine_simailarity
import numpy as np
import timeit

#maxValue = 4.7
#minValue = -8.9

def kmeans(num_k, abstracts, maxValue, minValue, maxitr):
    maxrun = 0
    #create n number of centroids
    centroids = []
    for i in range(num_k):
        vector_size = len(abstracts[0])
        rand = np.random.uniform(low=minValue, high=maxValue, size=(vector_size,))
        #rand = np.round(rand, decimals=4)
        centroids.append(rand)
     
    
    repeat = True
    while(repeat):       
        clusters = {}
        for i in range(num_k):
            clusters[i] = []
            new_centroids = [] 
            
        start = timeit.default_timer()
        #assigns all points to closest centroid based on cosine
        for abstract in abstracts:
            cos = []

            #abstract = get_tfidf_vector(abstract, idf)
            for centroid in centroids:
                cos.append(cosine_simailarity(abstract, centroid))
            #find the max cos similarity
            max_cost_index = cos.index(max(cos))
             
            #add vector to the cluster
            cluster = clusters[max_cost_index]
            cluster.append(abstract)
            clusters[max_cost_index] = cluster
            
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        
        #recompute the centroid of each cluster until no changes
        new_centroids = []
        for i in range(num_k):
            vector_size = len(abstracts[0])
            new_vector = np.zeros(vector_size)
            cluster = clusters[i]
            for vector in cluster:
                new_vector = new_vector+vector
            new_vector = new_vector/len(cluster)
            #new_vector = np.round(new_vector, decimals=4)
            new_centroids.append(new_vector)
            print(np.sum(np.subtract(new_vector,centroids[i])))
        
        #difference = []
        #check if the new centroids are the same
        for i in range(num_k):
            if list(centroids[i]) != list(new_centroids[i]):
                break
            elif i+1 == num_k and list(centroids[i]) == list(new_centroids[i]):
                repeat = False
        if maxrun == maxitr:
            repeat = False
        
        # set new centroids
        centroids = new_centroids
        
        maxrun +=1
    return clusters, centroids