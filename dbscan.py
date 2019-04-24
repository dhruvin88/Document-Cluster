#!/usr/bin/env python3
"""
Created on Tue Nov 20 17:37:46 2018

@author: Dhruvin Patel
"""
import numpy as np
from utils import cosine_simailarity
from tqdm import tqdm 

# return all points within p's eps-neighborhood 
def region_query(abstracts, p, eps):
    neighbors = []
    #print('Finding neighbors')
    for i in range(len(abstracts)):
        # if the distance is below the threshold, add it to the neighbors list
        if (1-cosine_simailarity(abstracts[p],abstracts[i])) < eps:
            neighbors.append(i)
    return neighbors


def expand_cluster(abstracts, labels, p, neighbor_pts, cluster_n, eps, minPts):
    # Assign the cluster label to the seed point
    labels[p] = cluster_n
    
    i = 0
    print('expanding cluster')
    while (i < len(neighbor_pts)):
        #get next point from the list
        point = neighbor_pts[i]
        
        # if labeled as noise change to point in cluster 
        if labels[point] == -1:
            labels[point] = cluster_n
            
        # if abstract is not claimed, claim it as part of cluster_n
        elif labels[point] == -2:
            labels[point] = cluster_n
            
            # find all neighor to abstract
            new_neighbor_pts = region_query(abstracts, point, eps)
            
            # if the abstracts has at least minPts then all points 
            # to the neighbors list
            if len(new_neighbor_pts) >= minPts:
                neighbor_pts += new_neighbor_pts
        i += 1

def dbscan(abstracts, eps, minPts):    
    # list holds the assignment for just abstract
    # -1 indicates a noise points
    # -2 means the point hasn't been visited yet
    labels = [-2]*len(abstracts)
        
    cluster_n = 0
    for p in tqdm(range(len(abstracts))):
        #check if the point has not been seen
        if not labels[p] == -2:
            continue
        
        # find all abstracts that are neighboring points
        neighbor_pts = region_query(abstracts, p, eps)
        
        if len(neighbor_pts) < minPts:
            labels[p] = -1
        else:
            expand_cluster(abstracts, labels, p, neighbor_pts, cluster_n, eps, minPts)
            cluster_n += 1
        #print(p)
    return labels