# 17CH10051
# Yash Khandelwal
# Assignment Number 4
# python3 <program name>

import pandas as pd
import numpy as np
import math

def choose_random_points(data):
    data = data[np.random.choice(data.shape[0], 3, replace=False), :]

    return data[:, :4]

def find_cluster_alloted(data, centers, k):
    dist = {}
    for i in range(k):
        dist[i] = (data[:, :4] - centers[i, :4])**2
        dist[i] = np.sum(dist[i], axis=1)
        
        dist[i] = [math.sqrt(j) for j in dist[i]]

    combined_distance = np.array(list(zip(dist[0], dist[1], dist[2])))
    clusters = np.argmin(combined_distance, axis = 1)

    return clusters

def find_mean_cluster_centers(data, cluster_alloted, centers, k):
    
    index = []
    for i in range(k):

        lst = data[np.where(cluster_alloted == i)]
        centers[i][:4] = np.mean(lst[:, :4], axis = 0)
    return centers

def find_final_clusters(data, centers, no_of_iterations, k):

    for i in range(no_of_iterations):
        cluster_alloted = find_cluster_alloted(data, centers, k)
        centers = find_mean_cluster_centers(data, cluster_alloted, centers, k)

    return (centers, cluster_alloted)

def find_jacquard_matrix(data, centers, cluster_alloted, labels):

    jacquard = []

    for label in labels:

        temp = []
        for set_elem in set(cluster_alloted):
            predictions = cluster_alloted.copy()
            predictions[predictions == set_elem] = label
            ground_truth = list(data[:, 4])

            intersection = 0
            union = 0

            for i in range(len(predictions)):
                if predictions[i] == ground_truth[i] == label:
                    intersection += 1
                if predictions[i] == label or ground_truth[i] == label:
                    union += 1
            temp.append(1 - (intersection) / (union))

        jacquard.append(temp)

    return jacquard

def print_jaccquard_matrix(centers, labels, jacquard):

    print('printing final mean centers')
    for i in range(len(centers)):
        centers[i] = [ '%.6f' % elem for elem in centers[i]]
        print("clusters", i, ":", centers[i])
    
    print('\n')
    print("printing jacquard distance matrix")
    print('\t',list(labels))
    
    temp = 0
    for i in jacquard:
        temp += 1
        i = [ '%.10f' % elem for elem in i ]
        print("cluster", temp, ":", i)

def main():

    data = pd.read_csv('data4_19.csv', header = None)
    data = np.array(data)

    centers = choose_random_points(data)
    labels = set(data[:, 4])

    # hyperparameters
    k = 3
    no_of_iterations = 10

    centers, cluster_alloted = find_final_clusters(data, centers, no_of_iterations, k)

    jacquard = find_jacquard_matrix(data, centers, cluster_alloted.astype(str), labels)

    print_jaccquard_matrix(centers, labels, jacquard)

if __name__ == "__main__":
    main()