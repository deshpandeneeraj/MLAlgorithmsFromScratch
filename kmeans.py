import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
warnings.simplefilter("ignore")

class KMeans:
    """
    K-Means Clustering model
    @Author : Neeraj Deshpande
    StartDate : 26th July 2020
    Distance metric used is Euclidean distance
    Future Improvements:
        - Multiple distance metrics
    """
    def __init__(self, k = 3):
        self.k = k
        self.centroids = np.zeros(shape=(k,), dtype=np.float32)
        self.clusters = None
        self.data = None

    def calculate_clusters(self):
        current_centroids = self.centroids
        current_clusters = [[] for i in range(len(current_centroids))]
        for index, element in enumerate(self.data):
            current_distances = []
            if type(element) is not list and type(element) is not np.ndarray: element = [element]
            for c_index, centroid in enumerate(current_centroids):
                if type(centroid) is not list and type(centroid) is not np.ndarray: centroid = [centroid]
                current_distances.append([c_index, self.__class__.calculate_distance(element, centroid)])
            current_clusters[min(current_distances, key = lambda x: x[-1])[0]].append(index)
        return current_clusters

    def calculate_centroids(self, clusters:np.ndarray):
        current_centroids = []
        for cluster in clusters:
            centroid = []
            length = len(cluster)
            cluster_data = self.data[cluster,:]
            centroid = cluster_data.sum(axis=0) / length
            current_centroids.append(centroid)
        np.sort(current_centroids, axis=0)
        return current_centroids

    @staticmethod
    def calculate_distance(element, centroid):
        try:
            difference = 0
            element = element.ravel()
            #Check if input shape matches that of training data shape
            assert element.shape[-1] == centroid.shape[-1]
            for col in range(centroid.shape[-1]):
                #Euclidean distance
                difference += (element[col] - centroid[col]) ** 2
            distance = np.sqrt(difference)
            return distance
        except AssertionError:
            print(f"calculate_distance ERROR: {element.shape} vs {centroid.shape}")

    def initialize_centroids(self):
        self.centroids = self.data[:self.k,:]
        self.centroids = np.sort(self.centroids, axis=0)

    def fit(self, data):
        if type(data) is not np.ndarray:
            if len(data) > 1:
                self.data = np.array(data)
            else:
                raise ValueError(f'Size of data input is wrong: {len(data)}')
        else:
            self.data = data
        self.initialize_centroids()
        i = 0
        for i in tqdm(range(100)):
            clusters = self.calculate_clusters()
            if np.array_equal(clusters, self.clusters):
                break
            else:
                self.clusters = clusters
                self.centroids = self.calculate_centroids(clusters)
        if i <1000:print(f'Clusters found at iteration number {i}')
        else: print('Clusters not found (itertation limit exceeded)')

    def predict_one(self, X):
        current_distances = []
        if type(X) is not list and type(X) is not np.ndarray: X = [X]
        for c_index, centroid in enumerate(self.centroids):
            if type(centroid) is not list and type(centroid) is not np.ndarray: centroid = [centroid]
            current_distances.append([c_index, self.__class__.calculate_distance(element, centroid)])
        cluster_out =  min(current_distances, key = lambda x: x[-1])[0]
        return cluster_out, self.centroids[cluster_out]

    def predict_many(self, X):
        output_cluster, output_centroid = [], []
        for element in X:
            clust, cent = self.predict_one(element)
            output_cluster.append(clust)
            output_centroid.append(cent)
        return output_cluster, output_centroid

    def get_clusters(self):
        return self.clusters, self.centroids
if __name__ == '__main__':
    data = np.array([[18,73,75,57],[18,79,85,75],[23,70,70,52],[20,55,55,55],[22,85,86,87],[19,91,90,89],[20,70,65,60],[21,53,56,59],[47,75,76,77]])
    kmeans = KMeans(k=3)
    kmeans.fit(data)
    clusters, centroids = kmeans.get_clusters()
    print(f'Clusters = {clusters}\n Centroids = {centroids}')

# np. ndarray. all()
