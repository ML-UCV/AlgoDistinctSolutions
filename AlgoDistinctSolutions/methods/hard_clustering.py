import numpy as np
from typing import Literal
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

class HardClustering():
    def fit_predict(self, embeddings:np.ndarray, k: int, clustering_algo_name: Literal['kmeans', 'spectral_clustering', 'agglomerative_clustering'], **clustering_args):
        if clustering_algo_name == 'kmeans':
            clustering_algo = KMeans(k, **clustering_args)
        elif clustering_algo_name == 'spectral_clustering':
            clustering_algo = SpectralClustering(k, **clustering_args)
        elif clustering_algo_name == 'agglomerative_clustering':
            clustering_algo = AgglomerativeClustering(k, **clustering_args)
        else:
            raise Exception("Clustering algorithm is not valid")
    
        return clustering_algo.fit_predict(embeddings)



    