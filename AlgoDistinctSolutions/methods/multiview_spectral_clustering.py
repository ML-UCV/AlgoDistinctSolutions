import numpy as np
from multiview import MVSC

class MultiViewSpectralClustering():
    def fit_predict(self, embeddings_view:list[np.ndarray], k: int, **clustering_args):
        clustering_algo = MVSC(k, **clustering_args)

        labels, _, _, _ = clustering_algo.fit_transform(embeddings_view, np.array([False for _ in embeddings_view]))
        
        return labels



    