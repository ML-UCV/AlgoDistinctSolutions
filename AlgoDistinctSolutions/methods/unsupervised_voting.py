import copy
import numpy as np
from typing import Literal
from xgboost import XGBClassifier

class _Node:
    def __init__(self, cluster):
        self.children = []
        self.cluster = cluster
        self.count = 0

class _MultiAssigmentClustering():
    def __init__(self, num_views:int, num_k:int):
        self.num_views = num_views
        self.num_k = num_k

        self.found_k = 0
        self.current_score = 0 
        self.current_matching = [[]]
        self.best_score = -1
        self.best_matchings = None
        self.is_used = [[0]*num_k for _ in range(num_views)]

        self.root = _Node(-1)

    def _add_to_trie(self, root, v, num_views, value):
        next_node = root
        for level in range(num_views):
            view_cluster = v[level]

            found_node = None
            for child in next_node.children:
                if child.cluster == view_cluster:
                    found_node = child
                    break

            if found_node is None:
                found_node = _Node(view_cluster)
                next_node.children.append(found_node)
            next_node = found_node

        next_node.count = value

    def _fit(self, initial_root, root, level):
        if len(root.children) == 0:
            self.found_k += 1
            self.current_score += root.count
            self.current_matching.append([])

            if self.found_k == self.num_k:
                if self.best_score < self.current_score:
                    self.best_score = self.current_score
                    self.best_matchings = copy.deepcopy(self.current_matching)

                self.current_score -= root.count
                self.found_k -= 1
                self.current_matching.pop()
                return
            else:
                next_root = None
                i = 0
                for child in initial_root.children:
                    if i == self.found_k:
                        next_root = child
                        break
                    i += 1
                if next_root is not None:
                    self.current_matching[-1].append(next_root.cluster)
                    self._fit(initial_root, next_root, 1)
                self.current_score -= root.count
                self.found_k -= 1
                self.current_matching.pop()

        for child in root.children:
            if not self.is_used[level][child.cluster]:
                self.current_matching[-1].append(child.cluster)
                self.is_used[level][child.cluster] = True

                self._fit(initial_root, child, level + 1)
                self.is_used[level][child.cluster] = False
                self.current_matching[-1].pop()

    def fit_predict(self, clusters_per_view:list[list[int]], frequency: list[int]):
        for clusters, v in zip(clusters_per_view, frequency):
            self._add_to_trie(self.root, clusters, self.num_views, v)

        self.current_matching[-1].append(self.root.children[0].cluster)
        self._fit(self.root, self.root.children[0], 1)

        if self.best_score != -1:
            self.best_matchings = self.best_matchings[:-1]
        return self.best_score, self.best_matchings


class _CoTraining():
    def __init__(self, threshold:int = 20):
        self.threshold = threshold

    def _get_agreement(self, predictions):
        p = list(set(predictions))

        if len(p) == 1:
            return p[0]
        else:
            return None
    
    def fit_predict(self, emb_views:list[np.ndarray], subset_indices:list[int], subset_labels:np.ndarray, unlabelled_indices:list[int]):
        print('Doing cotraining')
        subset_indices = [si for si in subset_indices]
        subset_labels = [sl for sl in subset_labels]

        unlabelled_indices = [ui for ui in unlabelled_indices]

        while len(unlabelled_indices) > 0:
            classifiers = [XGBClassifier(seed = 42) for v in emb_views]

            predicted_solutions =[]
            for v in range(len(classifiers)):
                emb_v = [emb_views[v][si] for si in subset_indices]
                classifiers[v].fit(np.array(emb_v), np.array(subset_labels))

                unl_v = [emb_views[v][ui] for ui in unlabelled_indices]
                predicted_solutions.append(classifiers[v].predict(np.array(unl_v)))

            predicted_solutions_views = np.array(predicted_solutions).T

            new_unlabelled_indices = []

            for idx, p_s_v in enumerate(predicted_solutions_views):
                agg = self._get_agreement(p_s_v)
                if agg is None:
                    new_unlabelled_indices.append(idx)
                else:
                    subset_indices.append(idx)
                    subset_labels.append(agg)
            
            if len(unlabelled_indices) - len(new_unlabelled_indices) < self.threshold:
                print('Stopping from cotraining')
                break
            else:
                print(f'{len(unlabelled_indices) - len(new_unlabelled_indices)} samples added via cotraining. {len(new_unlabelled_indices)} left')
            unlabelled_indices = new_unlabelled_indices

        return subset_labels, subset_indices


class UnsupervisedVoting():
    def fit_predict(self,
                    embeddings_view:list[np.ndarray], 
                    clustering_view: np.ndarray,
                    k:int):
        mac = _MultiAssigmentClustering(len(embeddings_view), k)

        groups = {}

        for idx, c_row in enumerate(clustering_view):
            tuple_c_row = tuple(c_row)
            if tuple_c_row not in groups:
                groups[tuple_c_row] = []

            groups[tuple_c_row].append(idx)
        
        best_score, best_matchings = mac.fit_predict(list(groups.keys()), [len(v) for v in groups.values()])

        if best_score == -1:
            return None
        else:
            subset_indices = [ind for m in best_matchings for ind in groups[tuple(m)]]
            subset_clusters = clustering_view[subset_indices, 0]
            
            subset_indices_as_set = set(subset_indices)
            unlabelled_indices = [ind for ind in range(len(embeddings_view[0])) if ind not in subset_indices_as_set]

            cotraining = _CoTraining()
            predicted_labels_cotraining, indices_cotraining =  cotraining.fit_predict(embeddings_view, subset_indices, subset_clusters, unlabelled_indices)

            return subset_clusters, subset_indices, predicted_labels_cotraining, indices_cotraining

            





