import multiprocessing
from tqdm import tqdm
from typing import Any, Union, Iterable
import json
import parquet
import pandas as pd
import numpy as np
from itertools import chain, combinations, product, permutations
from sklearn.metrics import f1_score
from itertools import product

def _func_with_idx(el_with_func_idx):
    idx, el, func = el_with_func_idx

    return (idx, func(el))


def tqdm_multiprocess_map(func, elements:list[Any], max_workers:int, chunksize:int):

    elements_with_idx = [(idx, el, func) for idx, el in enumerate(elements)]

    with multiprocessing.Pool(max_workers) as pool:
        processed_with_idx = list(tqdm(pool.imap_unordered(_func_with_idx, elements_with_idx, chunksize = chunksize), total= len(elements_with_idx)))

    processed_with_idx = sorted(processed_with_idx, key = lambda x: x[0])

    return [p for _, p in processed_with_idx]

def to_batches(iterable: Iterable[Any], batch_size) -> Iterable[list[Any]]:
    batch = []
    for el in iterable:
        batch.append(el)
        if len(batch) == batch_size:
            yield batch
            batch = []

    if len(batch) > 0:
        yield batch

def load_dataset_with_embeddings(dataset_path:str, embeddings_path: dict[str, str] = None):
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf8') as fp:
        dataset = json.load(fp)

    dataset_map_id_to_index = {}

    for idx, s in enumerate(dataset):
        dataset_map_id_to_index[s['id']] = idx

    if embeddings_path is not None:
        for e_name, e_path in embeddings_path.items():
            emb_data = pd.read_parquet(e_path)

            for _, emb_row in emb_data.iterrows():
                idx, emb = emb_row 
                sample_id = dataset_map_id_to_index[idx]

                sample = dataset[sample_id]

                if sample['id'] != idx:
                    raise Exception ("Not the same id!")

                if 'embeddings' not in sample:
                    sample['embeddings'] = {}
                
                sample['embeddings'][e_name] = emb

    dataset  = [d for d in dataset if 'embeddings' in d]      
    dataset = [d for d in dataset if len(d['embeddings']) == len(embeddings_path.keys())]


    dataset_per_problem = {}

    for d in dataset:
        if d['problem'] not in dataset_per_problem:
            dataset_per_problem[d['problem']] = []
        dataset_per_problem[d['problem']].append(d)

    return dataset_per_problem
    
def shuffle_arrays(*arr):
    list_arr = list(arr)

    anchor_arr = list_arr[0]
    n_rows = len(anchor_arr)

    for a in list_arr:
        if len(a) != n_rows:
            raise Exception("Arrays have different row sizes")
    
    indices = np.array(range(n_rows))
    np.random.shuffle(indices)

    return [a[indices] for a in list_arr]

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))

def all_products(iterable):
    s = list(iterable)
    return product(s, repeat=len(s))

def find_cluster_labels_best_f1_score(true_labels, predicted_labels):
    if len(true_labels) != len(predicted_labels):
        raise Exception("Should have the same number of samples")
    
    all_true_labels = list(set(true_labels))
    all_predicted_labels = list(set(predicted_labels))

    # if len(all_true_labels) != len(all_predicted_labels):
    #     raise Exception("Should have the same number of labels")
    
    best_score = -1
    for p in permutations(all_predicted_labels):
        mapping = {k:v for k,v in zip(p, all_true_labels)}

        permutated_predicted_labels = list(map(lambda x: mapping[x], predicted_labels))

        score = f1_score(true_labels, permutated_predicted_labels, average='macro')
        if score > best_score:
            best_score = score
            best_predicted_labels = permutated_predicted_labels
    
    return best_predicted_labels
