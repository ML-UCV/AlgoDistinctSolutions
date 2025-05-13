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
import time
from sklearn.metrics import classification_report, confusion_matrix
from scipy.optimize import linear_sum_assignment

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
    dataset = [d for d in dataset if len(set(embeddings_path.keys()).difference(d['embeddings'].keys())) == 0]

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

    return [[a[idx] for idx in indices] for a in list_arr]

def powerset(iterable, max_size = -1):
    s = list(iterable)
    if max_size == -1:
        return chain.from_iterable(combinations(s, r) for r in range(2, len(s)+1))
    else:
        return chain.from_iterable(combinations(s, r) for r in range(2, max_size + 1))

def all_products(iterable, repeat):
    s = list(iterable)
    return product(s, repeat=repeat)

def find_best_cluster_mapping(true_labels:list[int], predicted_labels:list[int]):
    if len(true_labels) != len(predicted_labels):
        raise Exception("Should have the same number of samples")
    
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)

    cm = confusion_matrix(true_labels, predicted_labels)

    matching = linear_sum_assignment(cm, maximize = True)

    best_mapping = {y:x for x,y in zip(matching[0], matching[1])}

    return best_mapping

def execute_function_with_retries(func, max_retries:int = 3):
    current_retries = 0

    while current_retries < max_retries:
        try:
            result = func()
            return result
        except Exception as e:
            print(f"Exception {e}. Retrying...")
            time.sleep(current_retries * 1)
            
            current_retries+=1
    
    raise Exception("Too many retries!")