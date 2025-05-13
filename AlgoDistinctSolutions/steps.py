import scipy.stats
from sklearn.model_selection import train_test_split
import os
import json
import logging
from typing import Any
from model_facades.Word2VecFacade import Word2VecFacade
from model_facades.TfIdfFacade import TfidfFacade
from model_facades.SafeFacade import SafeFacade
from model_facades.UniXcoderFacade import UniXcoderFacade
from model_facades.CodeT5plusFacade import CodeT5plusFacade
from model_facades.OpenAIFacade import OpenAIFacade
from model_facades.MistralFacade import MistralFacade

from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from utils import load_dataset_with_embeddings, shuffle_arrays, find_best_cluster_mapping, powerset, all_products
import numpy as np
import matplotlib.pyplot as plt
import scipy
import copy
from methods.hard_clustering import HardClustering
from methods.multiview_spectral_clustering import MultiViewSpectralClustering
from methods.unsupervised_voting import UnsupervisedVoting
import pandas as pd

logger = logging.getLogger()

def split_dataset(dataset_path: str,
                  test_ratio: float = 0.1,
                  random_state = 42) -> None:
    
    def group_submissions_by_dict(dataset:list[dict[str, Any]]) -> dict[str, list[dict[str,Any]]]:
        submissions_per_problem = {}

        for submission in dataset:
            problem = submission['problem']
            if problem not in submissions_per_problem:
                submissions_per_problem[problem] = []

            submissions_per_problem[problem].append(submission)
        return submissions_per_problem
    
    def print_small_statistics(dataset:list[dict[str, Any]]):
        logger.debug("Small statistics about the dataset")
        submissions_per_problem = group_submissions_by_dict(dataset)
        logger.debug(f"Number of problems: {len(submissions_per_problem)}")
        
        for problem, submissions in submissions_per_problem.items():
            logger.debug(f"\t{problem}:")
            logger.debug(f"\t\tnumber submissions: {len(submissions)}")
            logger.debug(f"\t\talgorithmic solutions: {list(set([s['algorithmic_solution'] for s in submissions]))}")

    logger.info(f"Splitting dataset with test_ratio={test_ratio}.")

    if test_ratio is None or test_ratio == 0:
        return Exception("test_ratio must be greater than 0.")

    with open(os.path.join(dataset_path, 'dataset.json'), 'r', encoding='utf-8') as fp:
        dataset_info = json.load(fp)

    print_small_statistics(dataset_info)

    submissions_per_problem  = group_submissions_by_dict(dataset_info)

    train_dataset_info = []
    test_dataset_info = []

    for problem, submissions in submissions_per_problem.items(): 
        logger.info(f"Splitting problem {problem}...")

        train_submissions, test_submissions = train_test_split(submissions,
                                                                   stratify = [s['algorithmic_solution'] for s in submissions],
                                                                   test_size = test_ratio,
                                                                   random_state = random_state)
    
        train_dataset_info.extend(train_submissions)
        test_dataset_info.extend(test_submissions)

    with open(os.path.join(dataset_path, 'train.json'), 'w', encoding='utf-8') as fp:
        json.dump(train_dataset_info, fp)

    logger.info("Created train split!")
    print_small_statistics(train_dataset_info)

    with open(os.path.join(dataset_path, 'test.json'), 'w', encoding='utf-8') as fp:
        json.dump(test_dataset_info, fp)

    logger.info("Created test split!")  
    print_small_statistics(test_dataset_info)

def pretrain_embeddings(args: dict[str, Any]):
    dataset_info_path = args['dataset_info_path']
    preprocessing_workers = args.get('preprocessing_workers', 1)

    emb_model_mapping = {
        "w2v": Word2VecFacade(),
        "tfidf": TfidfFacade()
    }

    for emb_model_name, model_facade in emb_model_mapping.items():
        if emb_model_name in args:
            model_args = args[emb_model_name]
            destination_dir = model_args.pop('destination_dir')

            model_facade.pretrain(dataset_info_path, 
                                destination_dir,
                                preprocessing_workers,
                                **model_args)
        
def generate_embeddings(args: list[dict[str, Any]]):
    for args_dataset in args:
        dataset_info_path = args_dataset['dataset_info_path']
        preprocessing_workers = args_dataset.get('preprocessing_workers', 1)

        emb_model_mapping = {
        "w2v": Word2VecFacade(),
        "tfidf": TfidfFacade(),
        "safe": SafeFacade(),
        "unixcoder": UniXcoderFacade(),
        "codet5plus": CodeT5plusFacade(),
        "openai": OpenAIFacade(),
        "mistral": MistralFacade()
        }

        for emb_model_name, model_facade in emb_model_mapping.items():
            if emb_model_name in args_dataset:
                model_args = args_dataset[emb_model_name]
                destination_dir = model_args.pop('destination_dir')
                
                model_facade.generate_embeddings(dataset_info_path, 
                                destination_dir,
                                preprocessing_workers,
                                **model_args)
                
# def plot(train_dataset, test_dataset, embeddings_names:list[str], problem_name:str):
#     print(f"Plotting problem {problem_name}")
#     percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#     tries = 30

#     import scipy.stats as st
    
#     np.random.seed(24)

#     for e in embeddings_names:
#         print(f'\tPlotting embedding {e}')
#         train_X = np.array([s['embeddings'][e] for s in train_dataset])
#         train_Y = np.array([s['algorithmic_solution'] for s in train_dataset])

#         test_X = np.array([s['embeddings'][e] for s in test_dataset])
#         test_Y = np.array([s['algorithmic_solution'] for s in test_dataset])

#         percentiles_values = [int(len(train_X) * p) + 1 for p in percentiles]

#         mean_per_percentile = [0]
#         ci_l_per_percentile = [0]
#         ci_r_per_percentile = [0]

#         for p in percentiles_values:
#             scores_per_try = []
#             for _ in range(tries):
#                 train_X_shuffled, train_Y_shuffled = shuffle_arrays(train_X, train_Y)

#                 clf = XGBClassifier()
                
#                 clf.fit(train_X_shuffled[:p], train_Y_shuffled[:p])

#                 predictions = clf.predict(test_X)

#                 score = f1_score(test_Y, predictions, average = 'macro')

#                 scores_per_try.append(score)

#             scores_per_try  =np.array(scores_per_try)

#             mean_per_percentile.append(scores_per_try.mean())

#             ci_l, ci_r = scipy.stats.norm.interval(0.90, loc = np.mean(scores_per_try), scale = scipy.stats.sem(scores_per_try))

#             ci_l_per_percentile.append(ci_l)
#             ci_r_per_percentile.append(ci_r)
        
#         fig, ax = plt.subplots()
        
#         ax.plot([0] + percentiles, mean_per_percentile)
    
#         for idx in range(len(mean_per_percentile) - 1):
#             x_range = percentiles[idx : idx + 2]
#             ci_l = ci_l_per_percentile[idx : idx + 2]
#             ci_r = ci_r_per_percentile[idx : idx + 2]

#             ax.fill_between(x_range, ci_l, ci_r, color='b', alpha=.1)

#         ax.set_title(f'RF Prediction on {problem_name} using {e} embeddings')
#         ax.set_xlabel("Ratio of samples")
#         ax.set_xlabel("F1-score")


#         fig.savefig(f'Data/Plots/{problem_name}_{e}.png')

#         plt.close()

def plot_score_per_samples(args:dict[str,Any]):
    train_args = args['train']

    train_dataset_path = train_args['dataset_info_path']
    train_embeddings_path = train_args['embeddings_path']

    train_dataset = load_dataset_with_embeddings(train_dataset_path, train_embeddings_path)

    test_args = args['test']

    test_dataset_path = test_args['dataset_info_path']
    test_embeddings_path = test_args['embeddings_path']

    test_dataset = load_dataset_with_embeddings(test_dataset_path, test_embeddings_path)

    problems = list(train_dataset.keys())

    def plot(axes, train_dataset, test_dataset, embeddings_names:list[str], problem_name:str):
        print(f"Plotting problem {problem_name}")
        percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        for e in embeddings_names:
            print(f'\tPlotting embedding {e}')
            train_X = np.array([s['embeddings'][e] for s in train_dataset])
            train_Y = np.array([s['algorithmic_solution'] for s in train_dataset])

            test_X = np.array([s['embeddings'][e] for s in test_dataset])
            test_Y = np.array([s['algorithmic_solution'] for s in test_dataset])

            percentiles_values = [int(len(train_X) * p) + 1 for p in percentiles]


            label2id = {y:idx for idx, y in enumerate(set(train_Y))}
            id2label = {idx:y for y, idx in label2id.items()}

            scores = []

            for p in percentiles_values:
                _, x_percentile_samples, _, y_percentile_samples = train_test_split(train_X, train_Y, test_size=p, stratify=train_Y, random_state=42)

                clf = XGBClassifier()
                
                clf.fit(x_percentile_samples, [label2id[y] for y in y_percentile_samples])

                predictions = clf.predict(test_X)
                predictions = [id2label[y] for y in predictions]

                score = f1_score(test_Y, predictions, average = 'macro')

                scores.append(score)
            
            axes.plot(percentiles, scores, label = e)

        axes.set_title(problem_name)
        axes.set_xlabel("Ratio of samples")
        axes.set_xlabel("F1-score")

    
    fig, axes = plt.subplots(len(problems) // 5 + (len(problems) % 5 != 0), 5, figsize = (16, 9), sharex=True, sharey=True)

    for idx, p in enumerate(problems):
        problem_train_dataset = train_dataset[p]
        problem_test_dataset = test_dataset[p]
        
        plot(axes[idx // 5, idx % 5], problem_train_dataset, problem_test_dataset, list(train_embeddings_path.keys()), p)


    fig.suptitle('F1-scores per ratio of samples used as training')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'Data/Plots/plot.png')


def validate_hard_clustering(samples, embeddings, clustering_algos):
    print('Validating hard clustering')
    df = {"embeddings":[],
          "f1_score": [],
          "size":[]}
    
    true_labels = [s['algorithmic_solution'] for s in samples]
    label2id = {k:idx for idx, k in enumerate(set(true_labels))}
    id2label = {idx:k for k,idx in label2id.items()}

    true_labels = [label2id[l] for l in true_labels]
    k = len(set(true_labels))

    cache_embeddings_cluster = {}
    for e in embeddings:
        cache_embeddings_cluster[e] = {}
        samples_per_embeddings = [s['embeddings'][e] for s in samples]
        samples_per_embeddings = np.array(samples_per_embeddings)
        
        for c in clustering_algos:
            hc = HardClustering()
            predicted_labels = hc.fit_predict(samples_per_embeddings, k, c)

            cache_embeddings_cluster[e][c] = predicted_labels
            best_mapping =  find_best_cluster_mapping(true_labels, predicted_labels)

            predicted_mapped_labels = [best_mapping[l] for l in predicted_labels]

            f1_score_p = f1_score(true_labels, predicted_mapped_labels, average = 'macro')
            print(f"Using embedding {e} with clustering method {c} with f1-score {f1_score_p}")

            df['embeddings'].append(c)
            df['f1_score'].append(f1_score_p)
            df['size'].append(len(true_labels))
    df = pd.DataFrame(df)

    return cache_embeddings_cluster, df

def validate_multiview_spectral_clustering(samples, embeddings):
    print('Validating multiview spectral clustering')
    df = {"embeddings":[],
          "f1_score": [],
          "size":[]}
    
    true_labels = [s['algorithmic_solution'] for s in samples]
    label2id = {k:idx for idx, k in enumerate(set(true_labels))}
    id2label = {idx:k for k,idx in label2id.items()}

    true_labels = [label2id[l] for l in true_labels]
    k = len(set(true_labels))

    for view_embeddings in powerset(embeddings, 4):
        samples_per_embeddings = [np.array([s['embeddings'][e]  for s in samples]) for e in view_embeddings]
        embeddings_as_string = "/".join(view_embeddings)
        
        mvsc = MultiViewSpectralClustering()
        predicted_labels = mvsc.fit_predict(samples_per_embeddings, k)
        best_mapping =  find_best_cluster_mapping(true_labels, predicted_labels)

        predicted_mapped_labels = [best_mapping[l] for l in predicted_labels]

        f1_score_p = f1_score(true_labels, predicted_mapped_labels, average = 'macro')
        print(f"Using embeddings as views {embeddings_as_string} with f1-score {f1_score_p}")

        df['embeddings'].append(embeddings_as_string)
        df['f1_score'].append(f1_score_p)
        df['size'].append(len(true_labels))

    df = pd.DataFrame(df)
    
    return df

def validate_unsupervised_voting(samples, embeddings, clusterings, cache_embeddings_cluster):
    print('Validating unsupervised clustering')
    true_labels = [s['algorithmic_solution'] for s in samples]
    label2id = {k:idx for idx, k in enumerate(set(true_labels))}
    id2label = {idx:k for k,idx in label2id.items()}

    true_labels = [label2id[l] for l in true_labels]

    def calculate_support(labels):
        d = {}
        for l in labels:
            if l not in d:
                d[l] = 0
            d[l]+=1
        return d
    
    true_labels_support = calculate_support(true_labels)

    k = len(set(true_labels))

    df = {"embeddings":[],
          "clusterings":[],
          "subset_size":[],
          "f1_score_subset": [],
          "cotraining_size":[],
          "f1_score_cotraining": [],
          "size":[]}
    
    for l in label2id.keys():
        df[f'label_{l}_subset_ratio'] = []
        df[f'label_{l}_cotraining_ratio'] = []

        df[f'true_label_{l}_ratio'] = []

    for view_embeddings in powerset(embeddings, 4):
        for view_clusterings in all_products(clusterings, len(view_embeddings)):
            samples_emb_views = [np.array([s['embeddings'][e]  for s in samples]) for e in view_embeddings]

            embeddings_as_string = "/".join(view_embeddings)
            clusterings_as_string = "/".join(view_clusterings)

            samples_clustering_views = []

            for e, c in zip(view_embeddings, view_clusterings):
                samples_clustering_views.append(cache_embeddings_cluster[e][c])
            
            samples_clustering_views = np.array(samples_clustering_views).T
            
            uv = UnsupervisedVoting()

            result = uv.fit_predict(samples_emb_views, samples_clustering_views, k)

            if result is None:
                df['embeddings'].append(embeddings_as_string)
                df['clusterings'].append(clusterings_as_string)
                df['subset_size'].append(0)
                df['f1_score_subset'].append(0)
                df['cotraining_size'].append(0)
                df['f1_score_cotraining'].append(0)
                df['size'].append(len(true_labels))

                for l, l_idx in label2id.items():
                    df[f'label_{l}_subset_ratio'].append(0)
                    df[f'label_{l}_cotraining_ratio'].append(0)

                    df[f'true_label_{l}_ratio'].append(true_labels_support.get(l_idx, 0) / len(true_labels))
                print({k:v[-1] for k,v in df.items()})
                continue


            clusters_subset, indices_subset, predicted_labels_cotraining, indices_cotraining = result

            true_labels_subset = [true_labels[ind] for ind in indices_subset]
            best_mapping =  find_best_cluster_mapping(true_labels_subset, clusters_subset)
            clusters_subset = [best_mapping[l] for l in clusters_subset]
            f1_score_subset = f1_score(true_labels_subset, clusters_subset, average = 'macro')
            subset_support = calculate_support(clusters_subset)

            true_labels_cotraining = [true_labels[ind] for ind in indices_cotraining]
            predicted_labels_cotraining = [best_mapping[p] for p in predicted_labels_cotraining]
            f1_score_cotraining = f1_score(true_labels_cotraining, predicted_labels_cotraining, average = 'macro')
            cotraining_support = calculate_support(predicted_labels_cotraining)

            df['embeddings'].append(embeddings_as_string)
            df['clusterings'].append(clusterings_as_string)
            df['subset_size'].append(len(indices_subset))
            df['f1_score_subset'].append(f1_score_subset)
            df['cotraining_size'].append(len(indices_cotraining))
            df['f1_score_cotraining'].append(f1_score_cotraining)
            df['size'].append(len(true_labels))

            for l, l_idx in label2id.items():
                 df[f'label_{l}_subset_ratio'].append(subset_support.get(l_idx, 0) / len(indices_subset))
                 df[f'label_{l}_cotraining_ratio'].append(cotraining_support.get(l_idx, 0) / len(indices_cotraining))

                 df[f'true_label_{l}_ratio'].append(true_labels_support.get(l_idx, 0) / len(true_labels))

            print({k:v[-1] for k,v in df.items()})

    df = pd.DataFrame(df)
    return df     

def validate(args:dict[str, Any]):
    os.makedirs(args['destination_dir'], exist_ok= True)

    np.random.seed(42)
    train_args = args['train']

    train_dataset_path = train_args['dataset_info_path']
    train_embeddings_path = train_args['embeddings_path']

    train_dataset = load_dataset_with_embeddings(train_dataset_path, train_embeddings_path)

    test_args = args['test']

    test_dataset_path = test_args['dataset_info_path']
    test_embeddings_path = test_args['embeddings_path']

    test_dataset = load_dataset_with_embeddings(test_dataset_path, test_embeddings_path)

    dataset = copy.deepcopy(train_dataset)

    for k, v in test_dataset.items():
        dataset[k].extend(v)

    for problem_name,  samples in dataset.items():
        if problem_name in ['strmatch', 'swap', 'villages']:
            print(f"Validating problem {problem_name}")
            cache_hard_clustering, hard_clustering_df = validate_hard_clustering(samples, args['parameters']['embeddings'], args['parameters']['clustering_algos'])
            multi_view_clustering_df =  validate_multiview_spectral_clustering(samples, args['parameters']['embeddings'])
            unsupervised_voting_df = validate_unsupervised_voting(samples, args['parameters']['embeddings'], args['parameters']['clustering_algos'], cache_hard_clustering)

            hard_clustering_df.to_csv(os.path.join(args['destination_dir'], f'{problem_name}_hard_clustering.csv'))
            multi_view_clustering_df.to_csv(os.path.join(args['destination_dir'], f'{problem_name}_multi_view_clustering.csv'))
            unsupervised_voting_df.to_csv(os.path.join(args['destination_dir'], f'{problem_name}_unsupervised_voting.csv'))










    







    



