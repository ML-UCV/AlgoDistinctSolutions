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

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from utils import load_dataset_with_embeddings, shuffle_arrays
import numpy as np
import matplotlib.pyplot as plt
import scipy


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
        "codet5plus": CodeT5plusFacade()
        }

        for emb_model_name, model_facade in emb_model_mapping.items():
            if emb_model_name in args_dataset:
                model_args = args_dataset[emb_model_name]
                destination_dir = model_args.pop('destination_dir')
                
                model_facade.generate_embeddings(dataset_info_path, 
                                destination_dir,
                                preprocessing_workers,
                                **model_args)
                
def plot(train_dataset, test_dataset, embeddings_names:list[str], problem_name:str):
    print(f"Plotting problem {problem_name}")
    percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tries = 30

    import scipy.stats as st
    
    np.random.seed(24)

    for e in embeddings_names:
        print(f'\tPlotting embedding {e}')
        train_X = np.array([s['embeddings'][e] for s in train_dataset])
        train_Y = np.array([s['algorithmic_solution'] for s in train_dataset])

        test_X = np.array([s['embeddings'][e] for s in test_dataset])
        test_Y = np.array([s['algorithmic_solution'] for s in test_dataset])

        percentiles_values = [int(len(train_X) * p) + 1 for p in percentiles]

        mean_per_percentile = [0]
        ci_l_per_percentile = [0]
        ci_r_per_percentile = [0]

        for p in percentiles_values:
            scores_per_try = []
            for _ in range(tries):
                train_X_shuffled, train_Y_shuffled = shuffle_arrays(train_X, train_Y)

                clf = RandomForestClassifier()
                
                clf.fit(train_X_shuffled[:p], train_Y_shuffled[:p])

                predictions = clf.predict(test_X)

                score = f1_score(test_Y, predictions, average = 'macro')

                scores_per_try.append(score)

            scores_per_try  =np.array(scores_per_try)

            mean_per_percentile.append(scores_per_try.mean())

            ci_l, ci_r = scipy.stats.norm.interval(0.90, loc = np.mean(scores_per_try), scale = scipy.stats.sem(scores_per_try))

            ci_l_per_percentile.append(ci_l)
            ci_r_per_percentile.append(ci_r)
        
        fig, ax = plt.subplots()
        
        ax.plot([0] + percentiles, mean_per_percentile)
    
        for idx in range(len(mean_per_percentile) - 1):
            x_range = percentiles[idx : idx + 2]
            ci_l = ci_l_per_percentile[idx : idx + 2]
            ci_r = ci_r_per_percentile[idx : idx + 2]

            ax.fill_between(x_range, ci_l, ci_r, color='b', alpha=.1)

        ax.set_title(f'RF Prediction on {problem_name} using {e} embeddings')
        ax.set_xlabel("Ratio of samples")
        ax.set_xlabel("F1-score")


        fig.savefig(f'Data/Plots/{problem_name}_{e}.png')

        plt.close()
    

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

    for p in problems:
        problem_train_dataset = train_dataset[p]
        problem_test_dataset = test_dataset[p]

        plot(problem_train_dataset, problem_test_dataset, list(train_embeddings_path.keys()), p)


    







    



