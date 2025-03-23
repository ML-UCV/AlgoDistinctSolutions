# from PredictionMethods import PredictionMethods
# from ValidationPipelines import ValidationPipelines
# from DetermineKValidationPipelines import DetermineKPipelines

# from AlgoLabelHelper import AlgoLabelHelper
# from StatisticsHelper import StatisticsHelper
import steps
import argparse
import os
import yaml
import sys
from pprint import pformat

import logging


if __name__ == '__main__':
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(handler)

        if(not os.path.exists('tmp')):
             os.mkdir('tmp')


        with open(f'Parameters.yaml','r') as f:
            parameters = yaml.safe_load(f)

        logger.info("Loaded the following parameters:")
        logger.info(pformat(parameters))

        parser = argparse.ArgumentParser(description='Code embeddings in the context of different solutions in a competitive programming problem')

        # parser.add_argument('--statistics', dest='statisticsType', action = 'append', choices=['dataset', 'incremental_tfidf'], help='Get statistics based on the choice (dataset, source code, results)')
        parser.add_argument('--split', action='store_true', help='Split dataset in train / test')
        parser.add_argument('--pretrain-embeddings', action='store_true', help='Pretrain w2v embeddings and tfidf.')
        parser.add_argument('--generate-embeddings', action='store_true', help='Computes the embeddings for datasets')
        parser.add_argument('--plot-score-per-samples', action = 'store_true', help = 'Determine how classifier metrics evolve based on number of samples used')
        # parser.add_argument('--evaluate', action='store_true', help='Evaluates how well the embeddings contribute in the distinct solutions problem')
        # parser.add_argument('--evaluate-k-selection', action='store_true', help='Evaluates how well the embeddings contribute to determining the optimal number k')

       
        args = parser.parse_args(['--plot-score-per-samples'])

        # algoLabelHelper = AlgoLabelHelper()
        # statisticsHelper = StatisticsHelper()

        # if(args.statisticsType is not None and len(args.statisticsType) > 0):
        #     statisticsHelper.handleStatistics(args.statisticsType)

        if (args.split is True):
                steps.split_dataset(**parameters['split'])

        if(args.pretrain_embeddings is True):
                steps.pretrain_embeddings(parameters['pretrain-embeddings'])

        if(args.generate_embeddings is True):
                steps.generate_embeddings(parameters['generate-embeddings'])
        if(args.plot_score_per_samples is True):
                steps.plot_score_per_samples(parameters['plot-score-per-samples'])

        # if(args.evaluate is True):
        #     validationPipelines = ValidationPipelines()

        #     validationPipelines.k_clustering_pipeline()
        #     validationPipelines.estimator_pipeline()
        #     validationPipelines.unsupervised_voting_pipeline()
        #     validationPipelines.semi_supervised_multiview_spectral_clustering()

        # if(args.evaluate_k_selection is True):
        #     kvalidationPipelines = DetermineKPipelines()

        #     kvalidationPipelines.k_simple_clustering_pipeline()