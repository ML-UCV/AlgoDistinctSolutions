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

        parser.add_argument('--split', action='store_true', help='Split dataset in train / test')
        parser.add_argument('--pretrain-embeddings', action='store_true', help='Pretrain w2v embeddings and tfidf.')
        parser.add_argument('--generate-embeddings', action='store_true', help='Computes the embeddings for datasets')
        parser.add_argument('--plot-score-per-samples', action = 'store_true', help = 'Determine how classifier metrics evolve based on number of samples used')
        parser.add_argument('--validate', action='store_true', help='Validates all the methods')

        args = parser.parse_args()

        if args.split:
                steps.split_dataset(**parameters['split'])

        if args.pretrain_embeddings:
                steps.pretrain_embeddings(parameters['pretrain-embeddings'])

        if args.generate_embeddings:
                steps.generate_embeddings(parameters['generate-embeddings'])

        if args.plot_score_per_samples:
                steps.plot_score_per_samples(parameters['plot-score-per-samples'])

        if args.validate:
               steps.validate(parameters['validate'])