# AlgoDistinctSolutions

The scope of this repository is to offer a tool in which you can, for a given competitive programming problem, split it in K different solutions in terms of algorthmic approach.
We propose an multi-view unsupervised voting approach that uses multiple embeddings of the same dataset as views. We select a subset of solutions that is agreed by all the embeddings by clustering each view and converting the problem in a Multidimensional Assigment Problem. We extend the subset by employing a co-training self-learning scheme based on each view of the subset.

We use the following embeddings: Word2Vec, Tf-IDF, SAFE, UniXcoder, CodeT5+, OpenAI and MistralAI.

## Manual annotated dataset
To validate the method, a dataset(AlgoSol-15) which contains 15 problems from Infoarena was manually annotated. The dataset can be downloaded from here: https://huggingface.co/datasets/Arkimond9620/AlgoSol-15/tree/main.

The dataset is structured as follows:
- Each folder from root represents a problem. Each folder from a problem represents a distinct algorithmic solution and contains all the corresponding source codes.
- Dataset.json - contains metadata for each source code solution
- Train.json - contains 80% of source codes from Dataset.json per problem
- Test.json - contains 20% of source codes from Dataset.json per problem.

Metadata available for each sample in Dataset.json:
- id - a random guid
- path - path to the source code
- problem - problem name
- algorithmic_solution - label for the algorithmic solution used

## How to run
The repository provides a Dockerfile and is meant to be run as a devcontainer, in order to be reproducible. Clone the repository to a folder and run the following commannd `docker compose up`. 
The repository provides 5 commands that are available in `main.py`:
- split - split Dataset.json in train and test folds.
- pretrain-embeddings - Pretrain W2V and TfIdf on train.json
- generate-embeddings - Generate embeddings for all available
- plot-score-per-sample - Plot for each embedding, what is the validation score on `Test.json` if you train an XGBoost model but use only X samples from `Train.json`
- validate - Run the entire validation pipeline in order to obtain the scores for each available method.

Note that the parameters for each method can be found in `Parameters.yaml`.

Note that for OpenAI and MistralAI, one needs to set the following environment variables: OPENAI_API_KEY, MISTRAL_API_KEY.
