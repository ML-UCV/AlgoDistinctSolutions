from abc import ABC, abstractmethod
import json
import numpy as np
from gensim.models import Word2Vec

class EmbeddingsLoader(ABC):

    @abstractmethod
    def GetEmbeddings(self):
        pass

    @abstractmethod
    def GetName(self):
        pass


class W2VEmbeddingsLoader(EmbeddingsLoader):

    def GetEmbeddings(self):
        model = Word2Vec.load(self.__GetRelativePath("w2v_128_dict.emb"))
        matrix = np.array(model.wv.vectors)
        vocab = model.wv.vocab

        problems = json.load(open(self.__GetRelativePath("w2vEmbeddings.json"), "r"))

        for problem in problems:
            tokens = problem['tokens']

            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": [matrix[vocab[token].index].tolist() for token in tokens if(token in vocab)],
                    "tokens": tokens
                }
            
    def GetName(self):
        return 'w2v'
    
    def __GetRelativePath(self, path):
        return f'Data/Embeddings/w2v/{path}'


class SafeEmbeddingsLoader(EmbeddingsLoader):

    def GetEmbeddings(self):
        problems = json.load(open(self.__GetRelativePath("safeEmbeddings.json"), "r"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": problem["safe"]
                }
        
    def GetName(self):
        return 'safe'

    def __GetRelativePath(self, path):
        return f'Data/Embeddings/safe/{path}'
        

class TfidfEmbeddingsLoader(EmbeddingsLoader):
    
    def GetEmbeddings(self):
        problems = json.load(open(self.__GetRelativePath("tfidfEmbeddings.json"), "r"))

        for problem in problems:
            yield {
                    "index":problem["index"],
                    "label":problem["label"],
                    "embeddings": problem["tfidf"]
                }
            
    def GetName(self):
        return 'tfidf'
    
    def __GetRelativePath(self, path):
        return f'Data/Embeddings/tfidf/{path}'