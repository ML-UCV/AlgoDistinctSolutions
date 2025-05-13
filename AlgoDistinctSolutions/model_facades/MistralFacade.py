from preprocessing_operations.RemoveUnusedFunctionsPreprocessingOp import RemoveUnusedFunctionsPreprocessingOp
from preprocessing_operations.RemoveIncludesUsingPreprocessingOp import RemoveIncludesUsingPreprocessingOp
from preprocessing_operations.RemoveCommentsPreprocessingOp import RemoveCommentsPreprocessingOp
from preprocessing_operations.ReplaceMacrosPreprocessingOp import ReplaceMacrosPreprocessingOp
from preprocessing_operations.ExtractFunctionsAndMethodsProcessingOp import ExtractFunctionsAndMethodsPreprocessingOp

import logging
from tqdm import tqdm
from typing import Any
import torch
from utils import to_batches
from model_facades.BaseFacade import BaseFacade
import os
from utils import execute_function_with_retries
from mistralai import Mistral
import time

logger = logging.getLogger()

class MistralFacade(BaseFacade):
    def _preprocessing_fn(self, source_code:str) -> str:
        remove_comments_pre_op = RemoveCommentsPreprocessingOp()
        remove_includes_pre_op = RemoveIncludesUsingPreprocessingOp()
        replace_macros_pre_op = ReplaceMacrosPreprocessingOp()
        remove_unused_functions_pre_op = RemoveUnusedFunctionsPreprocessingOp()
        
        preprocessed_source_code = remove_comments_pre_op.preprocess(source_code)
        preprocessed_source_code = remove_includes_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = replace_macros_pre_op.preprocess(preprocessed_source_code)
        preprocessed_source_code = remove_unused_functions_pre_op.preprocess(preprocessed_source_code)
 
        return preprocessed_source_code

    def _generate_embeddings_fn(self, preprocessed_source_codes:list[Any], **kwargs):
        model_name:str = kwargs.pop('model_name')
        client = Mistral(api_key = os.environ['MISTRAL_API_KEY'])

        print('Generating embeddings')
        source_codes_embeddings = []

        for batch_source_codes in tqdm(list(to_batches(preprocessed_source_codes,5))):
            def get_embedding():
                response = client.embeddings.create(
                                model=model_name,
                                inputs=batch_source_codes
                        )
                return [r.embedding for r in response.data]
            
            sc_batch_embeddings = execute_function_with_retries(get_embedding)
            source_codes_embeddings.extend(sc_batch_embeddings)
            time.sleep(1)

        return source_codes_embeddings
