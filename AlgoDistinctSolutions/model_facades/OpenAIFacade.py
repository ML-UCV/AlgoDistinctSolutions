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
from openai import OpenAI
from model_facades.BaseFacade import BaseFacade

from utils import execute_function_with_retries

logger = logging.getLogger()

class OpenAIFacade(BaseFacade):
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
        client = OpenAI()

        logging.getLogger("openai").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)


        print('Generating embeddings')
        source_codes_embeddings = []

        for source_code in tqdm(preprocessed_source_codes):
            def get_embedding():
                response = client.embeddings.create(
                        input=source_code,
                        model=model_name
                    )
                return response.data[0].embedding
            
            sc_embedding = execute_function_with_retries(get_embedding)
            source_codes_embeddings.append(sc_embedding)

        return source_codes_embeddings
