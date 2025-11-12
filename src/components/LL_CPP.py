# wrapper class for llama_cpp

# target func.

from pydantic import BaseModel
from llama_cpp import Llama
import instructor
import os

os.environ["GGML_METAL_VERBOSE"] = "0" # this verbosity gets annoying 

VALID_MODELS = {"gemma-3-12B-it-qat.gguf","gemma-3-12b-it-q4_0.gguf"}

lookup = {"gemma-3-12b-it-q4_0.gguf" : ("lmstudio-community/gemma-3-12B-it-qat-GGUF","gemma-3-12B-it-QAT-Q4_0.gguf"),}

class LL_cpp:
    def __init__(self, model_str):
        val = lookup[model_str]
        self.repo_id = val[0]
        self.filename = val[1]
 
        
        self.llm = Llama.from_pretrained(
            repo_id=self.repo_id, 
            filename=self.filename, 
            n_ctx=3000, # should be a json parameter
            n_gpu_layers=-1,
            verbose=False
        )
        
        self.create = instructor.patch(
            create=self.llm.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
        )

    def __call__(self, messages, response_format, temperature):

        result = self.create(
            messages=messages,
            response_model=response_format,
            temperature=temperature,
        )


        return result.model_dump()

