"""
Utilities for invoking chat models across parser and annotator modules.
"""
from typing import Optional, Type, List, Literal
from pydantic import BaseModel
import lmstudio as lms
import openai
from hydra.utils import log
from components.LL_CPP import LL_cpp, VALID_MODELS  # custom wrapper

def get_provider(model: str) -> Literal['openai', 'lmstudio', 'llama_cpp']:
    lms_models = {m.model_key for m in lms.list_downloaded_models("llm")}
    openai_models = {m.id for m in openai.models.list().data}
    llm_cpp_models = VALID_MODELS

    if model in openai_models:
        return 'openai'
    elif model in lms_models:
        return 'lmstudio'
    elif model in llm_cpp_models:
        return 'llama_cpp'
    else:
        raise ValueError(f"Model '{model}' not found in OpenAI, LM Studio, or llama_cpp models.")

def call_chat_model(
    messages: list[dict],
    model: str,
    provider: Literal['openai', 'lmstudio', 'llama_cpp'] = 'openai',
    temperature: float = 0.0,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> BaseModel | str:
    """
    """
    if provider == "llama_cpp":
        model_llama_cpp = LL_cpp(model)
    if provider == 'openai':
        if openai is None:
            raise ImportError("openai library is required for openai models")
        response = openai.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            **kwargs,
        )
        return response.choices[0].message.parsed.model_dump()
    elif provider == 'lmstudio':
        lms_model = lms.llm(model)
        completion = lms_model.respond(
            {"messages": messages},
             config={"temperature": temperature}, 
             response_format=response_format
            )
                                       
        return completion.parsed

    elif provider == "llama_cpp":
        completion = model_llama_cpp(
            messages=messages,
            response_format=response_format,
            temperature=temperature
            )
                                       
        return completion


    else:
        raise ValueError(f"Provider '{provider}' not recognized. Use 'openai' or 'lmstudio'.")



class chat_model_llama_cpp:
    def __init__(self, model_name):
        self.model_llama_cpp = LL_cpp(model_name)
    
    def call_model(self, messages, response_format, temperature,model = None,provider=None):
        completion = self.model_llama_cpp(
        messages=messages,
        response_format=response_format,
        temperature=temperature
        )
                                    
        return completion

