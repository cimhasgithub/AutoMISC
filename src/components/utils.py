"""
Utilities for invoking chat models across parser and annotator modules.
"""
from typing import Optional, Type, List
from pydantic import BaseModel
import lmstudio as lms
import openai
from hydra.utils import log

def call_chat_model(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    response_format: Optional[Type[BaseModel]] = None,
    **kwargs,
) -> BaseModel | str:
    """
    """
    if model in ['openai/gpt-4o']:
        if openai is None:
            raise ImportError("openai library is required for openai models")
        response = openai.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            **kwargs,
        )
        return response.choices[0].message.parsed.model_dump()
    elif model in ['qwen/qwen3-30b-a3b', 'google/gemma-3-12b']:
        lms_model = lms.llm(model)
        completion = lms_model.respond(
            {"messages": messages},
             config={"temperature": temperature}, 
             response_format=response_format
            )
                                       
        return completion.parsed
    else:
        raise NotImplementedError(f"Model '{model}' not implemented")
