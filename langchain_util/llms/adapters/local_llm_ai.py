import os

from dotenv import load_dotenv

from langchain.llms import OpenAI, OpenAIChat
from langchain_util.prompts.prompt_config import PromptConfig

load_dotenv()


def _load_local_model_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a OpenAI object configured for a local LLM
    """
    llm = OpenAI(
        base_url=os.getenv('LOCAL_BASE_ENDPOINT'),
        api_key=os.getenv('LOCAL_BASE_API_KEY'),
        model_name=config.model_name,
        temperature=config.parameters.temperature,
        max_tokens=config.parameters.max_tokens,
        top_p=config.parameters.top_p,
        presence_penalty=config.parameters.presence_penalty,
        frequency_penalty=config.parameters.frequency_penalty
    )
    return llm


def _load_local_chat_model_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a OpenAIChat object configured for a local LLM
    """
    llm = OpenAIChat(
        base_url=os.getenv('LOCAL_BASE_ENDPOINT'),
        api_key=os.getenv('LOCAL_BASE_API_KEY'),
        model_name=config.model_name,
        temperature=config.parameters.temperature,
        max_tokens=config.parameters.max_tokens,
        top_p=config.parameters.top_p,
        presence_penalty=config.parameters.presence_penalty,
        frequency_penalty=config.parameters.frequency_penalty
    )
    return llm


def _load_local_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a OpenAI object configured for a local LLM
    """
    if config.type == 'completion':
        return _load_local_model_from_prompt_config(config)
    elif config.type == 'chat':
        return _load_local_chat_model_from_prompt_config(config)
    else:
        raise Exception(f'LLM type {config.type} not supported')
