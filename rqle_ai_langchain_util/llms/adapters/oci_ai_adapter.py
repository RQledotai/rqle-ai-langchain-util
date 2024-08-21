import os
from dotenv import load_dotenv

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _load_oci_ai_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Oracle Cloud AI LLMs
    """
    llm = OCIGenAI(
        model_id=config.model_name,
        service_endpoint=os.getenv('OCI_AI_ENDPOINT'),
        compartment_id=os.getenv('OCI_AI_COMPARTMENT_ID'),
        model_kwargs={
            'max_tokens': config.parameters.max_tokens,
            'temperature': config.parameters.temperature,
            'top_p': config.parameters.top_p,
        }
    )
    return llm


def _load_oci_ai_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Oracle Cloud AI LLMs
    """
    llm = OCIGenAIEmbeddings(
        model_id=config.model_name,
        service_endpoint=os.getenv('OCI_AI_ENDPOINT'),
        compartment_id=os.getenv('OCI_AI_COMPARTMENT_ID')
    )
    return llm


def load_oci_ai_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Oracle Cloud AI LLMs
    """
    if config.type == PromptTypeEnum.completion:
        return _load_oci_ai_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_oci_ai_embeddings_from_prompt_config(config)
    else:
        raise NotImplementedError(f'LLM type {config.type} not supported')
