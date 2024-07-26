from dotenv import load_dotenv

from langchain_openai import AzureOpenAI, AzureChatOpenAI, AzureOpenAIEmbeddings

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _load_azure_openai_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Azure OpenAI LLMs
    """
    llm = AzureOpenAI(
        deployment_name=config.model_name,
        model_name=config.model_name,
        temperature=config.parameters.temperature,
        max_tokens=config.parameters.max_tokens,
        top_p=config.parameters.top_p,
        presence_penalty=config.parameters.presence_penalty,
        frequency_penalty=config.parameters.frequency_penalty
    )
    return llm


def _load_azure_openai_chat_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Azure OpenAI LLMs
    """
    llm = AzureChatOpenAI(
        deployment_name=config.model_name,
        model_name=config.model_name,
        temperature=config.parameters.temperature,
        max_tokens=config.parameters.max_tokens
    )
    return llm


def _load_azure_openai_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Azure OpenAI LLMs
    """
    llm = AzureOpenAIEmbeddings(model=config.model_name)
    return llm


def load_azure_openai_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Azure OpenAI LLMs
    """
    if config.type == PromptTypeEnum.chat:
        return _load_azure_openai_chat_from_prompt_config(config)
    elif config.type == PromptTypeEnum.completion:
        return _load_azure_openai_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_azure_openai_embeddings_from_prompt_config(config)
