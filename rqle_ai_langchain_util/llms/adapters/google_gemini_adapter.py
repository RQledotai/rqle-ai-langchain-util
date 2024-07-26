import os
from dotenv import load_dotenv

from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _load_google_gemini_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Google Gemini LLMs
    """
    llm = GoogleGenerativeAI(
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        model=config.model_name,
        temperature=config.parameters.temperature,
        top_p=config.parameters.top_p,
        max_output_tokens=config.parameters.max_tokens
    )
    return llm


def _load_google_gemini_chat_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain Chat object configured for Google Gemini LLMs
    """
    llm = ChatGoogleGenerativeAI(
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        model=config.model_name,
        temperature=config.parameters.temperature,
        top_p=config.parameters.top_p,
        max_output_tokens=config.parameters.max_tokens
    )
    return llm


def _load_google_gemini_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain embeddings object configured for Google Gemini LLMs
    """
    llm = GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        model=config.model_name
    )
    return llm


def load_google_gemini_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for Google Gemini LLMs
    """
    if config.type == PromptTypeEnum.chat:
        return _load_google_gemini_chat_from_prompt_config(config)
    elif config.type == PromptTypeEnum.completion:
        return _load_google_gemini_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_google_gemini_embeddings_from_prompt_config(config)
