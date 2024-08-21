# Information about the different providers supported by AWS Bedrock -
# https://us-west-2.console.aws.amazon.com/bedrock/home
import os
from dotenv import load_dotenv

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, PromptTypeEnum

load_dotenv()


def _get_model_kwargs(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: the model arguments relevant for different models supported in AWS bedrock
    """
    provider = config.model_name.split('.')[0]
    if provider == 'ai21':
        return {
            "maxTokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "topP": config.parameters.top_p,
            "presencePenalty":{"scale": config.parameters.presence_penalty},
            "frequencyPenalty":{"scale": config.parameters.frequency_penalty}
        }
    elif provider == 'amazon':
        # TODO investigate '/basic-syntax/' text when using completion
        return {
            "maxTokenCount": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "topP": config.parameters.top_p
        }
    elif provider == 'cohere':
        return {
            "max_tokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "p": config.parameters.top_p
        }
    elif provider == 'meta':
        # TODO investigate why parameters work in direct call, but not when passed through UI
        # Malformed input request: #: extraneous key [max_gen_length] is not permitted
        return {
            "max_gen_len": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "top_p": config.parameters.top_p
        }
    elif provider == 'mistral':
        # TODO investigating issue with mistral model
        # Malformed input request: #: required key [prompt] not found#: extraneous key [inputText] is not permitted
        return {
            "max_tokens": config.parameters.max_tokens,
            "temperature": config.parameters.temperature,
            "top_p": config.parameters.top_p
        }
    else:
        raise NotImplementedError(f'LLM provider {provider} not supported')


def _load_aws_bedrock_llm_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for AWS Bedrock LLMs
    """
    # transform the model configuration in the appropriate dictionary
    model_kwargs = _get_model_kwargs(config)
    # create a new Bedrock LLM object
    llm = BedrockLLM(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name,
        model_kwargs= model_kwargs
    )
    return llm


def _load_aws_bedrock_chat_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain Chat object configured for AWS Bedrock LLMs
    """
    # transform the model configuration in the appropriate dictionary
    model_kwargs = _get_model_kwargs(config)
    # create a new Bedrock LLM object
    llm = ChatBedrock(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name,
        model_kwargs=model_kwargs
    )
    return llm


def _load_aws_bedrock_embeddings_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain embeddings object configured for AWS Bedrock LLMs
    """
    llm = BedrockEmbeddings(
        credentials_profile_name=os.environ.get("BWB_PROFILE_NAME"),
        region_name=os.environ.get("BWB_REGION_NAME"),
        endpoint_url=os.environ.get("BWB_ENDPOINT_URL"),
        model_id=config.model_name
    )
    return llm


def load_aws_bedrock_from_prompt_config(config: PromptConfig):
    """
    :param config: the configuration for the LLM execution
    :return: a LangChain object configured for AWS Bedrock LLMs
    """
    if config.type == PromptTypeEnum.chat:
        return _load_aws_bedrock_chat_from_prompt_config(config)
    elif config.type == PromptTypeEnum.completion:
        return _load_aws_bedrock_llm_from_prompt_config(config)
    elif config.type == PromptTypeEnum.embedding:
        return _load_aws_bedrock_embeddings_from_prompt_config(config)
