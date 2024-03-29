from rqle_ai_langchain_util.llms.adapters.llm_adapters import LLMAdapter
from rqle_ai_langchain_util.llms.adapters import (aws_bedrock_adapter, azure_openai_adapter, google_gemini_adapter,
                                                  ollama_adapter, oci_ai_adapter)
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig
from rqle_ai_langchain_util.prompts.prompt_template import PromptTemplate
from rqle_ai_langchain_util.prompts.prompt_example import PromptExample
from rqle_ai_langchain_util.settings import PROMPT_CONFIG_FOLDER
from rqle_ai_langchain_util.utils.file_util import file_exists, read_file


class LLMMediator:

    def info(self) -> str:
        """
        :return: the name of the class
        """
        return self.__class__.__name__

    def __init__(self, llm_adapter: LLMAdapter, prompt_config_name: str):
        """
        :param llm_adapter: Adapter for the execution of the LLM
        :param prompt_config_name: Name of the prompt
        """
        self._llm_adapter = llm_adapter
        self._prompt_name = prompt_config_name
        # populate the prompt config
        self._prompt_config = PromptConfig.from_json(f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}')
        # populate the prompt template
        if file_exists(f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}', 'prompt.json'):
            self._prompt_template = PromptTemplate.from_json(f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}')
        else:
            self._prompt_template = PromptTemplate.from_text(read_file(file_dir=f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}',
                                                                       file_name='prompt.txt'))
        # populate the prompt example
        if file_exists(f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}', 'example.json'):
            self._prompt_example = PromptExample.from_json(f'{PROMPT_CONFIG_FOLDER}/{prompt_config_name}')
        self._model = _load_model(llm_adapter, self._prompt_config)

    @property
    def prompt_name(self) -> str:
        """
        :return: Prompt name
        """
        return self._prompt_name

    @property
    def prompt_config(self) -> PromptConfig:
        """
        :return: Prompt configuration
        """
        return self._prompt_config

    @property
    def prompt_template(self) -> PromptTemplate:
        """
        :return: Prompt template
        """
        return self._prompt_template

    @property
    def prompt_example(self) -> PromptExample:
        """
        :return: Prompt example
        """
        return self._prompt_example

    @property
    def model(self):
        """
        :return: LLM model
        """
        return self._model


def _load_model(llm_adapter: LLMAdapter, llm_config: PromptConfig):
    """
    :param llm_adapter: Adapter for the execution of the LLM
    :param llm_config: Configuration for the LLM
    :return: Configured LLM
    """
    llm = None
    if llm_adapter == LLMAdapter.AWS_BEDROCK:
        llm = aws_bedrock_adapter.load_aws_bedrock_from_prompt_config(llm_config)
    elif llm_adapter == LLMAdapter.AZURE_OPENAI:
        llm = azure_openai_adapter.load_azure_openai_from_prompt_config(llm_config)
    elif llm_adapter == LLMAdapter.GOOGLE_GEMINI:
        llm = google_gemini_adapter.load_google_gemini_from_prompt_config(llm_config)
    elif llm_adapter == LLMAdapter.OLLAMA_AI:
        llm = ollama_adapter.load_ollama_from_prompt_config(llm_config)
    elif llm_adapter == LLMAdapter.OCI_AI:
        llm = oci_ai_adapter.load_oci_ai_llm_from_prompt_config(llm_config)
    else:
        raise NotImplementedError(f'LLM adapter {llm_adapter} not supported')
    return llm
