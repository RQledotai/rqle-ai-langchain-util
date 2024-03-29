import os

from unittest import TestCase, main
from parameterized import parameterized

from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings
from rqle_ai_langchain_util.llms.adapters.azure_openai_adapter import load_azure_openai_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, ExecutionParameters


AZURE_OPENAI_ENDPOINT=''
AZURE_OPENAI_API_KEY=''
OPENAI_API_VERSION=''

class AzureOpenAIAdapter(TestCase):
    @parameterized.expand([
        ['chat', AzureChatOpenAI],
        ['completion', AzureOpenAI],
        ['embeddings', AzureOpenAIEmbeddings]
    ])
    def test_load_azure_openai_from_prompt_config(self, llm_type: str, llm_class):
        config = PromptConfig(type=llm_type, deployment_name='test-deployment', model_name='test-model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_azure_openai_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)

    def test_load_azure_openai_from_prompt_config_error(self):
        config = PromptConfig(type='invalid', model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        with self.assertRaises(NotImplementedError):
            load_azure_openai_from_prompt_config(config)


if __name__ == '__main__':
    main()
