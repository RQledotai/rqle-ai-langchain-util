import os

from unittest import TestCase, main
from parameterized import parameterized

from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings

from rqle_ai_langchain_util.llms.adapters.azure_openai_adapter import load_azure_openai_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum

AZURE_OPENAI_ENDPOINT = ''
AZURE_OPENAI_API_KEY = ''
OPENAI_API_VERSION = ''


class AzureOpenAIAdapter(TestCase):
    @parameterized.expand([
        [PromptTypeEnum.chat, AzureChatOpenAI],
        [PromptTypeEnum.completion, AzureOpenAI],
        [PromptTypeEnum.embedding, AzureOpenAIEmbeddings]
    ])
    def test_load_azure_openai_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, deployment_name='test-deployment', model_name='test-model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_azure_openai_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)


if __name__ == '__main__':
    main()
