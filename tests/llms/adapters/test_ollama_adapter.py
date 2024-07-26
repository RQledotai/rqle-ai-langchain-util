from unittest import TestCase, main
from parameterized import parameterized

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

from rqle_ai_langchain_util.llms.adapters.ollama_adapter import load_ollama_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum


class TestOllamaAdapter(TestCase):

    @parameterized.expand([
        [PromptTypeEnum.chat, ChatOllama],
        [PromptTypeEnum.completion, Ollama],
        [PromptTypeEnum.embedding, OllamaEmbeddings]
    ])
    def test_load_ollama_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_ollama_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)


if __name__ == '__main__':
    main()
