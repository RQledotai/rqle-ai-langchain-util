from unittest import TestCase, main
from parameterized import parameterized

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from rqle_ai_langchain_util.llms.adapters.ollama_adapter import load_ollama_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, ExecutionParameters


class TestOllamaAdapter(TestCase):

    @parameterized.expand([
        ['chat', ChatOllama],
        ['completion', Ollama],
        ['embeddings', OllamaEmbeddings]
    ])
    def test_load_ollama_from_prompt_config(self, llm_type: str, llm_class):
        config = PromptConfig(type=llm_type, model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_ollama_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)

    def test_load_ollama_from_prompt_config_error(self):
        config = PromptConfig(type='invalid', model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        with self.assertRaises(NotImplementedError):
            load_ollama_from_prompt_config(config)


if __name__ == '__main__':
    main()
