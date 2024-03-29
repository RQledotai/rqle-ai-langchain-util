from unittest import TestCase, main
from parameterized import parameterized

from langchain_google_genai import GoogleGenerativeAI, ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig, ExecutionParameters
from rqle_ai_langchain_util.llms.adapters.google_gemini_adapter import load_google_gemini_from_prompt_config


class TestGoogleGeminiAdapter(TestCase):

    @parameterized.expand([
        ['chat', ChatGoogleGenerativeAI],
        ['completion', GoogleGenerativeAI],
        ['embeddings', GoogleGenerativeAIEmbeddings]
    ])
    def test_load_google_gemini_from_prompt_config(self, llm_type: str, llm_class):
        config = PromptConfig(type=llm_type, model_name='gemini-1.0-pro',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_google_gemini_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)

    def test_load_google_gemini_from_prompt_config_error(self):
        config = PromptConfig(type='invalid', model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        with self.assertRaises(NotImplementedError):
            load_google_gemini_from_prompt_config(config)


if __name__ == '__main__':
    main()
