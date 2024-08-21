from unittest import TestCase, main
from parameterized import parameterized


from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from rqle_ai_langchain_util.llms.adapters.google_gemini_adapter import load_google_gemini_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum


class TestGoogleGeminiAdapter(TestCase):

    @parameterized.expand([
        [PromptTypeEnum.chat, ChatGoogleGenerativeAI],
        [PromptTypeEnum.completion, GoogleGenerativeAI],
        [PromptTypeEnum.embedding, GoogleGenerativeAIEmbeddings]
    ])
    def test_load_google_gemini_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, model_name='gemini-1.0-pro',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_google_gemini_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)


if __name__ == '__main__':
    main()
