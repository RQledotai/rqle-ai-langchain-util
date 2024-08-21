from unittest import TestCase, main
from parameterized import parameterized

from langchain_community.llms import OCIGenAI
from langchain_community.embeddings import OCIGenAIEmbeddings

from rqle_ai_langchain_util.llms.adapters.oci_ai_adapter import load_oci_ai_llm_from_prompt_config
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum


class TestOCIAIAdapter(TestCase):

    @parameterized.expand([
        [PromptTypeEnum.completion, OCIGenAI],
        [PromptTypeEnum.embedding, OCIGenAIEmbeddings]
    ])
    def test_load_oci_ai_llm_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_oci_ai_llm_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)

    def test_load_oci_ai_llm_from_prompt_config_type_unsupported(self):
        config = PromptConfig(type=PromptTypeEnum.chat, model_name='test_model',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        with self.assertRaises(NotImplementedError):
            load_oci_ai_llm_from_prompt_config(config)


if __name__ == '__main__':
    main()
