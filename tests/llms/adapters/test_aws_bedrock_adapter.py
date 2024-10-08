from unittest import TestCase, main
from parameterized import parameterized

from langchain_aws.llms.bedrock import BedrockLLM
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_aws.embeddings.bedrock import BedrockEmbeddings

from rqle_ai_langchain_util.llms.adapters.aws_bedrock_adapter import load_aws_bedrock_from_prompt_config, _get_model_kwargs
from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig, PromptTypeEnum


class TestAWSBedrockAdapter(TestCase):

    @parameterized.expand([
        ('ai21', {'maxTokens': 100, 'temperature': 0.5, 'topP': 0.5, 'presencePenalty': {'scale': 0.5},
                  'frequencyPenalty': {'scale': 0.5}}),
        ('amazon', {'maxTokenCount': 100, 'temperature': 0.5, 'topP': 0.5}),
        ('cohere', {'max_tokens': 100, 'temperature': 0.5, 'p': 0.5}),
        ('meta', {'max_gen_len': 100, 'temperature': 0.5, 'top_p': 0.5})
    ])
    def test_get_model_kwargs(self, provider, expected):
        config = PromptConfig(model_name=f'{provider}.test',
                              parameters=ExecutionParameters(max_tokens=100, temperature=0.5, top_p=0.5,
                                                             presence_penalty=0.5, frequency_penalty=0.5))
        kwargs = _get_model_kwargs(config)
        self.assertEqual(kwargs, expected)

    def test_get_model_kwargs_invalid_provider(self):
        config = PromptConfig(model_name='invalid.test',
                              parameters=ExecutionParameters(max_tokens=100, temperature=0.5, top_p=0.5))
        with self.assertRaises(NotImplementedError):
            _get_model_kwargs(config)

    @parameterized.expand([
        [PromptTypeEnum.chat, ChatBedrock],
        [PromptTypeEnum.completion, BedrockLLM],
        [PromptTypeEnum.embedding, BedrockEmbeddings]
    ])
    def test_load_aws_bedrock_from_prompt_config(self, llm_type: PromptTypeEnum, llm_class):
        config = PromptConfig(type=llm_type, model_name='mistral',
                              parameters=ExecutionParameters(temperature=0.5, top_p=0.5, max_tokens=100))
        result = load_aws_bedrock_from_prompt_config(config)
        self.assertIsInstance(result, llm_class)


if __name__ == '__main__':
    main()
