from unittest import TestCase, main
from parameterized import parameterized

from pydantic_core import ValidationError

from rqle_ai_langchain_util.prompts.prompt_config import ExecutionParameters, PromptConfig


class TestPromptConfig(TestCase):

    def test_execution_parameters_default(self):
        execution_parameters = ExecutionParameters()
        self.assertEqual(execution_parameters.max_tokens, 2000)
        self.assertEqual(execution_parameters.temperature, 0.0)
        self.assertEqual(execution_parameters.top_p, 0.0)
        self.assertEqual(execution_parameters.presence_penalty, 0.0)
        self.assertEqual(execution_parameters.frequency_penalty, 0.0)

    def test_execution_parameters_temperature_validation(self):
        with self.assertRaises(ValidationError):
            ExecutionParameters(temperature=1.5)

    def test_execution_parameters_top_p_validation(self):
        with self.assertRaises(ValidationError):
            ExecutionParameters(top_p=1.5)

    def test_execution_parameters_presence_penalty_validation(self):
        with self.assertRaises(ValidationError):
            ExecutionParameters(presence_penalty=1.5)

    def test_execution_parameters_frequency_penalty_validation(self):
        with self.assertRaises(ValidationError):
            ExecutionParameters(frequency_penalty=1.5)

    def test_prompt_config_default(self):
        prompt_config = PromptConfig(model_name='test_model', parameters=ExecutionParameters())
        self.assertEqual(prompt_config.version, 1)
        self.assertEqual(prompt_config.type, 'completion')
        self.assertEqual(prompt_config.deployment_name, '')

    def test_prompt_config_type_validation(self):
        with self.assertRaises(ValidationError):
            PromptConfig(type='invalid', model_name='test_model', parameters=ExecutionParameters())

    def test_from_json(self):
        config = PromptConfig.from_json(f'tests/resources/prompt_config')
        self.assertEqual(config.type, 'chat')
        self.assertEqual(config.model_name, 'model')

    def test_from_json_parameters(self):
        config = PromptConfig.from_json(f'tests/resources/prompt_config')
        self.assertEqual(config.parameters.max_tokens, 2000)
        self.assertEqual(config.parameters.temperature, 0.3)
        self.assertEqual(config.parameters.top_p, 0.0)
        self.assertEqual(config.parameters.presence_penalty, 0.0)
        self.assertEqual(config.parameters.frequency_penalty, 0.0)

    @parameterized.expand([
        ['invalid/path.json', FileNotFoundError]
    ])
    def test_from_json_error(self, file_path, expected_error):
        with self.assertRaises(expected_error):
            PromptConfig.from_json(file_path)


if __name__ == '__main__':
    main()
