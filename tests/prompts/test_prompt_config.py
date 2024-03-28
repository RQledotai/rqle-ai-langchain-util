from unittest import TestCase, main

from parameterized import parameterized

from rqle_ai_langchain_util.prompts.prompt_config import PromptConfig
from rqle_ai_langchain_util.settings import PROMPT_CONFIG_FOLDER

class TestPromptConfig(TestCase):

    def test_from_json(self):
        config = PromptConfig.from_json(f'{PROMPT_CONFIG_FOLDER}/clarity_genie')
        self.assertEqual(config.type, 'chat')
        self.assertEqual(config.model_name, 'gemma:7b')

    @parameterized.expand([
        ['invalid/path.json', FileNotFoundError]
    ])
    def test_from_json_error(self, file_path, expected_error):
        with self.assertRaises(expected_error):
            PromptConfig.from_json(file_path)


if __name__ == '__main__':
    main()
