import re
from typing import List
from pydantic import BaseModel, Field

from rqle_ai_langchain_util.utils.file_util import read_file

SYSTEM_MESSAGE_PATTERN = r'(?:{{#system~}}|<<#system~>>)\n(.*?)\n(?:{{~/system}}|<<~/system>>)'
USER_MESSAGE_PATTERN = r'(?:{{#user~}}|<<#user~>>)\n(.*?)\n(?:{{~/user}}|<<~/user>>)'
ASSISTANT_MESSAGE_PATTERN = r'(?:{{#assistant~}}|<<#assistant~>>)\n(.*?)\n(?:{{~/assistant}}|<<~/assistant>>)'


class PromptTemplate(BaseModel):
    version: int = Field(description='Schema version', default=1)
    prompt: str = Field(description='Prompt', default='')
    system_message: str = Field(description='System message sets the initial context or '
                                            'instructions to the LLM on how to behave.', default='')
    user_messages: List[str] = Field(description='User message represents the input from the user.',
                                     default_factory=list)
    assistant_messages: List[str] = Field(description='Assistant message guides the output from the LLM.',
                                          default_factory=list)

    @staticmethod
    def from_text(prompt_text: str) -> 'PromptTemplate':
        """
        :param prompt_text: Prompt text
        :return: a PromptTemplate object
        """
        prompt_template = PromptTemplate()
        prompt_template.prompt = prompt_text
        # get the system message from the prompt
        system_messages = re.findall(SYSTEM_MESSAGE_PATTERN, prompt_text, re.DOTALL)
        if len(system_messages) > 0:
            prompt_template.system_message = system_messages[0]
        # get the user message from the prompt
        prompt_template.user_messages = re.findall(USER_MESSAGE_PATTERN, prompt_text, re.DOTALL)
        if len(prompt_template.user_messages) == 0:
            prompt_template.user_messages.append(prompt_text)
        # get the assistant message from the prompt
        prompt_template.assistant_messages = re.findall(ASSISTANT_MESSAGE_PATTERN, prompt_text, re.DOTALL)
        return prompt_template

    @staticmethod
    def from_json(template_json_path: str) -> 'PromptTemplate':
        """
        Create a PromptConfig from a JSON string
        :param template_json_path: Prompt configuration as a JSON string
        :return: a PromptConfig object
        """
        return PromptTemplate.model_validate_json(read_file(template_json_path, 'template.json'))
