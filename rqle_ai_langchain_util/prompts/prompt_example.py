from typing import List
from pydantic import BaseModel, Field

from rqle_ai_langchain_util.utils.file_util import read_file


class PromptExample(BaseModel):
    version: int = Field(description='Schema version', default=1)
    examples: List[dict] = Field(description='Examples of the prompt', default_factory=list)

    @staticmethod
    def from_json(example_json_path: str) -> 'PromptExample':
        """
        Create a PromptConfig from a JSON string
        :param example_json_path: Prompt configuration as a JSON string
        :return: a PromptConfig object
        """
        return PromptExample.model_validate_json(read_file(example_json_path, 'example.json'))