from dotenv import load_dotenv

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

from rqle_ai_langchain_util import settings
from rqle_ai_langchain_util.llms.adapters.llm_adapters import LLMAdapter
from rqle_ai_langchain_util.llms.llm_mediator import LLMMediator

# logging configuration
from time import time
import logging
from logging.config import dictConfig
dictConfig(settings.LOG_CONFIG)
logger = logging.getLogger(__name__)

load_dotenv()


class ClarityGenie:

    def info(self):
        """
        :return: the name of the class
        """
        return self.__class__.__name__

    def __init__(self, config_folder: str):
        """
        Initialise the ClarityGenie class
        :param config_folder: the folder with the prompt configuration
        """
        # configure the LLM to be executed
        self.config_folder = config_folder
        self.llm_mediator = LLMMediator(LLMAdapter.OLLAMA_AI, self.config_folder)

    def load_chain(self):
        """
        :return: a configured LLM to be executed
        """
        try:
            # configure the prompt
            prompt = PromptTemplate(template=self.llm_mediator.prompt_template.prompt,
                                    input_variables=['prompt'])
            logger.debug(f'Generated prompt: {prompt}')

            # return the LLM
            return LLMChain(llm=self.llm_mediator.model, prompt=prompt)
        except Exception as e:
            logger.error(f'Error loading chain: {self.config_folder}\n{e}', exc_info=True)
            raise e

    def invoke_chain(self, input_prompt: str):
        """
        :param input_prompt: the input prompt to be revised
        :return: the output from executing the LLM chain
        """
        try:
            # load the chain to be executed
            chain = self.load_chain()
            # execute the chain
            output = chain.invoke({'prompt': input_prompt})
            logger.debug(f'Output from ClarityGenie: {output}')
            return output
        except Exception as e:
            logger.error(f'Error executing chain: {self.config_folder}\n{e}', exc_info=True)
            raise e


if __name__ == '__main__':
    start = time()
    try:
        # instantiate and the ClarityGenie
        cg = ClarityGenie('clarity_genie')
        output = cg.invoke_chain("""
        Given a job description in triple backticks for {job_title} at {company}, your goal is to re-write the user's past experience delimited by <<<>>> to match the job-description.

The revised resume should include:
- two to three sentences that shine a line on the most relevant / impressive achievements based on the past 10 years
- rewrite past experience with 3-5 bullet per role to make it more actionable
- showcase my 10 most relevant skills that matches the job description

Do NOT use any information beyond the one provided by past experience.

Format the answer in JSON.
{format_instructions}

job description: ```{job_description}```
past experience: <<<{experience_profile}>>>
Output:
        """)
    except Exception as e:
        pass
    finally:
        # calculate elapsed time
        total_time = round((time() - start) / 60, 3)
        logger.info(f'Elapsed time to execute {cg.info()}: {total_time} min')
