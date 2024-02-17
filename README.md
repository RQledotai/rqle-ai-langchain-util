# langchain-util

- [Description](#description)
- [Development](#development)
  - [Requirements](#requirements)
  - [How to prepare the environment](#how-to-prepare-the-environment) 

## Description <a name="description"></a>
[LangChain](https://www.langchain.com/langchain) is a framework for developing applications powered by language models. 
It enables application that:
* **Are context aware**; i.e. connect a language model to sources context
* **Reason**; i.e. rely on a language model to reason

This library extends LangChain by providing facilities to define configurations, templates for execution of LLMs.

## Development <a name="development"></a>

### Requirements <a name="requirements"></a>
* Git
* Python >= 3.11
* Poetry >= 1.7.0

### How to prepare the environment <a name="how-to-prepare-the-environment"></a>
* Install dependencies
  ```
  poetry install
  ```
  ---
  **NOTE**
  To update dependencies, it may be needed to run the following command prior to installing the packages:
  ```
  poetry lock
  ```
  ---
* Test unit test coverage for the project
  ```
  poetry run coverage run -m pytest && poetry run coverage report -m
  ```
  **Note** Report is only generated if all unit test have completed successfully.
