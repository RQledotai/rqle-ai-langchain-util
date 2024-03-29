# Introduction
This library integrate with the following LLM providers:
* [AWS Bedrock](https://us-west-2.console.aws.amazon.com/bedrock/home)
* [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
* [Google AI Studio](https://aistudio.google.com/)
* [ollama](https://ollama.com/) 
* [Oracle Cloud Infrastructure AI](https://www.oracle.com/artificial-intelligence/ai-services/)

All LLMs implement the Runnable interface, which comes with default implementations of all methods, ie. ainvoke, batch, abatch, stream, astream. This gives all LLMs basic support for async, streaming and batch as provided by the [LangChain library](https://python.langchain.com/docs/integrations/llms/).

## Get started
This library relies on a configuration folder that contains:
* a JSON file with the configuration for the chosen LLM.
* a text file including the prompt to be executed

The JSON format is as follows:
```json
{
  "schema": 1,
  "type": "{type}",
  "model_name": "{model_name}",
  "parameters": {
    "max_tokens": 2000,
    "temperature": 0.3,
    "top_p": 0.0
  }
}
```
For most providers, our library supports "chat", "completion" and "embeddings" as `type` of LLMs. Similarly, `model_name` should be replaced by the name of the LLM supported by the provider (e.g. 'gemma:7b').

**Note** Some models include additional parameters, such `presence_penalty` or `frequency_penalty`.

## Supported Adapters
* [OCI Generative AI Service](adapters/oci_adapter.ipynb)