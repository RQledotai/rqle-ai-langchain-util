# Adapters

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

**Note** Some models (e.g. AI21 Jurassic) include additional parameters, such `presence_penalty` or `frequency_penalty`.

## Supported Adapters
* [AWS Bedrock](adapters/aws_bedrock_adapter.ipynb)
* [Azure OpenAI](adapters/azure_openai_adapter.ipynb)
* [Google Gemini](adapters/google_gemini_adapter.ipynb)
* [OCI Generative AI Service](adapters/oci_adapter.ipynb)
* [Ollama](adapters/ollama_adapter.ipynb)

The table below provides insights on the type of operations supported by each LLM provider:

| LLM Provider               | Chat | Completion | Embeddings |
|----------------------------|------|------------|------------|
| AWS Bedrock                | ✅    | ✅          | ✅          |
| Microsoft Azure OpenAI     | ✅    | ✅          | ✅          |
| Google AI Studio           | ✅    | ✅          | ✅          |
| OCI Generative AI Services | ❌*   | ✅          | ✅          |
| Ollama                     | ✅    | ✅          | ✅          |

* ⚠️**Note** This is mostly caused by the fact that no chat interface have yet been created for this LLM provider.

As this library primarily focuses on the configuration of various LLMs, they support the LangChain Runnable interface, which comes with default implementations of all methods, i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`. This gives all LLMs basic support for async, streaming and batch as provided by the [LangChain library](https://python.langchain.com/docs/integrations/llms/).

<div style="text-align: center;">
  <hr/>
  <img src="../../img/rqle_ai_logo_alt.jpeg" alt="RQle.AI" width="60"/>
  &nbsp; RQle.AI - 2024
</div>