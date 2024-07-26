from enum import Enum


class LLMAdapter(str, Enum):
    AWS_BEDROCK = "aws_bedrock_adapter"
    AZURE_OPENAI = "azure_openai_adapter"
    GOOGLE_GEMINI = "google_gemini_adapter"
    OLLAMA_AI = "ollama_adapter"
    OCI_AI = "oci_ai_adapter"

    __all__ = [
        AWS_BEDROCK,
        AZURE_OPENAI,
        GOOGLE_GEMINI,
        OLLAMA_AI,
        OCI_AI
    ]
