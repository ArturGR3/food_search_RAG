from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(usecwd=True))

class LLMProviderSettings(BaseSettings):
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 2

class OpenAISettings(LLMProviderSettings):
    api_key: str = os.getenv("OPENAI_API_KEY")
    default_model: str = "gpt-4o-mini"

class AnthropicSettings(LLMProviderSettings):
    api_key: str = os.getenv("ANTHROPIC_API_KEY")
    default_model: str = "claude-3-5-sonnet-20240620"
    max_tokens: int = 1024

class LlamaSettings(LLMProviderSettings):
    api_key: str = "key"  # required, but not used
    default_model: str = "llama3"
    base_url: str = "http://localhost:11434/v1"

class GroqSettings(LLMProviderSettings):
    api_key: str = os.getenv("GROQ_API_KEY")
    default_model: str = "mixtral-8x7b-32768"

class Settings(BaseSettings):
    app_name: str = "GenAI Project Template"
    openai: OpenAISettings = OpenAISettings()
    anthropic: AnthropicSettings = AnthropicSettings()
    llama: LlamaSettings = LlamaSettings()
    groq: GroqSettings = GroqSettings()

@lru_cache
def get_settings():
    return Settings()