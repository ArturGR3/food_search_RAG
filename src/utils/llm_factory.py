# Description: This script generates short, user-like prompts for recipe searches using OpenAI's Language Model API.

import sys
import os
from pathlib import Path
# Add the project root directory to the Python path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.append(project_root)

import asyncio
from typing import Type, Any, Dict, List
from pydantic import BaseModel
from src.utils.settings import get_settings
from openai import AsyncOpenAI, OpenAI
from anthropic import AsyncAnthropic, Anthropic
from groq import AsyncGroq, Groq
import instructor

class LLMFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        client_map = {
            "openai": (OpenAI, instructor.from_openai),
            "anthropic": (Anthropic, instructor.from_anthropic),
            "groq": (Groq, lambda client: instructor.from_groq(client, mode=instructor.Mode.TOOLS))
        }
        
        if self.provider not in client_map:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        ClientClass, wrapper = client_map[self.provider]
        return wrapper(ClientClass(api_key=self.settings.api_key))
    
    def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
            "stream": kwargs.get("stream", False),  # Add streaming option, default to False    
        }
        return self.client.chat.completions.create(**completion_params)

class AsyncLLMFactory:
    def __init__(self, provider: str, sem_number: int = 2):
        self.provider = provider
        self.settings = getattr(get_settings(), provider)
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        client_map = {
            "openai": (AsyncOpenAI, instructor.patch),
            "anthropic": (AsyncAnthropic, instructor.patch),
            "groq": (AsyncGroq, lambda client: instructor.patch(client, mode=instructor.Mode.TOOLS))
        }
        
        if self.provider not in client_map:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        ClientClass, wrapper = client_map[self.provider]
        return wrapper(ClientClass(api_key=self.settings.api_key))
    
    async def create_completion(
        self, response_model: Type[BaseModel], messages: List[Dict[str, str]], **kwargs
    ) -> Any:
        completion_params = {
            "model": kwargs.get("model", self.settings.default_model),
            "temperature": kwargs.get("temperature", self.settings.temperature),
            "max_retries": kwargs.get("max_retries", self.settings.max_retries),
            "max_tokens": kwargs.get("max_tokens", self.settings.max_tokens),
            "response_model": response_model,
            "messages": messages,
            "stream": kwargs.get("stream", False),  # Add streaming option, default to False
        }
        return await self.client.chat.completions.create(**completion_params)