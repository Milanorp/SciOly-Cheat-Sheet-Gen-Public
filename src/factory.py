import os
import yaml
import tenacity
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

class ModelFactory:
    """Centralized factory for creating pre-configured LLM and Embedding objects."""
    
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def get_config(self):
        return self.config

    def is_rate_limit(self, exception):
        err_str = str(exception).lower()
        return "429" in err_str or "quota" in err_str or "exhausted" in err_str

    def get_llm(self, purpose="researcher"):
        """Returns a configured ChatGoogleGenerativeAI instance with retry logic."""
        temp_map = {
            "architect": self.config['models']['temperature_architect'],
            "researcher": self.config['models']['temperature_researcher'],
            "auditor": self.config['models']['temperature_auditor'],
            "compiler": self.config['models']['temperature_compiler']
        }
        
        temperature = temp_map.get(purpose, 0.2)
        model_name = self.config['models']['primary']
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_retries=5
        )

    def get_embeddings(self):
        return GoogleGenerativeAIEmbeddings(model=self.config['models']['embedding'])

    def get_retry_decorator(self, before_sleep_func=None):
        """Returns a standard tenacity retry decorator for API calls."""
        return tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=2, min=4, max=65),
            stop=tenacity.stop_after_attempt(5),
            retry=tenacity.retry_if_exception(self.is_rate_limit),
            before_sleep=before_sleep_func
        )

# Create a singleton instance
factory = ModelFactory()
