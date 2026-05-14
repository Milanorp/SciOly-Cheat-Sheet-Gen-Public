import os
import yaml
import tenacity
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rich.console import Console
from rich.theme import Theme

load_dotenv()

# Define a professional theme for the entire pipeline
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "phase": "bold magenta",
    "topic": "bold white on blue",
    "grade_a": "bold green",
    "grade_f": "bold red"
})

console = Console(theme=custom_theme)

class ModelFactory:
    """Centralized factory for creating pre-configured LLM and Embedding objects."""
    
    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize Tracing if enabled and API key exists
        if self.config['models'].get('tracing_enabled', False):
            if os.getenv("LANGCHAIN_API_KEY"):
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "SciOly-Gen-v2")
                # console.print("[info]LangSmith Tracing Enabled.[/info]")
            
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
