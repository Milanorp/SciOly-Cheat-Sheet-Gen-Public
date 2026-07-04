import os
import warnings
import yaml
import tenacity
from dotenv import load_dotenv
from rich.console import Console
from rich.theme import Theme

# Suppress the LangChain ChatVertexAI deprecation warning.
# The warning incorrectly suggests switching to langchain-google-genai,
# but ChatVertexAI is still the correct class for Vertex AI (not the Developer API).
warnings.filterwarnings(
    "ignore",
    message=".*ChatVertexAI.*deprecated.*",
    category=DeprecationWarning,
)

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_google_vertexai.model_garden_maas import (
    get_vertex_maas_model,
    _MAAS_MODELS,
)
from langchain_google_vertexai.model_garden import ChatAnthropicVertex

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
    """Centralized factory for creating pre-configured LLM and Embedding objects.

    Supports multi-provider routing via Vertex AI Model Garden:
      - Google Gemini      -> ChatVertexAI  (gemini-3.1-pro, gemini-3.5-flash, etc.)
      - Anthropic Claude   -> ChatVertexAI  (claude-sonnet-4-6, claude-opus-4-7, etc.)
      - Meta Llama MaaS    -> get_vertex_maas_model  (meta/llama-4-maverick-...-maas)
      - Mistral MaaS       -> get_vertex_maas_model  (mistral-medium-3, mistral-small-2503)

    All routing is automatic based on the model name prefix in settings.yaml.
    Change any phase's model by editing config/settings.yaml — no code changes needed.
    """

    def __init__(self, config_path="config/settings.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # Initialize LangSmith tracing if enabled
        if self.config['models'].get('tracing_enabled', False):
            if os.getenv("LANGCHAIN_API_KEY"):
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "SciOly-Gen-v2")

    def get_config(self):
        return self.config

    def is_rate_limit(self, exception):
        err_str = str(exception).lower()
        return "429" in err_str or "quota" in err_str or "exhausted" in err_str

    def _detect_provider(self, model_name: str) -> str:
        """Auto-detect the Vertex routing strategy from the model name."""
        if model_name in _MAAS_MODELS:
            return "maas"   # Llama or Mistral — use get_vertex_maas_model
        if "claude" in model_name.lower():
            return "anthropic"
        return "vertex"     # Gemini — use ChatVertexAI

    def get_llm(self, purpose="researcher", cache_info=None):
        """Returns a configured LLM instance routed to the correct Vertex backend.

        Routing logic (auto-detected from model name):
          - Llama / Mistral MaaS models -> get_vertex_maas_model
          - Gemini / Claude on Vertex   -> ChatVertexAI
        """
        # Per-phase temperature map
        temp_map = {
            "architect": self.config['models']['temperature_architect'],
            "researcher": self.config['models']['temperature_researcher'],
            "auditor":    self.config['models']['temperature_auditor'],
            "compiler":   self.config['models']['temperature_compiler'],
        }
        temperature = temp_map.get(purpose, 0.2)

        # Per-phase model selection
        model_map = {
            "architect": self.config['models']['architect'],
            "researcher": self.config['models']['researcher'],
            "auditor":    self.config['models']['auditor'],
            "compiler":   self.config['models']['compiler'],
        }
        model_name = model_map.get(purpose, self.config['models']['researcher'])

        project  = os.getenv("GOOGLE_CLOUD_PROJECT", "gemini-api-499207")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        provider = self._detect_provider(model_name)

        console.print(f"[info]  [{purpose.upper()}] {model_name} ({provider})[/info]")

        if provider == "maas":
            # Llama 4 / Llama 3.3 / Mistral — fully managed serverless endpoints
            return get_vertex_maas_model(
                model_name,
                project=project,
                location=location,
            )
        elif provider == "anthropic":
            # Route Anthropic models to supported Vertex AI regions
            claude_location = "us-east5" if location == "us-central1" else location
            return ChatAnthropicVertex(
                model_name=model_name,
                project=project,
                location=claude_location,
                temperature=temperature,
                max_retries=5,
            )
        else:
            # Gemini (all versions) on Vertex Model Garden
            kwargs = {
                "model_name": model_name,
                "project": project,
                "location": location,
                "temperature": temperature,
                "max_retries": 5,
            }
            if cache_info and "cache_name" in cache_info:
                console.print(f"[info]  -> Using Context Cache: {cache_info['cache_name']}[/info]")
                kwargs["cached_content"] = cache_info["cache_name"]
            
            return ChatVertexAI(**kwargs)

    def get_embeddings(self):
        model_name = self.config['models']['embedding']
        
        if model_name.startswith("local/"):
            # Use free local embeddings
            hf_model_name = model_name.split("local/")[1]
            console.print(f"[info]  [EMBEDDING] Local HuggingFace ({hf_model_name})[/info]")
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name=hf_model_name)
            except ImportError:
                console.print("[error]  Missing langchain-huggingface! Falling back to Vertex AI.[/error]")
                model_name = "text-embedding-004"
                
        project  = os.getenv("GOOGLE_CLOUD_PROJECT", "gemini-api-499207")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        console.print(f"[info]  [EMBEDDING] Vertex AI ({model_name})[/info]")
        return VertexAIEmbeddings(
            model_name=model_name,
            project=project,
            location=location,
        )

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
