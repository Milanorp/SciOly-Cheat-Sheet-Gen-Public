import os
import sys
import json

# Add the project root to sys.path to allow 'from src.X import Y' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from src.factory import factory, console

def run() -> dict:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    os.makedirs(DATA_DIR, exist_ok=True)

    console.print("\n[phase]PHASE 1.5: CONTEXT CACHE SETUP[/phase]")

    load_dotenv()

    is_vertex = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"))
    
    if is_vertex:
        # Vertex AI Caching Implementation
        console.print("[info]  Vertex AI detected: initializing Vertex Context Caching...[/info]")
        try:
            import vertexai
            from vertexai.preview import caching
            from vertexai.generative_models import Part
            
            project = os.getenv("GOOGLE_CLOUD_PROJECT", "gemini-api-499207")
            location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
            # Caching is only supported in specific regions currently, like us-central1, us-east4, us-east5, europe-west1, europe-west4, europe-west9, asia-southeast1
            # But we are using global which also supports it for preview models.
            vertexai.init(project=project, location=location)
            
            console.print("1. Reading 'extracted_rules.md'...")
            with open("extracted_rules.md", "r", encoding="utf-8") as f:
                rules_content = f.read()
                
            console.print("2. Creating Vertex AI Context Cache (TTL: 60 minutes)...")
            model_name = config['models']['researcher']
            
            cache = caching.CachedContent.create(
                model_name=model_name,
                system_instruction="You are an expert Science Olympiad AI Assistant building a dense cheat sheet.",
                contents=[Part.from_text(rules_content)],
                ttl="3600s" # 60 minutes
            )
            console.print(f"   [success]Cache created! Cache Name: {cache.name}[/success]")

            with open(os.path.join(DATA_DIR, "cache_info.json"), "w", encoding="utf-8") as f:
                json.dump({"cache_name": cache.name}, f)
            return {"cache_name": cache.name}
            
        except Exception as e:
            console.print(f"[error]Error creating Vertex Cache: {e}[/error]")
            return {}
    else:
        # Gemini Developer API path (only reached when using GOOGLE_API_KEY)
        try:
            from google import genai
            from google.genai import types
            api_key = os.getenv("GOOGLE_API_KEY")
            client = genai.Client(api_key=api_key)
            
            console.print("1. Uploading 'extracted_rules.md' to Google Cloud...")
            file = client.files.upload(file="extracted_rules.md")
            console.print(f"   [success]File uploaded. URI: {file.uri}[/success]")

            console.print("\n2. Creating Context Cache (Valid for 60 minutes)...")
            model_name = config['models']['researcher']
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    contents=[file],
                    ttl="3600s"
                )
            )
            console.print(f"   [success]Cache created! Cache Name: {cache.name}[/success]")

            with open(os.path.join(DATA_DIR, "cache_info.json"), "w", encoding="utf-8") as f:
                json.dump({"cache_name": cache.name}, f)
            console.print("\nSUCCESS! Cache is ready for Dispatcher!")
            return {"cache_name": cache.name}
        except Exception as e:
            console.print(f"[error]Error with Gemini Dev API Cache: {e}[/error]")
            return {}

if __name__ == "__main__":
    run()
