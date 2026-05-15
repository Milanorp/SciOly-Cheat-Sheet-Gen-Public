import os
import sys
import json

# Add the project root to sys.path to allow 'from src.X import Y' imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.factory import factory

def run() -> dict:
    config = factory.get_config()
    DATA_DIR = config['paths']['data_dir']
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("PHASE 1.5: UPLOADING RULEBOOK TO GEMINI CACHE")
    print("="*60)

    load_dotenv()
    try:
        client = genai.Client()
    except Exception as e:
        print(f"❌ Error initializing Gemini Client: {e}")
        return {}

    # Upload file
    print("1. Uploading 'extracted_rules.md' to Google Cloud...")
    try:
        file = client.files.upload(file="extracted_rules.md")
        print(f"   ✅ File uploaded. URI: {file.uri}")
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return {}

    # Create cache
    print("\n2. Creating Context Cache (Valid for 24 hours)...")
    try:
        # 86400 seconds = 24 hours
        model_name = config['models']['standard']
        cache = client.caches.create(
            model=model_name,
            config=types.CreateCachedContentConfig(
                contents=[file],
                ttl="86400s"
            )
        )
        print(f"   ✅ Cache created successfully! Cache Name: {cache.name}")
        
        # Save cache name to a local file for checkpointing
        with open(os.path.join(DATA_DIR, "cache_info.json"), "w", encoding="utf-8") as f:
            json.dump({"cache_name": cache.name}, f)
        print("\nSUCCESS! Cache is ready for Dispatcher!")
        return {"cache_name": cache.name}

    except Exception as e:
        print(f"❌ Error creating cache: {e}")
        return {}

if __name__ == "__main__":
    run()
