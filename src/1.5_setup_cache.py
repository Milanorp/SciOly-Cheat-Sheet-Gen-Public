import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

print("\n" + "="*60)
print("PHASE 1.5: UPLOADING RULEBOOK TO GEMINI CACHE")
print("="*60)

load_dotenv()
try:
    client = genai.Client()
except Exception as e:
    print(f"❌ Error initializing Gemini Client: {e}")
    exit(1)

# Upload file
print("1. Uploading 'extracted_rules.md' to Google Cloud...")
try:
    file = client.files.upload(file="extracted_rules.md")
    print(f"   ✅ File uploaded. URI: {file.uri}")
except Exception as e:
    print(f"❌ Error uploading file: {e}")
    exit(1)

# Create cache
print("\n2. Creating Context Cache (Valid for 24 hours)...")
try:
    # 86400 seconds = 24 hours
    cache = client.caches.create(
        model="gemini-2.5-flash",
        config=types.CreateCachedContentConfig(
            contents=[file],
            ttl="86400s"
        )
    )
    print(f"   ✅ Cache created successfully! Cache Name: {cache.name}")
    
    # Save cache name to a local file so Dispatcher can use it
    with open("cache_info.json", "w", encoding="utf-8") as f:
        json.dump({"cache_name": cache.name}, f)
    print("\nSUCCESS! Saved cache info to 'cache_info.json'. You can now run the Dispatcher!")

except Exception as e:
    print(f"❌ Error creating cache: {e}")
