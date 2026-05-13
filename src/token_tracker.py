import json
import os
import threading
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

LOG_FILE = "token_usage_log.json"
_log_lock = threading.Lock()

class TokenTrackerCallback(BaseCallbackHandler):
    """
    A custom LangChain callback handler that intercepts every LLM response,
    extracts the exact token usage, and logs it to a central JSON file.
    """
    def __init__(self, script_name="Unknown Script"):
        self.script_name = script_name

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """This function automatically triggers the moment Gemini finishes sending a response."""
        input_tokens = 0
        output_tokens = 0
        
        try:
            # Attempt to extract usage from LangChain's standardized usage_metadata
            message = response.generations[0][0].message
            if hasattr(message, 'usage_metadata') and message.usage_metadata:
                input_tokens = message.usage_metadata.get('input_tokens', 0)
                output_tokens = message.usage_metadata.get('output_tokens', 0)
        except Exception:
            pass

        # Fallback for some models/versions that put it in llm_output
        if input_tokens == 0 and response.llm_output and 'token_usage' in response.llm_output:
            input_tokens = response.llm_output['token_usage'].get('prompt_tokens', 0)
            output_tokens = response.llm_output['token_usage'].get('completion_tokens', 0)

        # If we successfully captured tokens, log them!
        if input_tokens > 0 or output_tokens > 0:
            self._log_to_file(input_tokens, output_tokens)

    def _log_to_file(self, input_tokens, output_tokens):
        with _log_lock:
            log_data = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_api_calls": 0,
                "estimated_cost_usd": 0.0,
                "history": []
            }
            
            # Load existing log if it exists
            if os.path.exists(LOG_FILE):
                try:
                    with open(LOG_FILE, "r", encoding="utf-8") as f:
                        log_data = json.load(f)
                except Exception:
                    pass
                    
            # Update totals
            log_data["total_input_tokens"] += input_tokens
            log_data["total_output_tokens"] += output_tokens
            log_data["total_api_calls"] += 1
            
            # Calculate Cost for Gemini 2.5 Flash ($0.30 per 1M input, $2.50 per 1M output)
            cost_in = (log_data["total_input_tokens"] / 1_000_000) * 0.30
            cost_out = (log_data["total_output_tokens"] / 1_000_000) * 2.50
            log_data["estimated_cost_usd"] = round(cost_in + cost_out, 6)
            
            # Keep a brief history of the last 50 calls to avoid huge file sizes
            log_data["history"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "script": self.script_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            })
            
            if len(log_data["history"]) > 50:
                log_data["history"].pop(0)
                
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4)
