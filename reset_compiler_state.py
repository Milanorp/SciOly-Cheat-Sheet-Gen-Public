import json
import os

state_path = "pipeline_data/pipeline_state.json"
if os.path.exists(state_path):
    with open(state_path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    state["current_phase"] = 3.0
    
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=4)
        
    print("Pipeline state successfully reset to Phase 3 (Compiler)!")
else:
    print("Error: pipeline_state.json not found.")
