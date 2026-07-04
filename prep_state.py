import json
import os

data_dir = "pipeline_data"
os.makedirs(data_dir, exist_ok=True)

# Delete old files to start fresh
for f in ["pipeline_state.json", "cheat_sheet_blueprint.json", "raw_research_notes.json", "cache_info.json"]:
    p = os.path.join(data_dir, f)
    if os.path.exists(p):
        os.remove(p)

# Load frequency data
freq_data = None
freq_path = os.path.join(data_dir, "test_frequency_map.json")
if os.path.exists(freq_path):
    with open(freq_path, 'r', encoding='utf-8') as f:
        freq_data = json.load(f)

# Create a fresh pipeline state for Potions and Poisons
# We set current_phase to 1.0 to skip Phase 0 (test crunching)
state = {
    "current_phase": 1.0,
    "event_name": "Potions and Poisons",
    "frequency_data": freq_data,
    "blueprint": None,
    "research_notes": None,
    "cache_info": None,
    "failed_topics": [],
    "retry_count": 0
}

state_path = os.path.join(data_dir, "pipeline_state.json")
with open(state_path, 'w', encoding='utf-8') as f:
    json.dump(state, f, indent=4)

print("Fresh pipeline state for Potions and Poisons prepared successfully!")
