import os
import json
from src.models import PipelineState
from src.factory import factory

class StateManager:
    """Manages the persistence and recovery of the pipeline state."""
    
    def __init__(self):
        self.config = factory.get_config()
        self.data_dir = self.config['paths']['data_dir']
        self.state_file = os.path.join(self.data_dir, "pipeline_state.json")
        os.makedirs(self.data_dir, exist_ok=True)
        
    def load_state(self) -> PipelineState:
        """Loads the state from disk or returns a fresh state."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return PipelineState(**data)
            except Exception as e:
                print(f"⚠️ Warning: Could not load state file: {e}. Starting fresh.")
        
        return PipelineState(current_phase=0.0)

    def save_state(self, state: PipelineState):
        """Saves the current pipeline state to disk."""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                f.write(state.model_dump_json(indent=4))
        except Exception as e:
            print(f"❌ Error saving state: {e}")

    def clear_state(self):
        """Removes the state file."""
        if os.path.exists(self.state_file):
            os.remove(self.state_file)

# Singleton instance
state_manager = StateManager()
