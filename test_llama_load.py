import sys
import os
import yaml
import torch

# Ensure current directory is in path for imports
sys.path.append(os.getcwd())

from generated_texts_og.models.model_interface import ModelFactory

def test_load():
    print("Testing Llama 3 70B Loading...")
    
    # Load config manually to simulate engine
    # Assuming run from src/coherence/
    config_path = "generated_texts_og/config/model_config.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    model_config = config["models"]["llama3_70b"]
    print(f"Config: {model_config}")
    
    try:
        model = ModelFactory.create_model(model_config)
        print("✅ Model loaded successfully!")
        
        print("Testing Generation...")
        # Simple prompt
        out = model.generate("The quick brown fox jumps over the lazy dog.", max_new_tokens=20)
        print(f"Output: {out}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
