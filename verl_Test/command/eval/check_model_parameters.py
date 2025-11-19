#!/usr/bin/env python3
"""
Check if a model checkpoint has specific parameters (e.g., proj_z) in safetensors files
Usage: python check_model_parameters.py <model_path> [search_term]
"""

import sys
import glob
from pathlib import Path
import warnings

def load_model_with_transformers(model_path):
    """
    Load model using HuggingFace transformers to see warnings and errors

    Args:
        model_path: Path to the model checkpoint
    """
    print("="*60)
    print("Loading model with HuggingFace Transformers...")
    print("="*60)
    print("")

    try:
        from transformers import AutoModelForCausalLM, AutoConfig
        import torch

        print(f"Model path: {model_path}")
        print("Loading with low_cpu_mem_usage=True, torch_dtype=auto...")
        print("")

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                device_map="cpu"  # Keep on CPU to avoid GPU memory usage
            )

            # Print any warnings
            if w:
                print(f"⚠️  Caught {len(w)} warning(s) during model loading:")
                print("-"*60)
                for warning in w:
                    print(f"  {warning.category.__name__}: {warning.message}")
                print("")
            else:
                print("✓ No warnings during model loading")
                print("")

        # Print model info
        print("Model loaded successfully!")
        print(f"Model type: {type(model).__name__}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("")

        # Clean up
        del model
        torch.cuda.empty_cache()

        print("="*60)
        print("")

    except ImportError:
        print("✗ Error: transformers or torch not installed")
        print("Install with: pip install transformers torch")
        print("")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("")

def check_model_parameters(model_path, search_term="proj_z"):
    """
    Check safetensors files for parameters matching the search term

    Args:
        model_path: Path to the model checkpoint
        search_term: Parameter name to search for (default: "proj_z")
    """
    print("="*60)
    print(f"Checking checkpoint files in: {model_path}")
    print("="*60)

    try:
        from safetensors import safe_open

        model_path = Path(model_path)

        # Find all safetensors files
        safetensor_files = list(model_path.glob("*.safetensors"))

        if not safetensor_files:
            print("✗ No safetensors files found!")
            print("This model might use .bin format (PyTorch) instead.")
            return

        print(f"Found {len(safetensor_files)} safetensors file(s)")
        print("")

        # Collect all keys from all files
        all_keys = []
        found_params = []

        for file in safetensor_files:
            print(f"Checking: {file.name}")
            with safe_open(file, framework="pt") as f:
                file_keys = list(f.keys())
                all_keys.extend(file_keys)

                # Search in this file
                matching = [k for k in file_keys if search_term.lower() in k.lower()]
                if matching:
                    for key in matching:
                        tensor = f.get_tensor(key)
                        found_params.append((key, tensor.shape, file.name))

        print("")
        print("="*60)
        print(f"Total parameters in checkpoint: {len(all_keys)}")
        print("="*60)
        print("")

        # Report results
        print(f"Searching for parameters containing '{search_term}'...")
        print("-"*60)

        if found_params:
            print(f"✓ Found {len(found_params)} parameter(s) matching '{search_term}':")
            print("")
            for name, shape, filename in found_params:
                print(f"  - Name: {name}")
                print(f"    Shape: {shape}")
                print(f"    File: {filename}")
                print("")
        else:
            print(f"✗ No parameters found containing '{search_term}'")
            print("")
            print("First 20 parameter names in checkpoint:")
            print("-"*60)
            for i, key in enumerate(all_keys[:20]):
                print(f"  {i+1}. {key}")
            if len(all_keys) > 20:
                print(f"  ... and {len(all_keys) - 20} more")

        print("")
        print("="*60)
        print("Check complete!")
        print("="*60)

    except ImportError:
        print("✗ Error: safetensors library not installed")
        print("Install with: pip install safetensors")
    except Exception as e:
        print(f"✗ Error checking checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Using provided model path: {model_path}")
        print("")
    else:
        print("Usage: python check_model_parameters.py <model_path> [search_term]")
        print("Example: python check_model_parameters.py /path/to/model proj_z")
        sys.exit(1)

    # Optional: search for different parameter name
    search_term = sys.argv[2] if len(sys.argv) > 2 else "proj_z"

    print(f"Searching for parameters containing: '{search_term}'")
    print("")

    # First, try loading the model with HuggingFace to see warnings
    load_model_with_transformers(model_path)

    # Then, check the actual safetensors files
    check_model_parameters(model_path, search_term)
