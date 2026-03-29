"""Test 4-bit quantization loading."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing 4-bit quantization loading...")
try:
    from thesis_platform.models.backends import build_text_backend
    from pathlib import Path

    repo_root = Path("thesis_platform").resolve().parent

    # Test server model with 4-bit quantization + device_map="auto"
    backend = build_text_backend(
        engine_or_config={
            "engine": "transformers",
            "model_name_or_path": "thesis_platform/open_model/llama_2_13b_chat_hf",
            "device": "auto",
            "dtype": "auto",
            "load_in_4bit": True,
        },
        repo_root=repo_root,
    )
    print(f"Backend created: {backend.backend_name}")
    result = backend.generate("Hello world", max_new_tokens=10)
    print(f"Generation: {repr(result)}")
    print("QUANTIZATION TEST PASSED")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
