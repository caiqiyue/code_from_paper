"""Debug the full pipeline step by step."""
import sys, os
os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")
sys.path.insert(0, r"D:\学习记录\导师项目\研究\caiqiyue_file")

import torch
print(f"CUDA: {torch.cuda.is_available()}", flush=True)

from thesis_platform.core.config import load_experiment_config
from thesis_platform.core.experiment_runner import ExperimentRunner

config_path = "thesis_platform/configs/experiments/validation/windows_real_test_minimal.yaml"
config = load_experiment_config(config_path)

runner = ExperimentRunner(config)
print("ExperimentRunner created", flush=True)

# Build text backends
print("Building text backends...", flush=True)
client_backend, server_backend = runner._build_text_backends()
print(f"Text backends OK: client={client_backend.backend_name}, server={server_backend.backend_name}", flush=True)

# Load public seeds
print("Loading public seeds...", flush=True)
public_seed_samples = runner._load_public_seed_samples()
print(f"Seeds loaded: {len(public_seed_samples)}", flush=True)

# Load client contexts
print("Loading client contexts...", flush=True)
client_contexts = runner._load_client_contexts(client_backend=client_backend)
print(f"Client contexts: {len(client_contexts)}", flush=True)

# Run the experiment
print("Running experiment...", flush=True)
result = runner.run()
print(f"Result keys: {list(result.keys()) if hasattr(result, 'keys') else result}", flush=True)
print("ALL PASSED!", flush=True)
