"""Run pipeline in stages to debug."""
import sys
import os
os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")
sys.path.insert(0, r"D:\学习记录\导师项目\研究\caiqiyue_file")

import logging
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

print("=" * 60, flush=True)
print("STAGE 1: Load config", flush=True)
print("=" * 60, flush=True)
from thesis_platform.core.config import load_experiment_config
config_path = "thesis_platform/configs/experiments/validation/windows_real_test_minimal.yaml"
config = load_experiment_config(config_path)
print(f"Config loaded: {config.meta.get('experiment_id')}", flush=True)

print()
print("=" * 60, flush=True)
print("STAGE 2: Build text backends", flush=True)
print("=" * 60, flush=True)
from thesis_platform.models.backends import build_text_backend
repo_root = config.repo_root()
print(f"repo_root: {repo_root}", flush=True)

llm_cfg = config.llm
client_cfg = dict(llm_cfg.get("client", {}))
server_cfg = dict(llm_cfg.get("server", {}))
print(f"Client cfg: {client_cfg.get('model_name_or_path')}", flush=True)
print(f"Server cfg: {server_cfg.get('model_name_or_path')}", flush=True)

print("Building client backend...", flush=True)
client_backend = build_text_backend({**client_cfg, "role": "client"}, repo_root=repo_root)
print(f"Client backend: {client_backend.backend_name}", flush=True)

print("Building server backend...", flush=True)
server_backend = build_text_backend({**server_cfg, "role": "server"}, repo_root=repo_root)
print(f"Server backend: {server_backend.backend_name}", flush=True)

print()
print("=" * 60, flush=True)
print("STAGE 3: Build embedder", flush=True)
print("=" * 60, flush=True)
from thesis_platform.models.embedding import build_embedder
embedder = build_embedder(config.retriever.get("embedding_model"), device="cpu")
print(f"Embedder built", flush=True)

print()
print("=" * 60, flush=True)
print("STAGE 4: Load dataset", flush=True)
print("=" * 60, flush=True)
from thesis_platform.data.loaders import load_samples
from thesis_platform.data.partition import partition_samples

train_path = config.resolve_path(config.data.get("train_path"))
print(f"Loading train samples from {train_path}", flush=True)
train_samples = load_samples(train_path)
print(f"Loaded {len(train_samples)} train samples", flush=True)

clients = partition_samples(
    train_samples,
    num_clients=2,
    max_samples_per_client=2,
    seed=42,
)
print(f"Partitioned into {len(clients)} clients", flush=True)
for cid, samples in clients.items():
    print(f"  Client {cid}: {len(samples)} samples", flush=True)

print()
print("=" * 60, flush=True)
print("ALL STAGES PASSED", flush=True)
print("=" * 60, flush=True)
