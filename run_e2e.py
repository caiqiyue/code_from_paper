"""Run end-to-end experiment with CUDA (fast)."""
import sys
import os
import logging
import time

log_file = open("e2e_cuda_output.log", "w", buffering=1)

class FlushFile:
    def __init__(self, f):
        self.f = f
    def write(self, x):
        self.f.write(x)
        self.f.flush()
    def flush(self):
        self.f.flush()
    def __getattr__(self, name):
        return getattr(self.f, name)

sys.stdout = FlushFile(sys.stdout)
sys.stderr = FlushFile(sys.stderr)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    stream=sys.stdout
)

os.chdir(r"D:\学习记录\导师项目\研究\caiqiyue_file")
sys.path.insert(0, r"D:\学习记录\导师项目\研究\caiqiyue_file")

print("=" * 60, flush=True)
print("STARTING END-TO-END EXPERIMENT (CUDA MODE)", flush=True)
print("=" * 60, flush=True)

try:
    import torch
    print(f"CUDA: {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB", flush=True)

    from thesis_platform.core.pipeline import run_pipeline
    config_path = "thesis_platform/configs/experiments/validation/windows_real_test_minimal.yaml"
    print(f"Config: {config_path}", flush=True)

    t0 = time.time()
    result = run_pipeline(config_path)
    elapsed = time.time() - t0

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.1f}s", flush=True)
    print(f"Result keys: {list(result.keys()) if hasattr(result, 'keys') else result}", flush=True)
    print("=" * 60, flush=True)

    log_file.write(f"\nSUCCESS in {elapsed:.1f}s\n")
    log_file.close()
except Exception as e:
    import traceback
    traceback.print_exc()
    log_file.write(f"\nERROR: {e}\n")
    log_file.close()
    sys.exit(1)
