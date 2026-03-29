"""Run minimal windows_real_test experiment end-to-end."""
import sys
import os
import traceback
import logging

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Set up logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    stream=sys.stdout
)

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

print("=" * 60, flush=True)
print("STARTING END-TO-END EXPERIMENT", flush=True)
print("=" * 60, flush=True)

try:
    from thesis_platform.core.pipeline import run_pipeline

    config_path = "thesis_platform/configs/experiments/validation/windows_real_test_minimal.yaml"
    print(f"Config: {config_path}", flush=True)
    print(f"Config exists: {os.path.exists(config_path)}", flush=True)
    print(flush=True)

    print("=== Running pipeline ===", flush=True)
    result = run_pipeline(config_path)
    print(flush=True)
    print("=== Pipeline result received ===", flush=True)
    print(f"Result keys: {result.keys() if hasattr(result, 'keys') else result}", flush=True)
    print(flush=True)
    print("SUCCESS: Full pipeline completed", flush=True)
except Exception as e:
    print(flush=True)
    print("=== ERROR ===", flush=True)
    traceback.print_exc()
    sys.exit(1)
